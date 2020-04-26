

#include <mitsuba/core/fresolver.h>

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

#include <sstream>

#define DJ_BRDF_IMPLEMENTATION 1
#include "dj_brdf.h"

#include "microfacet.h"

MTS_NAMESPACE_BEGIN


class dj_merl : public BSDF {
public:
	dj_merl(const Properties &props)
		: BSDF(props) {

		m_reflectance = new ConstantSpectrumTexture(props.getSpectrum(
			props.hasProperty("reflectance") ? "reflectance"
				: "diffuseReflectance", Spectrum(.5f)));

		fs::path m_filename = Thread::getThread()->getFileResolver()->resolve(props.getString("filename"));

		// load MERL
		m_brdf = new djb::merl(m_filename.string().c_str());
		// load tabulated
		m_tabular = new djb::tabular(*m_brdf, 90, true);
	}

	dj_merl(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {
		
		m_reflectance = new ConstantSpectrumTexture(Spectrum(.5f));

		int size = stream->readInt();
		auto samples = std::vector<double>();
		for(int i = 0; i < size; i++){
			samples.push_back(stream->readDouble());
		}
		int thetaDResolution = stream->readInt();
		int thetaHResolution = stream->readInt();
		int phiDResolution = stream->readInt();
		m_brdf = new djb::merl(samples, thetaDResolution, thetaHResolution, phiDResolution);
		m_tabular = new djb::tabular(*m_brdf, 90, true);

		configure();
	}

	~dj_merl()
	{
		delete m_brdf;
		delete m_tabular;
	}

	void configure() {
		/* Verify the input parameter and fix them if necessary */
		m_components.clear();	
		m_components.push_back(EDiffuseReflection | EFrontSide | 0);
		m_usesRayDifferentials = false;
		BSDF::configure();
	}

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		djb::vec3 o(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 i(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		djb::vec3 fr_p = m_brdf->evalp(i, o);
		return Color3(fr_p.x, fr_p.y, fr_p.z);
	}

	int getDifferentiableParameters() {
		// Return the number of differentiable parameters.
		// Add one for the diff w.r.t normal
		return m_brdf->getThetaDResolution() *
			   m_brdf->getThetaHResolution() * 
			   (m_brdf->getPhiDResolution()/2) + 1;
	}

	std::vector<std::string> getDifferentiableParameterNames() {
		// MERL parameters are marked 'merl-xx'
		// At the end there is a default derivative w.r.t normal marked 'normal'
		std::vector<std::string> svec;
		int parameterCount = (m_brdf->getThetaDResolution()) *
							 (m_brdf->getThetaHResolution()) * 
							 (m_brdf->getPhiDResolution()/2);
		for(int i = 0; i < parameterCount; i++){
			std::stringstream ss;
			ss << "merl-" << i;
			svec.push_back(ss.str().c_str());
		}
		svec.push_back("normal");
		return svec;
	}

	std::map<int, Spectrum> eval_diff(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		std::map<int, Spectrum> diffs;
		//Log(EInfo, "BRDF size: %dx%dx%d", m_brdf->getThetaDResolution(), m_brdf->getThetaHResolution(), m_brdf->getPhiDResolution()/2);
		int N = m_brdf->getThetaDResolution() * m_brdf->getThetaHResolution() * (m_brdf->getPhiDResolution()/2);
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0) {
			//for(int n=0; n < N+1; n++)
			if (N < 0)
				Log(EError, "Invalid Diff size N %d", N);

			diffs[N] = Spectrum(0.f);
			return diffs;
		}

		djb::vec3 o(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 i(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		auto diffTbale = m_brdf->eval_diff(i, o);
		return diffTbale;
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;

		djb::vec3 o(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 i(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		return m_tabular->pdf(i, o);
	}


	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
			return Spectrum(0.0f);

		/* Sample the tabulated microfacet BRDF */
		djb::vec3 o = djb::vec3(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 i = m_tabular->sample(sample.x, sample.y, o);

		/* Setup Mitsuba variables */
		bRec.wo = Vector(i.x, i.y, i.z);
		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EGlossyReflection;

		/* Side check */
		if (Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		djb::vec3 fr_p = m_brdf->evalp(i, o) / pdf(bRec, ESolidAngle);
		return Color3(fr_p.x, fr_p.y, fr_p.z);

	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf_, const Point2 &sample_) const {
		Spectrum res = sample(bRec, sample_);
		pdf_ = pdf(bRec, ESolidAngle);
		return res;
	}

	void addChild(const std::string &name, ConfigurableObject *child) {
		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))
				&& (name == "reflectance" || name == "diffuseReflectance")) {

		} else {
			BSDF::addChild(name, child);
		}
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);
		std::vector<double> samples = m_brdf->get_samples();
		stream->writeInt(samples.size());
		for (int i = 0; i < samples.size(); i++){
			stream->writeDouble(samples.at(i));
		}
		stream->writeInt(m_brdf->getThetaDResolution());
		stream->writeInt(m_brdf->getThetaHResolution());
		stream->writeInt(m_brdf->getPhiDResolution());
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "dj_merl[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  thetaDResolution = " << m_brdf->getThetaDResolution() << endl
			<< "  thetaHResolution = " << m_brdf->getThetaHResolution() << endl
			<< "  phiDResolution = " << m_brdf->getPhiDResolution() << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_reflectance;
	djb::merl* m_brdf;
	djb::tabular * m_tabular;
};

// ================ Hardware shader implementation ================

class dj_merl_shader : public Shader {
public:
	dj_merl_shader(Renderer *renderer, const Texture *reflectance)
		: Shader(renderer, EBSDFShader), m_reflectance(reflectance) {
		m_reflectanceShader = renderer->registerShaderForResource(m_reflectance.get());
	}

	bool isComplete() const {
		return m_reflectanceShader.get() != NULL;
	}

	void cleanup(Renderer *renderer) {
		renderer->unregisterShaderForResource(m_reflectance.get());
	}

	void putDependencies(std::vector<Shader *> &deps) {
		deps.push_back(m_reflectanceShader.get());
	}

	void generateCode(std::ostringstream &oss,
			const std::string &evalName,
			const std::vector<std::string> &depNames) const {
		oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
			<< "    	return vec3(0.0);" << endl
			<< "    return " << depNames[0] << "(uv) * inv_pi * cosTheta(wo);" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    return " << evalName << "(uv, wi, wo);" << endl
			<< "}" << endl;
	}

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_reflectance;
	ref<Shader> m_reflectanceShader;
};

Shader *dj_merl::createShader(Renderer *renderer) const {
	return new dj_merl_shader(renderer, m_reflectance.get());
}

MTS_IMPLEMENT_CLASS(dj_merl_shader, false, Shader)
MTS_IMPLEMENT_CLASS_S(dj_merl, false, BSDF)
MTS_EXPORT_PLUGIN(dj_merl, "dj_merl BRDF")
MTS_NAMESPACE_END
