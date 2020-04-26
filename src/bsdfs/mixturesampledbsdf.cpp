/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/gpuprogram.h>

MTS_NAMESPACE_BEGIN

/*! \plugin{mixturebsdf}{Mixture material}
 * \order{16}
 * \parameters{
 *     \parameter{weights}{\String}{A comma-separated list of BSDF weights}
 *     \parameter{\Unnamed}{\BSDF}{Multiple BSDF instances that should be
 *     mixed according to the specified weights}
 * }
 * \renderings{
 *     \medrendering{Smooth glass}{bsdf_mixturebsdf_smooth}
 *     \medrendering{Rough glass}{bsdf_mixturebsdf_rough}
 *     \medrendering{An mixture of 70% smooth glass and 30% rough glass
 *     results in a more realistic smooth material with imperfections
 *     (\lstref{mixture-example})}{bsdf_mixturebsdf_result}
 * }
 *
 * This plugin implements a ``mixture'' material, which represents
 * linear combinations of multiple BSDF instances. Any surface scattering
 * model in Mitsuba (be it smooth, rough, reflecting, or transmitting) can
 * be mixed with others in this manner to synthesize new models. There
 * is no limit on how many models can be mixed, but their combination
 * weights must be non-negative and sum to a value of one or less to ensure
 * energy balance. When they sum to less than one, the material will
 * absorb a proportional amount of the incident illlumination.
 *
 * \vspace{4mm}
 * \begin{xml}[caption={A material definition for a mixture of 70% smooth
 *     and 30% rough glass},
 *     label=lst:mixture-example]
 * <bsdf type="mixturebsdf">
 *     <string name="weights" value="0.7, 0.3"/>
 *
 *     <bsdf type="dielectric"/>
 *
 *     <bsdf type="roughdielectric">
 *         <float name="alpha" value="0.3"/>
 *     </bsdf>
 * </bsdf>
 * \end{xml}
 */

class MixedSamplingBSDF : public BSDF {
public:
    MixedSamplingBSDF(const Properties &props)
        : BSDF(props) {
        /* Parse the weight parameter */
        std::vector<std::string> weights =
            tokenize(props.getString("weights", ""), " ,;");
        std::vector<std::string> samples =
            tokenize(props.getString("samples", ""), " ,;");
        if (weights.size() == 0)
            Log(EError, "No weights were supplied!");
        if (samples.size() == 0)
            Log(EWarn, "No sampling weights were supplied. Reverting to the original");
        m_weights.resize(weights.size());
        m_samples.resize(samples.size());

        char *end_ptr = NULL;
        for (size_t i=0; i<weights.size(); ++i) {
            Float weight = (Float) strtod(weights[i].c_str(), &end_ptr);
            Float sampleWt = (Float) strtod(samples[i].c_str(), &end_ptr);
            if (*end_ptr != '\0')
                SLog(EError, "Could not parse the BSDF weights!");
            if (weight < 0)
                SLog(EError, "Invalid BSDF weight!");
            if (sampleWt < 0)
                SLog(EError, "Invalid sampling weight!");
            
            m_weights[i] = weight;
            m_samples[i] = sampleWt;

        }

        m_diffOffsets.push_back(0);
    }

    MixedSamplingBSDF(Stream *stream, InstanceManager *manager)
     : BSDF(stream, manager) {
        size_t bsdfCount = stream->readSize();
        m_weights.resize(bsdfCount);
        m_samples.resize(bsdfCount);
        m_diffOffsets.resize(bsdfCount);
        m_normalIDs.resize(bsdfCount);
        for (size_t i=0; i<bsdfCount; ++i) {
            m_weights[i] = stream->readFloat();
            m_samples[i] = stream->readFloat();
            m_diffOffsets[i] = stream->readInt();
            m_normalIDs[i] = stream->readInt();
            BSDF *bsdf = static_cast<BSDF *>(manager->getInstance(stream));
            bsdf->incRef();
            m_bsdfs.push_back(bsdf);
        }
        configure();
    }

    virtual ~MixedSamplingBSDF() {
        for (size_t i=0; i<m_bsdfs.size(); ++i)
            m_bsdfs[i]->decRef();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        stream->writeSize(m_bsdfs.size());
        for (size_t i=0; i<m_bsdfs.size(); ++i) {
            stream->writeFloat(m_weights[i]);
            stream->writeFloat(m_samples[i]);
            stream->writeInt(m_diffOffsets[i]);
            stream->writeInt(m_normalIDs[i]);
            manager->serialize(stream, m_bsdfs[i]);
        }
    }

    void configure() {
        m_usesRayDifferentials = false;
        size_t componentCount = 0;

        if (m_bsdfs.size() != m_weights.size())
            Log(EError, "BSDF count mismatch: " SIZE_T_FMT " bsdfs, but specified " SIZE_T_FMT " weights",
                m_bsdfs.size(), m_weights.size());
        
        if (m_bsdfs.size() != m_samples.size())
            Log(EError, "BSDF count mismatch: " SIZE_T_FMT " bsdfs, but specified " SIZE_T_FMT " sample weights",
                m_bsdfs.size(), m_samples.size());

        Float totalWeight = 0;
        for (size_t i=0; i<m_weights.size(); ++i)
            totalWeight += m_weights[i];

        if (totalWeight <= 0)
            Log(EError, "The weights must sum to a value greater than zero!");

        Float totalSampleWt = 0;
        for (size_t i=0; i<m_samples.size(); ++i)
            totalSampleWt += m_samples[i];

        if (totalSampleWt <= 0)
            Log(EError, "The sample weights must sum to a value greater than zero!");

        if (m_ensureEnergyConservation && totalWeight > 1) {
            std::ostringstream oss;
            Float scale = 1.0f / totalWeight;
            oss << "The BSDF" << endl << toString() << endl
                << "potentially violates energy conservation, since the weights "
                << "sum to " << totalWeight << ", which is greater than one! "
                << "They will be re-scaled to avoid potential issues. Specify "
                << "the parameter ensureEnergyConservation=false to prevent "
                << "this from happening.";
            Log(EWarn, "%s", oss.str().c_str());
            for (size_t i=0; i<m_weights.size(); ++i)
                m_weights[i] *= scale;
        }

        Float sampleScale = 1.0f / totalSampleWt;
        for (size_t i=0; i<m_samples.size(); ++i)
                m_samples[i] *= sampleScale;

        for (size_t i=0; i<m_bsdfs.size(); ++i)
            componentCount += m_bsdfs[i]->getComponentCount();

        m_pdf = DiscreteDistribution(m_bsdfs.size());
        m_components.reserve(componentCount);
        m_components.clear();
        m_indices.reserve(componentCount);
        m_indices.clear();
        m_offsets.reserve(m_bsdfs.size());
        m_offsets.clear();

        int offset = 0;
        for (size_t i=0; i<m_bsdfs.size(); ++i) {
            const BSDF *bsdf = m_bsdfs[i];
            m_offsets.push_back(offset);

            for (int j=0; j<bsdf->getComponentCount(); ++j) {
                int componentType = bsdf->getType(j);
                m_components.push_back(componentType);
                m_indices.push_back(std::make_pair((int) i, j));
            }

            offset += bsdf->getComponentCount();
            m_usesRayDifferentials |= bsdf->usesRayDifferentials();
            //m_pdf.append(m_weights[i]);
            m_pdf.append(m_samples[i]);
        }
        m_pdf.normalize();
        BSDF::configure();
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        Spectrum result(0.0f);

        if (bRec.component == -1) {
            for (size_t i=0; i<m_bsdfs.size(); ++i)
                result += m_bsdfs[i]->eval(bRec, measure) * m_weights[i];
        } else {
            /* Pick out an individual component */
            int idx = m_indices[bRec.component].first;
            BSDFSamplingRecord bRec2(bRec);
            bRec2.component = m_indices[bRec.component].second;
            return m_bsdfs[idx]->eval(bRec2, measure) * m_weights[idx];
        }

        return result;
    }

    virtual int getDifferentiableParameters() {
        int num = 0;
        for(int i = 0; i < m_bsdfs.size(); i++) {
            num += m_bsdfs[i]->getDifferentiableParameters();
        }
        return num + m_bsdfs.size() + 1;
    }

    /**
    * \brief Return the tags/names of the parameters that can be differentiated
    * This is optional. The names are merely for convenience.
    */
    virtual std::vector<std::string> getDifferentiableParameterNames() {
        std::vector<std::string> slist;
        for(int i = 0; i < m_weights.size(); i++) {
            std::ostringstream ss;
            ss << "weight" << i;
            slist.push_back(ss.str());
        }

        for(int i = 0; i < m_weights.size(); i++) {
            auto pslist = m_bsdfs[i]->getDifferentiableParameterNames();
            Log(EInfo, "Differentiable component: %d", m_diffOffsets[i]);
            for(int j = 0; j < pslist.size(); j++) {
                std::ostringstream ss;
                ss << "component" << i << ":" << pslist[j];
                slist.push_back(ss.str());
                Log(EInfo, "BSDF Differentiable parameter detected: %s", ss.str().c_str()); 
            }
        }
        slist.push_back("normal");
        return slist;
    }
		
    DifferentialList eval_diff(const BSDFSamplingRecord &bRec, EMeasure measure) const {
			if (bRec.component != -1) {
				NotImplementedError("eval_diff multi component");
			}

			//std::vector<Spectrum> sp;
            DifferentialList lst;

            // Add the normal to it;
            Spectrum dNormal = Spectrum(0.0);
            int id = 0;
            for(int i = 0; i < m_bsdfs.size(); i++) {
                //Spectrum pdiff = Spectrum(m_bsdfs[i]->eval(bRec, measure).getLuminance() - m_bsdfs[m_bsdfs.size() - 1]->eval(bRec, measure).getLuminance());
                Spectrum pdiff = Spectrum(m_bsdfs[i]->eval(bRec, measure).getLuminance());
                lst[i] = pdiff;
                DifferentialList difflist = m_bsdfs[i]->eval_diff(bRec, measure);

                if(m_normalIDs[i] != -1)
                    dNormal += m_weights[i] * difflist[m_normalIDs[i]];

                for(int j = 0; j < difflist.size(); j++) {
                    id = m_bsdfs.size() + m_diffOffsets[i] + j;
                    lst[id] = difflist[j];// * m_weights[i];
                    //Log(EInfo, "lst %d -> %d, %d\n", m_bsdfs.size() + m_diffOffsets[i] + j, i,j);
                }
            }

            lst[id+1] = dNormal;

			//Spectrum dalpha = m_bsdfs[0]->eval_diff(bRec, measure).at(1) + m_bsdfs[1]->eval_diff(bRec, measure).at(1);
			//Spectrum dw = m_bsdfs[0]->eval(bRec, measure) - m_bsdfs[1]->eval(bRec, measure);
			
			//sp.push_back(dw);
			//sp.push_back(dalpha);
			return lst;
		}

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        Float result = 0.0f;

        if (bRec.component == -1) {
            for (size_t i=0; i<m_bsdfs.size(); ++i)
                result += m_bsdfs[i]->pdf(bRec, measure) * m_pdf[i];
        } else {
            /* Pick out an individual component */
            int idx = m_indices[bRec.component].first;
            BSDFSamplingRecord bRec2(bRec);
            bRec2.component = m_indices[bRec.component].second;
            return m_bsdfs[idx]->pdf(bRec2, measure);
        }

        return result;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &_sample) const {
        Point2 sample(_sample);
        if (bRec.component == -1) {
            /* Choose a component based on the normalized weights */
            size_t entry = m_pdf.sampleReuse(sample.x);

            Float pdf;
            Spectrum result = m_bsdfs[entry]->sample(bRec, pdf, sample);
            if (result.isZero()) // sampling failed //WHAAAAAT????
                return result;

            result *= m_weights[entry] * pdf;
            pdf *= m_pdf[entry];

            EMeasure measure = BSDF::getMeasure(bRec.sampledType);
            for (size_t i=0; i<m_bsdfs.size(); ++i) {
                if (entry == i)
                    continue;
                pdf += m_bsdfs[i]->pdf(bRec, measure) * m_pdf[i];
                result += m_bsdfs[i]->eval(bRec, measure) * m_weights[i];
            }

            bRec.sampledComponent += m_offsets[entry];
            return result / pdf;
        } else {
            /* Pick out an individual component */
            int requestedComponent = bRec.component;
            int bsdfIndex = m_indices[requestedComponent].first;
            bRec.component = m_indices[requestedComponent].second;
            Spectrum result = m_bsdfs[bsdfIndex]->sample(bRec, sample)
                * m_weights[bsdfIndex];
            bRec.component = bRec.sampledComponent = requestedComponent;
            return result;
        }
    }

    Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &_sample) const {
        Point2 sample(_sample);
        if (bRec.component == -1) {
            /* Choose a component based on the normalized weights */
            size_t entry = m_pdf.sampleReuse(sample.x);

            Spectrum result = m_bsdfs[entry]->sample(bRec, pdf, sample);
            if (result.isZero()) // sampling failed
                return result;

            result *= m_weights[entry] * pdf;
            pdf *= m_pdf[entry];

            EMeasure measure = BSDF::getMeasure(bRec.sampledType);
            for (size_t i=0; i<m_bsdfs.size(); ++i) {
                if (entry == i)
                    continue;
                pdf += m_bsdfs[i]->pdf(bRec, measure) * m_pdf[i];
                result += m_bsdfs[i]->eval(bRec, measure) * m_weights[i];
            }

            bRec.sampledComponent += m_offsets[entry];
            return result/pdf;
        } else {
            /* Pick out an individual component */
            int requestedComponent = bRec.component;
            int bsdfIndex = m_indices[requestedComponent].first;
            bRec.component = m_indices[requestedComponent].second;
            Spectrum result = m_bsdfs[bsdfIndex]->sample(bRec, pdf, sample)
                * m_weights[bsdfIndex];
            bRec.component = bRec.sampledComponent = requestedComponent;
            return result;
        }
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(BSDF))) {
            BSDF *bsdf = static_cast<BSDF *>(child);
            m_bsdfs.push_back(bsdf);

            m_diffOffsets.push_back( m_diffOffsets[m_diffOffsets.size()-1] + bsdf->getDifferentiableParameters() );

            auto names = bsdf->getDifferentiableParameterNames();
            bool found = false;
            for(int i = 0; i < names.size(); i++) {
                if( names[i] == "normal" ) {
                    m_normalIDs.push_back(i);
                    found = true;
                    break;
                }
            }

            if (!found) m_normalIDs.push_back(-1);

            bsdf->incRef();
        } else {
            BSDF::addChild(name, child);
        }
    }

    Float getRoughness(const Intersection &its, int component) const {
        int bsdfIndex = m_indices[component].first;
        component = m_indices[component].second;
        return m_bsdfs[bsdfIndex]->getRoughness(its, component);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "MixtureBSDF[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  weights = {";
        for (size_t i=0; i<m_bsdfs.size(); ++i) {
            oss << " " << m_weights[i];
            if (i + 1 < m_bsdfs.size())
                oss << ",";
        }
        oss << " }," << endl
            << "  sampleWeights = {";
        for (size_t i=0; i<m_bsdfs.size(); ++i) {
            oss << " " << m_samples[i];
            if (i + 1 < m_bsdfs.size())
                oss << ",";
        }
        oss << " }," << endl
            << "  bsdfs = {" << endl;
        for (size_t i=0; i<m_bsdfs.size(); ++i)
            oss << "    " << indent(m_bsdfs[i]->toString(), 2) << "," << endl;
        oss << "  }" << endl
            << "]";
        return oss.str();
    }

    Shader *createShader(Renderer *renderer) const;

    MTS_DECLARE_CLASS()
private:
    std::vector<Float> m_weights;
    std::vector<Float> m_samples;
    std::vector<std::pair<int, int> > m_indices;
    std::vector<int> m_offsets;
    std::vector<BSDF *> m_bsdfs;
    std::vector<int> m_diffOffsets;
    std::vector<int> m_normalIDs;
    DiscreteDistribution m_pdf;
};


Shader *MixedSamplingBSDF::createShader(Renderer *renderer) const {
    Log(EError, "MixureSamplingBSDF does not support real-time shading");
    return nullptr;
}

//MTS_IMPLEMENT_CLASS(MixtureBSDFShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(MixedSamplingBSDF, false, BSDF)
MTS_EXPORT_PLUGIN(MixedSamplingBSDF, "Mixture Sampleable BSDF")
MTS_NAMESPACE_END
