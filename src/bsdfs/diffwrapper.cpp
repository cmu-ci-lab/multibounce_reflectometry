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

#include <mitsuba/render/scene.h>
#include <mitsuba/hw/basicshader.h>

MTS_NAMESPACE_BEGIN

/*! \plugin{diffwrapper}{Adds mesh normal differentiability to a given material}
 * \order{13}
 * \icon{bsdf_bumpmap}
 *
 * \parameters{
 *     \parameter{\Unnamed}{\BSDF}{A BSDF model that should
 *     be affected by the normal map}
 * }
 */
class DifferentialWrapper : public BSDF {
    public:
        DifferentialWrapper(const Properties &props) : BSDF(props) { 
            m_indexCount = props.getInteger("varCount", -1); // Number of variables.
            m_indices = NULL;
            m_normals = NULL;
        }

        DifferentialWrapper(Stream *stream, InstanceManager *manager)
            : BSDF(stream, manager) {
                m_nested = static_cast<BSDF *>(manager->getInstance(stream));
                //m_normals = static_cast<Texture *>(manager->getInstance(stream));

                // TODO: WARN: mitsuba-diff Modifications
                //m_indices = static_cast<Texture *>(manager->getInstance(stream));

                m_indexCount = stream->readInt();
                m_nonNormalDiffCount = stream->readInt();

                for(int i = 0; i < m_nonNormalDiffCount; i++){
                    //m_diffNames.push_back(stream->readString());
                    m_diffIndices.push_back(stream->readInt());
                }

                int invDiffIndices = stream->readInt();
                for(int i = 0; i < invDiffIndices; i++){
                    m_invDiffIndices.push_back(stream->readInt());
                }

                configure();
            }

        int nestedDifferentialIndex(std::string str) {
            auto difflist = m_nested->getDifferentiableParameterNames();
            for(int i = 0; i < difflist.size(); i++) {
                if( difflist.at(i) == str )
                    return i;
            }

            return -1;
        }

        void nestedDifferentialSize() {
            auto difflist = m_nested->getDifferentiableParameterNames();
            m_nonNormalDiffCount = 0;
            m_invDiffIndices.clear();
            m_diffIndices.clear();

            for(int i = 0; i < difflist.size(); i++) {
                auto const dotpos = difflist.at(i).find_last_of(':');
                if( difflist.at(i) == "normal" )
                    continue;
                else if(difflist.at(i).substr(dotpos+1) == "normal")
                    continue;
                else {
                    m_diffNames.push_back(difflist.at(i));
                    m_diffIndices.push_back(i);
                    //Log(EInfo, "non-normal map := %d -> %s, %d", m_nonNormalDiffCount, difflist.at(i).c_str(), i);
                    m_nonNormalDiffCount += 1;
                }
            }
            // Create inverse map.
            int maxDiffIndex = 0;
            for(int i = 0; i < m_diffIndices.size(); i++){
                if(maxDiffIndex < m_diffIndices[i]) maxDiffIndex = m_diffIndices[i];
            }

            m_invDiffIndices.resize(maxDiffIndex + 1, -1);
            for(int i = 0; i < m_diffIndices.size(); i++){
                m_invDiffIndices[m_diffIndices[i]] = i;
                //Log(EInfo, "m_invDiffIndices[%d]=%d", m_diffIndices[i], i);
            }

        }

        void configure() {
            if (!m_nested)
                Log(EError, "A child BSDF instance is required");

            m_components.clear();
            for (int i=0; i<m_nested->getComponentCount(); ++i)
                m_components.push_back(m_nested->getType(i));

            m_usesRayDifferentials = true;

            // Find the normal differentials 'normalX' 'normalY' and 'normalZ'
            m_idxNormal = nestedDifferentialIndex("normal");
            nestedDifferentialSize();

            if(m_idxNormal == -1) 
                Log(EError, "Couldn't find one or more differentials w.r.t the normal in the nested BSDF instance");

            BSDF::configure();
        }

        void serialize(Stream *stream, InstanceManager *manager) const {
            BSDF::serialize(stream, manager);

            manager->serialize(stream, m_nested.get());
            //manager->serialize(stream, m_normals.get());
            //manager->serialize(stream, m_indices.get());

            stream->writeInt(m_indexCount);
            stream->writeInt(m_nonNormalDiffCount);
            for(int i = 0; i < m_nonNormalDiffCount; i++){
                //stream->writeString(m_diffNames[i]);
                stream->writeInt(m_diffIndices[i]);
            }

            stream->writeInt(m_invDiffIndices.size());
            for(int i = 0; i < m_invDiffIndices.size(); i++){
                stream->writeInt(m_invDiffIndices[i]);
            }
        }

        Spectrum getDiffuseReflectance(const Intersection &its) const {
            return m_nested->getDiffuseReflectance(its);
        }

        Spectrum getSpecularReflectance(const Intersection &its) const {
            return m_nested->getSpecularReflectance(its);
        }

        void addChild(const std::string &name, ConfigurableObject *child) {
            if (child->getClass()->derivesFrom(MTS_CLASS(BSDF))) {
                if (m_nested != NULL)
                    Log(EError, "Only a single nested BSDF can be added!");
                m_nested = static_cast<BSDF *>(child);
            } /*else if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
                if (m_normals != NULL && m_indices != NULL)
                    Log(EError, "Only two textures can be specified for this BSDF!");
                const Properties &props = child->getProperties();

                if (!props.getBoolean("indexmap", false)){
                    // No "indexmap" property. It's the normal map.
                    if (props.getPluginName() == "bitmap" && !props.hasProperty("gamma"))
                        Log(EError, "When using a bitmap texture as a normal map, please explicitly specify "
                                "the 'gamma' parameter of the bitmap plugin. In most cases the following is the correct choice: "
                                "<float name=\"gamma\" value=\"1.0\"/>");
                    if(m_normals != NULL) {
                        Log(EError, "The normal map has already been set." 
                                "Please specify <boolean name=\"indexmap\" value=\"true\"/> for the index map texture");
                    }
                    
                    m_normals = static_cast<Texture *>(child);
                    m_indexCount = m_normals->getResolution().x * m_normals->getResolution().y;
                } else {
                    // Has "indexmap" as a property. It's a mapping from texels to indices.
                    // Note: The values in this texture have to be 'integers' even though they are specified
                    // as doubles.
                    
                    if(m_indices != NULL) {
                        Log(EError, "The index map has already been set.");
                    }

                    m_indices = static_cast<Texture *>(child);
                    int k = m_indices->getResolution().x * m_indices->getResolution().y;
                    Log(EInfo, "indexcount: %d\n", k);
                    if(m_indexCount != k) {
                        Log(EError, "The index map does not have the same resolution as the normal map %d vs %d. Also "
                                "make sure that the index map is specified after the normal map.", m_indexCount, k);
                    }
                }

            }*/ else {
                BSDF::addChild(name, child);
            }
        }

        Frame getFrame(const Intersection &its) const {
            Frame result;
            /*Normal n;

            m_normals->eval(its, false).toLinearRGB(n.x, n.y, n.z);
            for (int i=0; i<3; ++i)
                n[i] = 2 * n[i] - 1;

            Frame frame = BSDF::getFrame(its);
            result.n = normalize(frame.toWorld(n));

            result.s = normalize(its.dpdu - result.n
                    * dot(result.n, its.dpdu));

            result.t = cross(result.n, result.s);

            return result;*/
        }

        void getFrameDerivative(const Intersection &its, Frame &du, Frame &dv) const {
            Vector n;

            /*m_normals->eval(its, false).toLinearRGB(n.x, n.y, n.z);
            for (int i=0; i<3; ++i)
                n[i] = 2 * n[i] - 1;

            Spectrum dn[2];
            Vector dndu, dndv;
            m_normals->evalGradient(its, dn);
            Spectrum(2*dn[0]).toLinearRGB(dndu.x, dndu.y, dndu.z);
            Spectrum(2*dn[1]).toLinearRGB(dndv.x, dndv.y, dndv.z);

            Frame base_du, base_dv;
            Frame base = BSDF::getFrame(its);
            BSDF::getFrameDerivative(its, base_du, base_dv);

            Vector worldN = base.toWorld(n);

            Float invLength_n = 1/worldN.length();
            worldN *= invLength_n;

            du.n = invLength_n * (base.toWorld(dndu) + base_du.toWorld(n));
            dv.n = invLength_n * (base.toWorld(dndv) + base_dv.toWorld(n));
            du.n -= dot(du.n, worldN) * worldN;
            dv.n -= dot(dv.n, worldN) * worldN;

            Vector s = its.dpdu - worldN * dot(worldN, its.dpdu);
            Float invLen_s = 1.0f / s.length();
            s *= invLen_s;

            du.s = invLen_s * (-du.n * dot(worldN, its.dpdu) - worldN * dot(du.n, its.dpdu));
            dv.s = invLen_s * (-dv.n * dot(worldN, its.dpdu) - worldN * dot(dv.n, its.dpdu));

            du.s -= s * dot(du.s, s);
            dv.s -= s * dot(dv.s, s);

            du.t = cross(du.n, s) + cross(worldN, du.s);
            dv.t = cross(dv.n, s) + cross(worldN, dv.s);*/
        }

        Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
            /*const Intersection& its = bRec.its;
            Intersection perturbed(its);
            perturbed.shFrame = getFrame(its);

            BSDFSamplingRecord perturbedQuery(perturbed,
                    perturbed.toLocal(its.toWorld(bRec.wi)),
                    perturbed.toLocal(its.toWorld(bRec.wo)), bRec.mode);

            if (Frame::cosTheta(bRec.wo) * Frame::cosTheta(perturbedQuery.wo) <= 0)
                return Spectrum(0.0f);

            perturbedQuery.sampler = bRec.sampler;
            perturbedQuery.typeMask = bRec.typeMask;
            perturbedQuery.component = bRec.component;*/

            return m_nested->eval(bRec, measure);
        }

        Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
            /*const Intersection& its = bRec.its;
            Intersection perturbed(its);
            perturbed.shFrame = getFrame(its);

            BSDFSamplingRecord perturbedQuery(perturbed,
                    perturbed.toLocal(its.toWorld(bRec.wi)),
                    perturbed.toLocal(its.toWorld(bRec.wo)), bRec.mode);
            if (Frame::cosTheta(bRec.wo) * Frame::cosTheta(perturbedQuery.wo) <= 0)
                return 0;
            perturbedQuery.mode = bRec.mode;
            perturbedQuery.sampler = bRec.sampler;
            perturbedQuery.typeMask = bRec.typeMask;
            perturbedQuery.component = bRec.component;*/
            return m_nested->pdf(bRec, measure);
        }

        virtual int getDifferentiableParameters() {
            // Return all the differentiable parameters (one index for each point on the grid).
            Log(EError, "This class should not be used with a fixed count differential system. Number of parameters not known at init().");
            return 0;
        }

        /**
         * \brief Return the tags/names of the parameters that can be differentiated
         * This is optional. The names are merely for convenience.
         */
        std::vector<std::string> getDifferentiableParameterNames() {
            /*std::vector<std::string> svec;
            //svec.push_back("alpha");
            for(int i = 0; i < m_indexCount; i++) {
                std::stringstream ss;
                ss << "grid_normal_" << i;

                svec.push_back(ss.str());
            }
            return svec;*/
            std::vector<std::string> svec;
            for(int i = 0; i < m_diffNames.size(); i++){
                svec.push_back(m_diffNames[i]);
            }

            return svec;
        }

        std::map<int,Spectrum>  eval_diff(const BSDFSamplingRecord &bRec, EMeasure measure) const {
            /*std::map<int, Spectrum> diffs;
            Spectrum hitIndex = m_indices->eval(its, false);


            Intersection perturbed(its);
            perturbed.shFrame = getFrame(its);

            BSDFSamplingRecord perturbedQuery(perturbed, bRec.sampler, bRec.mode);
            perturbedQuery.wi = perturbed.toLocal(its.toWorld(bRec.wi));
            perturbedQuery.sampler = bRec.sampler;
            perturbedQuery.typeMask = bRec.typeMask;
            perturbedQuery.component = bRec.component;
            //std::map<int,Spectrum> resultDiffs = m_nested->sample(perturbedQuery, sample);
            std::map<int,Spectrum> resultDiffs = m_nested->eval_diff(perturbedQuery, measure);
            Spectrum result = m_nested->eval(perturbedQuery, measure);

            if (!result.isZero()) {
                bRec.sampledComponent = perturbedQuery.sampledComponent;
                bRec.sampledType = perturbedQuery.sampledType;
                bRec.wo = its.toLocal(perturbed.toWorld(perturbedQuery.wo));
                bRec.eta = perturbedQuery.eta;
                if (Frame::cosTheta(bRec.wo) * Frame::cosTheta(perturbedQuery.wo) <= 0){
                    return diffs;
                }
            }

            diffs[static_cast<int>(hitIndex[0])] = resultDiffs.at(m_idxNormal); // Each component of the Spectrum accounts for one differential.

            return result;*/
            // --
            
            const Intersection& its = bRec.its;
            std::map<int, Spectrum> diffs;

            /*Float x, y, z;

            m_indices->eval(its, false).toLinearRGB(x, y, z);
            x = round(x * m_indexCount);
            
            Normal n;
            m_normals->eval(its, false).toLinearRGB(n.x, n.y, n.z);
            for (int i=0; i<3; ++i)
                n[i] = 2 * n[i] - 1;*/

            //Log(EInfo, "At %d: %f %f %f nmap %f %f %f", static_cast<int>(x), its.p.x, its.p.y, its.p.z, n.x, n.y, n.z);
            //std::cout << "Hit Index " << hitIndex.toString() << std::endl;
            //if(x != 0)
            //    Log(EInfo, "Hit index: %f\n", x);

            /*if(static_cast<int>(x) > 100 || static_cast<int>(x) < 0) {
                std::cout << its.toString() << std::endl;
                Log(EError, "Invalid hit index: %d %f: %s", static_cast<int>(x), x);
                
                //return diffs;
            }*/
            //Log(EError, "Found a hitIndex %f", hitIndex[0]);

            //Intersection perturbed(its);
            //m_normals->eval(its, false).toLinearRGB(n.x, n.y, n.z);
            //for (int i=0; i<3; ++i)
            //    n[i] = 2 * n[i] - 1;
            /*Log(EInfo, "At %d: %f %f %f normal: %f %f %f", static_cast<int>(x),  its.p.x, its.p.y, its.p.z, tup.x, tup.y, tup.z);
            Log(EInfo, "At %d: %f %f %f ldir: %f %f %f", static_cast<int>(x),  its.p.x, its.p.y, its.p.z, ldir.x, ldir.y, ldir.z);
            Log(EInfo, "At %d: %f %f %f fin: %f %f %f", static_cast<int>(x),  its.p.x, its.p.y, its.p.z, fin.x, fin.y, fin.z);
            Log(EInfo, "At %d: %f %f %f dpdu: %f %f %f", static_cast<int>(x),  its.p.x, its.p.y, its.p.z, its.dpdu.x, its.dpdu.y, its.dpdu.z);
            Log(EInfo, "At %d: %f %f %f dpdv: %f %f %f", static_cast<int>(x),  its.p.x, its.p.y, its.p.z, its.dpdv.x, its.dpdv.y, its.dpdv.z);*/
            //Log(EInfo, "bRec.wo: %f %f %f", bRec.wo.x, bRec.wo.y, bRec.wo.z);
            //Log(EInfo, "bRec.wi: %f %f %f", bRec.wi.x, bRec.wi.y, bRec.wi.z);
            //Log(EInfo, "At %d: perturbedQuery.wo: %f %f %f", static_cast<int>(x), perturbedQuery.wo.x, perturbedQuery.wo.y, perturbedQuery.wo.z);
            //Log(EInfo, "At %d: perturbedQuery.wi: %f %f %f", static_cast<int>(x), perturbedQuery.wi.x, perturbedQuery.wi.y, perturbedQuery.wi.z);
            
            //Log(EError, "");

            std::map<int,Spectrum> resultDiffs = m_nested->eval_diff(bRec, measure);
            //Log(EInfo, "resultDiffs size: %d, m_idxNormal: %d", resultDiffs.size(), m_idxNormal);
            Spectrum tmp = resultDiffs.at(m_idxNormal); // Each component of the Spectrum accounts for one differential.
            Vector v(tmp[0], tmp[1], tmp[2]);
            Vector transformed = its.toWorld(v);
            //Vector k(0, 0, 1);
            //Vector kt = its.toLocal(k);

            // Load the standard non-normal indices.
            //for(int i = 0; i < m_nonNormalDiffCount; i++) {
            //    diffs[i] = resultDiffs[m_diffIndices[i]];
            //}
            // The following is a faster implementation of the above code, especially for sparse differentials.
            for (std::map<int, Spectrum>::iterator it = resultDiffs.begin(); it != resultDiffs.end(); ++it)
            {   
                
                if(it->first >= m_invDiffIndices.size())
                    continue;

                int idx = m_invDiffIndices.at(it->first);
                if(idx != -1)
                    diffs[idx] = it->second;
            }

            // Compute the normal indices.
            //Log(EInfo, "kt: %f %f %f", kt.x, kt.y, kt.z);
            //Log(EInfo, "v: %f %f %f", v.x, v.y, v.z);
            Float tmpf[3] = {transformed.x, transformed.y, transformed.z};
            
            Vector bary(1 - (its.uv.x + its.uv.y), its.uv.x, its.uv.y);
            // TODO: Assumes that the shape is a trimesh.
            if(!its.shape->getClass()->derivesFrom(MTS_CLASS(TriMesh))) {
                Log(EError, "Non tri mesh detected in intersection test. Not supported");
            }
            auto tr = reinterpret_cast<const TriMesh*>(its.shape)->getTriangles()[its.primIndex];
            //auto tmesh = reinterpret_cast<const TriMesh*>(its.shape);
            //Log(EInfo, "transformed: %f %f %f at %d %d %d", transformed.x, transformed.y, transformed.z, tr.idx[0], tr.idx[1], tr.idx[2]);
            /*Log(EInfo, "Indices: %d %d %d", tr.idx[0], tr.idx[1], tr.idx[2]);
            Log(EInfo, "Intersection: %f %f %f", its.p.x, its.p.y, its.p.z);
            Log(EInfo, "Barycentrics: %f %f %f", bary.x, bary.y, bary.z);
            auto pt0 = tmesh->getVertexPositions()[tr.idx[0]];
            Log(EInfo, "Point0: %f %f %f", pt0.x, pt0.y, pt0.z);
            auto pt1 = tmesh->getVertexPositions()[tr.idx[1]];
            Log(EInfo, "Point1: %f %f %f", pt1.x, pt1.y, pt1.z);
            auto pt2 = tmesh->getVertexPositions()[tr.idx[2]];
            Log(EInfo, "Point2: %f %f %f", pt2.x, pt2.y, pt2.z);

            auto bar0 = cross(pt1 - its.p, pt2 - its.p).length() / cross(pt0 - pt1, pt1 - pt2).length();
            auto bar1 = cross(pt2 - its.p, pt0 - its.p).length() / cross(pt0 - pt1, pt1 - pt2).length();
            auto bar2 = cross(pt0 - its.p, pt1 - its.p).length() / cross(pt0 - pt1, pt1 - pt2).length();
            Log(EInfo, "Barycentrics[E]: %f %f %f", bar0, bar1, bar2);*/

            diffs[tr.idx[0] + m_nonNormalDiffCount] = bary.x * Spectrum(tmpf);
            diffs[tr.idx[1] + m_nonNormalDiffCount] = bary.y * Spectrum(tmpf);
            diffs[tr.idx[2] + m_nonNormalDiffCount] = bary.z * Spectrum(tmpf);
            return diffs;
        }

        Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
            /*const Intersection& its = bRec.its;
            Intersection perturbed(its);
            perturbed.shFrame = getFrame(its);

            BSDFSamplingRecord perturbedQuery(perturbed, bRec.sampler, bRec.mode);
            perturbedQuery.wi = perturbed.toLocal(its.toWorld(bRec.wi));
            perturbedQuery.sampler = bRec.sampler;
            perturbedQuery.typeMask = bRec.typeMask;
            perturbedQuery.component = bRec.component;*/
            Spectrum result = m_nested->sample(bRec, sample);
            /*if (!result.isZero()) {
                bRec.sampledComponent = perturbedQuery.sampledComponent;
                bRec.sampledType = perturbedQuery.sampledType;
                bRec.wo = its.toLocal(perturbed.toWorld(perturbedQuery.wo));
                bRec.eta = perturbedQuery.eta;
                if (Frame::cosTheta(bRec.wo) * Frame::cosTheta(perturbedQuery.wo) <= 0)
                    return Spectrum(0.0f);
            }*/
            return result;
        }

        Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const {
            /*const Intersection& its = bRec.its;
            Intersection perturbed(its);
            perturbed.shFrame = getFrame(its);

            BSDFSamplingRecord perturbedQuery(perturbed, bRec.sampler, bRec.mode);
            perturbedQuery.wi = perturbed.toLocal(its.toWorld(bRec.wi));
            perturbedQuery.typeMask = bRec.typeMask;
            perturbedQuery.component = bRec.component;*/
            Spectrum result = m_nested->sample(bRec, pdf, sample);

            /*if (!result.isZero()) {
                bRec.sampledComponent = perturbedQuery.sampledComponent;
                bRec.sampledType = perturbedQuery.sampledType;
                bRec.wo = its.toLocal(perturbed.toWorld(perturbedQuery.wo));
                bRec.eta = perturbedQuery.eta;
                if (Frame::cosTheta(bRec.wo) * Frame::cosTheta(perturbedQuery.wo) <= 0)
                    return Spectrum(0.0f);
            }*/

            return result;
        }

        Float getRoughness(const Intersection &its, int component) const {
            return m_nested->getRoughness(its, component);
        }

        std::string toString() const {
            std::ostringstream oss;
            oss << "NormalMap[" << endl
                << "  id = \"" << getID() << "\"," << endl
                << "  normals = " << indent(m_normals->toString()) << endl
                << "  indices = " << indent(m_indices->toString()) << endl
                << "  nested = " << indent(m_nested->toString()) << endl
                << "]";
            return oss.str();
        }

        Shader *createShader(Renderer *renderer) const;

        MTS_DECLARE_CLASS()
            protected:
            ref<Texture> m_normals;
            ref<Texture> m_indices;
            ref<BSDF> m_nested;
            int m_idxNormal;
            int m_indexCount;
            std::vector<std::string> m_diffNames;
            std::vector<int> m_diffIndices;
            std::vector<int> m_invDiffIndices;
            int m_nonNormalDiffCount;
        };

        // ================ Hardware shader implementation ================

        /**
         * This is a quite approximate version of the normal map model -- it likely
         * won't match the reference exactly, but it should be good enough for
         * preview purposes
         */
        class DiffShader : public Shader {
            public:
                DiffShader(Renderer *renderer, const BSDF *nested, const Texture *normals)
                    : Shader(renderer, EBSDFShader), m_nested(nested), m_normals(normals) {
                        m_nestedShader = renderer->registerShaderForResource(m_nested.get());
                        m_normalShader = renderer->registerShaderForResource(m_normals.get());
                    }

                bool isComplete() const {
                    return m_nestedShader.get() != NULL;
                }

                void cleanup(Renderer *renderer) {
                    renderer->unregisterShaderForResource(m_nested.get());
                    renderer->unregisterShaderForResource(m_normals.get());
                }

                void putDependencies(std::vector<Shader *> &deps) {
                    deps.push_back(m_nestedShader.get());
                    deps.push_back(m_normalShader.get());
                }

                void generateCode(std::ostringstream &oss,
                        const std::string &evalName,
                        const std::vector<std::string> &depNames) const {
                    oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
                        << "    vec3 n = normalize(2.0*" << depNames[1] << "(uv) - vec3(1.0));" << endl
                        << "    vec3 s = normalize(vec3(1.0-n.x*n.x, -n.x*n.y, -n.x*n.z)); " << endl
                        << "    vec3 t = cross(s, n);" << endl
                        << "    wi = vec3(dot(wi, s), dot(wi, t), dot(wi, n));" << endl
                        << "    wo = vec3(dot(wo, s), dot(wo, t), dot(wo, n));" << endl
                        << "    return " << depNames[0] << "(uv, wi, wo);" << endl
                        << "}" << endl
                        << endl
                        << "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
                        << "    vec3 n = normalize(2.0*" << depNames[1] << "(uv) - vec3(1.0));" << endl
                        << "    vec3 s = normalize(vec3(1.0-n.x*n.x, -n.x*n.y, -n.x*n.z)); " << endl
                        << "    vec3 t = cross(s, n);" << endl
                        << "    wi = vec3(dot(wi, s), dot(wi, t), dot(wi, n));" << endl
                        << "    wo = vec3(dot(wo, s), dot(wo, t), dot(wo, n));" << endl
                        << "    return " << depNames[0] << "_diffuse(uv, wi, wo);" << endl
                        << "}" << endl
                        << endl;
                }

                MTS_DECLARE_CLASS()
            private:
                    ref<const BSDF> m_nested;
                    ref<const Texture> m_normals;
                    ref<Shader> m_nestedShader;
                    ref<Shader> m_normalShader;
        };

        Shader *DifferentialWrapper::createShader(Renderer *renderer) const {
            return new DiffShader(renderer, m_nested.get(), m_normals.get());
        }

        MTS_IMPLEMENT_CLASS(DiffShader, false, Shader)
            MTS_IMPLEMENT_CLASS_S(DifferentialWrapper, false, BSDF)
            MTS_EXPORT_PLUGIN(DifferentialWrapper, "Normal map modifier");
        MTS_NAMESPACE_END
