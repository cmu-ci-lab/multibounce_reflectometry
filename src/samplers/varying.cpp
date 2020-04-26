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

#include <mitsuba/render/sampler.h>
#include <fstream>
MTS_NAMESPACE_BEGIN

/*!\plugin{independent}{Independent sampler}
 * \order{1}
 * \parameters{
 *     \parameter{sampleCount}{\Integer}{
 *       Number of samples per pixel \default{4}
 *     }
 * }
 *
 * \renderings{
 *     \unframedrendering{A projection of the first 1024 points
 *     onto the first two dimensions. Note the sample clumping.}{sampler_independent}
 * }
 *
 * The independent sampler produces a stream of independent and uniformly
 * distributed pseudorandom numbers. Internally, it relies on a fast SIMD version
 * of the Mersenne Twister random number generator \cite{Saito2008SIMD}.
 *
 * This is the most basic sample generator; because no precautions are taken to avoid
 * sample clumping, images produced using this plugin will usually take longer to converge.
 * In theory, this sampler is initialized using a deterministic procedure, which means
 * that subsequent runs of Mitsuba should create the same image. In practice, when
 * rendering with multiple threads and/or machines, this is not true anymore, since the
 * ordering of samples is influenced by the operating system scheduler.
 *
 * Note that the Metropolis-type integrators implemented in Mitsuba are incompatible with
 * the more sophisticated sample generators shown in this section. They \emph{require} this
 * specific sampler and refuse to work otherwise.
 */
class SpatiallyVaryingSampler : public Sampler {
public:
    SpatiallyVaryingSampler() : Sampler(Properties()) { }

    SpatiallyVaryingSampler(const Properties &props) : Sampler(props) {
        /* Number of samples per pixel when used with a sampling-based integrator */
        m_sampleCount = 4;
        m_sampleMultiplier = props.getFloat("sampleMultiplier", 1.f);
        m_random = new Random();

        /* Obtain the sampling weight texture */
        std::string spatialFile = props.getString("samplerFile", "");

        if(spatialFile.size() == 0) {
            Log(EError, "'samplerFile' not specified");
        }

        std::ifstream fspatial(spatialFile);
        int w,h,c;
        fspatial.read(reinterpret_cast<char*>(&w), sizeof(int));
        fspatial.read(reinterpret_cast<char*>(&h), sizeof(int));
        fspatial.read(reinterpret_cast<char*>(&c), sizeof(int));
        m_spatialSize.x = w;
        m_spatialSize.y = h;
        if (c != 1) {
            Log(EError, "Reductor file must be single channel. Number of channels detected: %d", c);
        }

        m_spatialSampler = new Float[w * h * c];
        fspatial.read(reinterpret_cast<char*>(m_spatialSampler), sizeof(Float) * w * h * c);
        Log(EInfo, "Finished reading from file. %f, %dx%d", m_spatialSampler[w * h - 1], m_spatialSize.x, m_spatialSize.y);
    }

    SpatiallyVaryingSampler(Stream *stream, InstanceManager *manager)
     : Sampler(stream, manager) {
        m_random = static_cast<Random *>(manager->getInstance(stream));
        m_spatialSize.x = stream->readInt();
        m_spatialSize.y = stream->readInt();
        m_sampleMultiplier = stream->readFloat();
        m_spatialSampler = new Float[m_spatialSize.x * m_spatialSize.y];
        stream->readFloatArray(m_spatialSampler, m_spatialSize.x * m_spatialSize.y);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        Sampler::serialize(stream, manager);
        manager->serialize(stream, m_random.get());
        stream->writeInt(m_spatialSize.x);
        stream->writeInt(m_spatialSize.y);
        stream->writeFloat(m_sampleMultiplier);
        stream->writeFloatArray(m_spatialSampler, m_spatialSize.x * m_spatialSize.y);
    }

    ref<Sampler> clone() {
        ref<SpatiallyVaryingSampler> sampler = new SpatiallyVaryingSampler();
        sampler->m_sampleCount = m_sampleCount;
        sampler->m_random = new Random(m_random);
        sampler->m_spatialSize = Vector2i(m_spatialSize.x, m_spatialSize.y);
        sampler->m_spatialSampler = new Float[m_spatialSize.x * m_spatialSize.y];
        sampler->m_sampleMultiplier = m_sampleMultiplier;
        memcpy(
            sampler->m_spatialSampler,
            m_spatialSampler,
            sizeof(Float) * m_spatialSize.x * m_spatialSize.y);

        for (size_t i=0; i<m_req1D.size(); ++i)
            sampler->request1DArray(m_req1D[i]);
        for (size_t i=0; i<m_req2D.size(); ++i)
            sampler->request2DArray(m_req2D[i]);
        return sampler.get();
    }

    inline virtual size_t getSampleCount(const Point2i& offset) const {
        return static_cast<size_t>(
            m_sampleMultiplier * std::abs( 
                m_spatialSampler[
                    m_spatialSize.x * std::max(0, std::min(offset.y, m_spatialSize.y-1)) +
                    std::max(0, std::min(offset.x, m_spatialSize.x-1))]
            )
        );
    }

    void generate(const Point2i &offset) {
        size_t sampleCount = getSampleCount(offset);

        for (size_t i=0; i<m_req1D.size(); i++)
            for (size_t j=0; j<sampleCount * m_req1D[i]; ++j)
                m_sampleArrays1D[i][j] = m_random->nextFloat();
        for (size_t i=0; i<m_req2D.size(); i++)
            for (size_t j=0; j<sampleCount * m_req2D[i]; ++j)
                m_sampleArrays2D[i][j] = Point2(
                    m_random->nextFloat(),
                    m_random->nextFloat());
        m_sampleIndex = 0;
        m_dimension1DArray = m_dimension2DArray = 0;

    }

    Float next1D() {
        return m_random->nextFloat();
    }

    Point2 next2D() {
        Float value1 = m_random->nextFloat();
        Float value2 = m_random->nextFloat();
        return Point2(value1, value2);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "SpatiallyVaryingSampler[" << endl
            << "  sampleCount = " << m_sampleCount << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Random> m_random;
    Float* m_spatialSampler;
    Vector2i m_spatialSize;
    Float m_sampleMultiplier;
};

MTS_IMPLEMENT_CLASS_S(SpatiallyVaryingSampler, false, Sampler)
MTS_EXPORT_PLUGIN(SpatiallyVaryingSampler, "Spatially Varying sampler");
MTS_NAMESPACE_END
