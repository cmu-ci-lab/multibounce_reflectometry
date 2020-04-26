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

#if !defined(__BDPT_DIFFWR_H)
#define __BDPT_DIFFWR_H

#include <mitsuba/render/sparseimageblock.h>
#include <mitsuba/render/reductorimageblock.h>
#include <mitsuba/core/fresolver.h>
#include "bdpt.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                             Work result                              */
/* ==================================================================== */

/**
   Bidirectional path tracing needs its own WorkResult implementation,
   since each rendering thread simultaneously renders to a small 'camera
   image' block and potentially a full-resolution 'light image'.
*/
class BDPTDiffWorkResult : public WorkResult {
public:
	BDPTDiffWorkResult(const BDPTConfiguration &conf, const ReconstructionFilter *filter,
            const Sampler *sampler,
            Vector2i blockSize = Vector2i(-1, -1), bool isReductor = false, const float* reductor = NULL, Vector2i size = Vector2i(0,0));

    // Clear the contents of the work result
    void clear();

    /// Fill the work result with content acquired from a binary data stream
    virtual void load(Stream *stream);

    /// Serialize a work result to a binary data stream
    virtual void save(Stream *stream) const;

    /// Aaccumulate another work result into this one
    void put(const BDPTDiffWorkResult *workResult);

#if BDPT_DEBUG == 1
    /* In debug mode, this function allows to dump the contributions of
       the individual sampling strategies to a series of images */
    void dump(const BDPTConfiguration &conf,
            const fs::path &prefix, const fs::path &stem) const;

    inline void putDebugSample(int s, int t, const Point2 &sample,
            const Spectrum &spec) {
        m_debugBlocks[strategyIndex(s, t)]->put(sample, (const Float *) &spec);
    }
#endif

    /*inline void putSample(const Point2 &sample, const Spectrum &spec) {
        m_block->put(sample, spec, 1.0f);
    }*/

    /*inline void putLightSample(const Point2 &sample, const Spectrum &spec) {
        m_lightImage->put(sample, spec, 1.0f);
    }*/
    
    // TODO: Change float to Float to preserve mitsuba convention.
    inline void putSample(std::tuple<float, float, int> t, const Spectrum& spec, Float pdf) {
        //Log(EInfo, "PUT: %f, %f, %d\n", std::get<0>(t), std::get<1>(t), std::get<2>(t));
        m_block->put(t, spec, pdf);
    }

    inline void putHistogramSample(int s, int t, Float hn, Float hv, const Spectrum &spec) {
        m_block->putHistogramSample(s, t, hn, hv, spec);
    }

    inline const ReductorImageBlock *getImageBlock() const {
        return m_block.get();
    }

    inline const ReductorImageBlock *getLightImage() const {
        return m_lightImage.get();
    }

    inline void setSize(const Vector2i &size) {
        m_block->setSize(size);
    }

    inline void setOffset(const Point2i &offset) {
        m_block->setOffset(offset);
    }

    /// Return a string representation
    std::string toString() const;

    MTS_DECLARE_CLASS()
protected:
    /// Virtual destructor
    virtual ~BDPTDiffWorkResult();

    inline int strategyIndex(int s, int t) const {
        int above = s+t-2;
        return s + above*(5+above)/2;
    }
protected:
#if BDPT_DEBUG == 1
    ref_vector<ImageBlock> m_debugBlocks;
#endif
    ref<ReductorImageBlock> m_block, m_lightImage;
};

MTS_NAMESPACE_END

#endif /* __BDPT_WR_H */
