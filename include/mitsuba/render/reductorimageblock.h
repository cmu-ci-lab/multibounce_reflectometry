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

#pragma once
#if !defined(__MITSUBA_RENDER_REDUCTORIMAGEBLOCK_H_)
#define __MITSUBA_RENDER_REDUCTORIMAGEBLOCK_H_

#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/sched.h>
#include <mitsuba/core/rfilter.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/texture.h>
#include <map>
#include <tuple>

MTS_NAMESPACE_BEGIN

/**
 * \brief Storage for an image sub-block (a.k.a render bucket)
 *
 * This class is used by image-based parallel processes and encapsulates
 * computed rectangular regions of an image. This allows for easy and efficient
 * distributed rendering of large images. Image blocks usually also include a
 * border region storing contribuctions that are slightly outside of the block,
 * which is required to support image reconstruction filters.
 *
 * \ingroup librender
 */
class MTS_EXPORT_RENDER ReductorImageBlock : public WorkResult {
public:
    /**
     * Construct a new image block of the requested properties
     *
     * \param fmt
     *    Specifies the pixel format -- see \ref Bitmap::EPixelFormat
     *    for a list of possibilities
     * \param size
     *    Specifies the block dimensions (not accounting for additional
     *    border pixels required to support image reconstruction filters)
     * \param channels
     *    Specifies the number of output channels. This is only necessary
     *    when \ref Bitmap::EMultiChannel is chosen as the pixel format
     * \param warn
     *    Warn when writing bad sample values?
     */
    ReductorImageBlock(Bitmap::EPixelFormat fmt, const Vector2i &size,
            const ReconstructionFilter *filter = NULL, 
            const Sampler *sampler = NULL, int channels = -1, bool warn = true, const float* reductor = NULL, Vector2i reductorSize = Vector2i(0,0), int ignoreIndex = -1);

    /// Set the current block offset
    inline void setOffset(const Point2i &offset) { m_offset = offset; }

    /// Return the current block offset
    inline const Point2i &getOffset() const { return m_offset; }

    /// Set the current block size
    inline void setSize(const Vector2i &size) { m_size = size; }

    /// Return the current block size
    inline const Vector2i &getSize() const { return m_size; }

    /// Return the bitmap's width in pixels
    inline int getWidth() const { return m_size.x; }

    /// Return the bitmap's height in pixels
    inline int getHeight() const { return m_size.y; }

    /// Warn when writing bad sample values?
    inline bool getWarn() const { return m_warn; }

    /// Warn when writing bad sample values?
    inline void setWarn(bool warn) { m_warn = warn; }

    /// Return the border region used by the reconstruction filter
    inline int getBorderSize() const { return m_borderSize; }

    /// Return the number of channels stored by the image block
    inline int getChannelCount() const { return 3; }
    
    /// Return the underying map
    inline std::map<int, Spectrum> getData() const { return m_data; }

    /// Return the underying uncompressed map
    inline std::map<std::tuple<int,int,int>, Spectrum> getUncompressedData() const { return m_dataUncompressed; }

    /// Set internal data
    inline void setData(std::map<int, Spectrum> data) { m_data = data; }

    /// 
    inline int getIgnoreIndex() { return m_ignoreIndex; }

    /// Get histogram data.
    inline std::map<std::tuple<int, int, int, int>, Spectrum> getHistogramData() const { return m_histogramData; }

    /// Get histogram meta data.
    inline int getHistogramHNResolution() const { return m_hnResolution; }

    /// Get histogram meta data.
    inline int getHistogramHVResolution() const { return m_hvResolution; }


    /// Return the underlying pixel format
    //inline Bitmap::EPixelFormat getPixelFormat() const { return m_bitmap->getPixelFormat(); }

    /// Return a pointer to the underlying bitmap representation
    //inline Bitmap *getBitmap() { return m_bitmap; }

    /// Return a pointer to the underlying bitmap representation (const version)
    //inline const Bitmap *getBitmap() const { }

    /// Clear everything to zero
    inline void clear() { m_data.clear(); m_dataUncompressed.clear(); m_histogramData.clear(); }

    /// Accumulate another image block into this one
    inline void put(const ReductorImageBlock *block) {
        //m_bitmap->accumulate(block->getBitmap(),
        //    Point2i(block->getOffset() - m_offset
        //        - Vector2i(block->getBorderSize() - m_borderSize)));

        // Combine the map elements of each one together.
        Log(EInfo, "Accumulating %d/%d elements %d, %d border: %d, %d localsize: %d, %d targetsize: %d, %d", block->m_data.size(), block->m_dataUncompressed.size(), block->getOffset().x, block->getOffset().y, block->getBorderSize(), m_borderSize, m_size.x, m_size.y, block->m_size.x, block->m_size.y);
        for(auto p : block->m_data) {
            // Assumption is that there is no overlap.
            // TODO: Explicitly evaluate assumption.
            auto t = p.first;
            if(m_data.count(t))
                m_data[t] += p.second;
            else
                m_data[t] = p.second;
        }

        for(auto p : block->m_dataUncompressed) {
            auto t = p.first;
            if(m_dataUncompressed.count(t)){
                m_dataUncompressed[t] += p.second;
            }else{
                m_dataUncompressed[t] = p.second;
            }
        }

        for(auto p : block->m_histogramData) {
            auto t = p.first;
            if(m_histogramData.count(t))
                m_histogramData[t] += p.second;
            else
                m_histogramData[t] = p.second;
        }
    }

    /**
     * \brief Store a single sample inside the image block
     *
     * This variant assumes that the image block stores spectrum,
     * alpha, and reconstruction filter weight values.
     *
     * \param pos
     *    Denotes the sample position in fractional pixel coordinates
     * \param spec
     *    Spectrum value assocated with the sample
     * \param alpha
     *    Alpha value assocated with the sample
     * \return \c false if one of the sample values was \a invalid, e.g.
     *    NaN or negative. A warning is also printed in this case
     */
    FINLINE bool put(const std::tuple<float, float, int> &fpos, const Spectrum &spec, Float alpha, Float pdf) {
        //Float temp[SPECTRUM_SAMPLES + 2];
        //for (int i=0; i<SPECTRUM_SAMPLES; ++i)
        //    temp[i] = spec[i];
        //temp[SPECTRUM_SAMPLES] = alpha;
        //temp[SPECTRUM_SAMPLES + 1] = 1.0f;
        //Log(EInfo, "PUT: %f, %f, %d\n", std::get<0>(fpos), std::get<1>(fpos), std::get<2>(fpos));
        // Alpha not supported. Use the usual put() function.
        return put(fpos, spec, pdf);

    }

    /**
     * \brief Angle sampler histogram collection function.
     */
    FINLINE bool putHistogramSample(int s, int t, Float hn, Float hv, const Spectrum &spec) {
        if(hn < 0 || hv < 0) {
            // Negative dot products not supported.
            return false;
        }

        uint hnBucket = static_cast<uint>((hn / m_hnUnitSize));
        uint hvBucket = static_cast<uint>((hv / m_hvUnitSize));
        hnBucket = hnBucket == m_hnResolution ? hnBucket - 1 : hnBucket;
        hvBucket = hvBucket == m_hvResolution ? hvBucket - 1 : hvBucket;

        Float v[3];
        v[0] = spec.average();
        v[1] = 1.0f;
        v[2] = 0.0f;
        Spectrum spec2 = Spectrum(v);

        auto k = std::make_tuple(s, t, hnBucket, hvBucket);
        if(m_histogramData.count(k))
            m_histogramData[k] += spec2;
        else
            m_histogramData[k] = spec2;

    }

    /**
     * \brief Store a single sample inside the block
     * Modified to work with spatially varying pixel samples. Uses importance weights
     * to account for the change in pdf from a adjacent pixel's sample.
     * \param fpos
     *    A tuple of 3 parameters. The first two denote the fractional pixel types.
     * \param value
     *    Pointer to an array containing each channel of the sample values.
     *    The array must match the length given by \ref getChannelCount()
     * \return \c false if one of the sample values was \a invalid, e.g.
     *    NaN or negative. A warning is also printed in this case
     */
    FINLINE bool put(const std::tuple<float, float, int> &fpos, Spectrum value, Float pdf) {
        //const int channels = m_bitmap->getChannelCount();
        //Log(EInfo, "PUT: %f, %f, %d\n", std::get<0>(fpos), std::get<1>(fpos), std::get<2>(fpos));

        //if(channels != 3) {
        //    Log(EError, "Illegal channel count.");
        //}
        int channels = 3;
        /* Check if all sample values are valid */
        for (int i=0; i<channels; ++i) {
            if (EXPECT_NOT_TAKEN((!std::isfinite(value[i]))))
                goto bad_sample;
        }
        
        {
            const Float filterRadius = m_filter->getRadius();
            //const Float filterRadius = 0.0f;
            const Vector2i &size = m_size;
            
            //filterRadius = 0.0f;

            // Convert to pixel coordinates within the image block 
            const Point2 pos(
                std::get<0>(fpos) - 0.5f - (m_offset.x - m_borderSize),
                std::get<1>(fpos) - 0.5f - (m_offset.y - m_borderSize));

            // Determine the affected range of pixels 
            /*const Point2i min(std::max((int) std::ceil (pos.x - filterRadius), 0),
                              std::max((int) std::ceil (pos.y - filterRadius), 0)),
                          max(std::min((int) std::floor(pos.x + filterRadius), size.x - 1),
                              std::min((int) std::floor(pos.y + filterRadius), size.y - 1));*/
            const Point2i min((int) std::ceil (pos.x - filterRadius),
                              (int) std::ceil (pos.y - filterRadius)),
                          max((int) std::floor(pos.x + filterRadius),
                              (int) std::floor(pos.y + filterRadius));
            // std::tuple<int, int, int> discrete_pt = std::make_tuple(pos.x, pos.y, std::get<2>(fpos));

            // if(!m_data.count(discrete_pt))  m_data[discrete_pt] = Spectrum(0.0);
	        // m_data[discrete_pt] += value;

            // Lookup values from the pre-rasterized filter
            for (int x=min.x, idx = 0; x<=max.x; ++x)
                m_weightsX[idx++] = m_filter->evalDiscretized(x-pos.x);
            for (int y=min.y, idx = 0; y<=max.y; ++y)
                m_weightsY[idx++] = m_filter->evalDiscretized(y-pos.y);

            auto pixdata = m_reductor;
            auto stride = m_reductorSize.x;

            int discrete_pt = std::get<2>(fpos);
            if(m_ignoreIndex != -1 && discrete_pt > m_ignoreIndex) {
                // Ignore this index. This feature is used in conjunction with raw rendering to
                // extract index-slices for a limited number of indices.
                return true;
            }
            // Rasterize the filtered sample into the framebuffer 
            for (int y=min.y, yr=0; y<=max.y; ++y, ++yr) {
                const Float weightY = m_weightsY[yr];

                //Float *dest = m_bitmap->getFloatData()
                //    + (y * (size_t) size.x + min.x) * channels;

                for (int x=min.x, xr=0; x<=max.x; ++x, ++xr) {
                    const Float weight = m_weightsX[xr] * weightY;
                    int discrete_pt = std::get<2>(fpos);

                    /*if (x < 0 || x >= size.x || y < 0 || y >= size.y) {
                        // Discard values outside the frame.
                            continue;
                    }*/

                    if(!m_data.count(discrete_pt))  m_data[discrete_pt] = Spectrum(0.0);

                    auto borderSize = m_borderSize;
	                auto offset = m_offset - Vector2i(borderSize, borderSize);


                    // Convert to global coordinates.
                    // TODO: This is a clear problem :O. Not including border radius..
                    // why??
                    // TODO: WARN: Fixed it for now. ensure the results still work out.
                    auto _x = x + (offset.x);
                    auto _y = y + (offset.y);
                    
                    Float importanceWeight = 1.f;
                    // Obtain the sample count for this offset.
                    if(pdf != 0.f){
                        Float localPdf = static_cast<Float>(m_sampler->getSampleCount(Point2i(_x,_y)));
                        importanceWeight = localPdf / pdf;
                    }
                    auto invSampleCount = 1.f / static_cast<Float>(m_sampler->getSampleCount(Point2i(_x,_y)));

                    if(_x < 0 || _x >= m_reductorSize.x || _y < 0 || _y >= m_reductorSize.y) {
                        // Discard values outside the entire image.
                        continue;
                    }

                    //printf("Accumulating: Wt: %f, Value: %f,%f,%f, x:%d, y:%d, pixvalue: %.2e\n", weight, value[0], value[1], value[2], _x, _y, pixdata[_y * stride + _x]);
                    if (weight == 0.0f) continue;
                    if (pixdata){
                        m_data[discrete_pt] +=     weight * 
                                                    value * 
                                pixdata[_y * stride + _x] *
                                         importanceWeight;
                    } else {
                        
                        auto t = std::make_tuple(_x, _y, discrete_pt);
                        if(!m_dataUncompressed.count(t)) m_dataUncompressed[t] = Spectrum(0.f);
                        m_dataUncompressed[t] += weight * value * importanceWeight;
                    }
                    //for (int k=0; k<channels; ++k)
                    //    *dest++ += weight * value[k];
                }
            }
            
            //m_data[t] = pos;
        }

        return true;

        bad_sample:
        {
            std::ostringstream oss;
            oss << "Invalid sample value : [";
            for (int i=0; i<channels; ++i) {
                oss << value[i];
                if (i+1 < channels)
                    oss << ", ";
            }
            oss << "]";
            oss << ",[";
            oss << std::get<0>(fpos) << ", " << std::get<1>(fpos) << ", " << std::get<2>(fpos);
            oss << "]";
            Log(EWarn, "%s", oss.str().c_str());
        }
        return false;
    }

    /// Create a clone of the entire image block
    ref<ReductorImageBlock> clone() const {
        ref<ReductorImageBlock> clone = new ReductorImageBlock(m_pixelFormat,
            m_reductorSize - Vector2i(2*m_borderSize, 2*m_borderSize), m_filter, m_sampler, 3, true, m_reductor, m_reductorSize);
        copyTo(clone);
        return clone;
    }

    /// Copy the contents of this image block to another one with the same configuration
    void copyTo(ReductorImageBlock *copy) const {
        /*memcpy(copy->getBitmap()->getUInt8Data(), m_bitmap->getUInt8Data(), m_bitmap->getBufferSize());*/

        for(auto p : m_data) {
            copy->m_data[p.first] = p.second;
        }

        copy->m_size = m_size;
        copy->m_offset = m_offset;
        copy->m_warn = m_warn;
        copy->m_reductor = m_reductor;
        copy->m_reductorSize = m_reductorSize;
        //copy->m_size = m_size;

        //Log(EError, "EError is not permitted for ReductorImageBlock\n");
    }

    // ======================================================================
    //! @{ \name Implementation of the WorkResult interface
    // ======================================================================

    void load(Stream *stream);
    void save(Stream *stream) const;
    std::string toString() const;

    //! @}
    // ======================================================================

    MTS_DECLARE_CLASS()
protected:
    /// Virtual destructor
    virtual ~ReductorImageBlock();
protected:
    float* m_reductor;
    Vector2i m_reductorSize;
    Point2i m_offset;
    Vector2i m_size;
    int m_borderSize;
    Vector2i m_paddedSize;
    const ReconstructionFilter *m_filter;
    const Sampler *m_sampler;
    Float *m_weightsX, *m_weightsY;
    bool m_warn;
    Bitmap::EPixelFormat m_pixelFormat;
    std::map<int, Spectrum> m_data;
    std::map<std::tuple<int, int, int>, Spectrum> m_dataUncompressed;
    int m_ignoreIndex;

    // h.v and h.n sampling histogram data.
    std::map<std::tuple<int, int, int, int>,Spectrum> m_histogramData;
    int m_hnResolution;
    int m_hvResolution;
    Float m_hnUnitSize;
    Float m_hvUnitSize;
};


MTS_NAMESPACE_END

#endif /* __MITSUBA_RENDER_IMAGEBLOCK_H_ */
