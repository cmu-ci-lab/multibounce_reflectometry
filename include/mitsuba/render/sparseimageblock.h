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
#if !defined(__MITSUBA_RENDER_SPARSEIMAGEBLOCK_H_)
#define __MITSUBA_RENDER_SPARSEIMAGEBLOCK_H_

#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/sched.h>
#include <mitsuba/core/rfilter.h>
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
class MTS_EXPORT_RENDER SparseImageBlock : public WorkResult {
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
    SparseImageBlock(Bitmap::EPixelFormat fmt, const Vector2i &size,
            const ReconstructionFilter *filter = NULL, int channels = -1, bool warn = true);

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
    inline std::map<std::tuple<int, int, int>, Spectrum> getData() { return m_data; }


    /// Return the underlying pixel format
    //inline Bitmap::EPixelFormat getPixelFormat() const { return m_bitmap->getPixelFormat(); }

    /// Return a pointer to the underlying bitmap representation
    //inline Bitmap *getBitmap() { return m_bitmap; }

    /// Return a pointer to the underlying bitmap representation (const version)
    //inline const Bitmap *getBitmap() const { }

    /// Clear everything to zero
    inline void clear() { m_data.clear(); }

    /// Accumulate another image block into this one
    inline void put(const SparseImageBlock *block) {
        //m_bitmap->accumulate(block->getBitmap(),
        //    Point2i(block->getOffset() - m_offset
        //        - Vector2i(block->getBorderSize() - m_borderSize)));

        // Combine the map elements of each one together.
        Log(EInfo, "Accumulating %d elements %d, %d border: %d, %d\n", block->m_data.size(), block->getOffset().x, block->getOffset().y, block->getBorderSize(), m_borderSize);
        for(auto p : block->m_data) {
            // Assumption is that there is no overlap.
            // TODO: Explicitly evaluate assumption.
            auto t = p.first;
	    auto borderSize = block->getBorderSize() - m_borderSize;
	    auto offset = (block->getOffset() - m_offset) - Vector2i(borderSize, borderSize);
            auto t2 = std::make_tuple(std::get<0>(t) + offset.x, std::get<1>(t) + offset.y, std::get<2>(t));
            if(m_data.count(t2))
                m_data[t2] += p.second;
            else
                m_data[t2] = p.second;
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
    FINLINE bool put(const std::tuple<float, float, int> &fpos, const Spectrum &spec, Float alpha) {
        //Float temp[SPECTRUM_SAMPLES + 2];
        //for (int i=0; i<SPECTRUM_SAMPLES; ++i)
        //    temp[i] = spec[i];
        //temp[SPECTRUM_SAMPLES] = alpha;
        //temp[SPECTRUM_SAMPLES + 1] = 1.0f;
        //Log(EInfo, "PUT: %f, %f, %d\n", std::get<0>(fpos), std::get<1>(fpos), std::get<2>(fpos));
        // Alpha not supported. Use the usual put() function.
        return put(fpos, spec);

    }

    /**
     * \brief Store a single sample inside the block
     *
     * \param fpos
     *    A tuple of 3 parameters. The first two denote the fractional pixel types.
     * \param value
     *    Pointer to an array containing each channel of the sample values.
     *    The array must match the length given by \ref getChannelCount()
     * \return \c false if one of the sample values was \a invalid, e.g.
     *    NaN or negative. A warning is also printed in this case
     */
    FINLINE bool put(const std::tuple<float, float, int> &fpos, Spectrum value) {
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
                std::get<0>(fpos) - 0.5f - (m_offset.x - filterRadius),
                std::get<1>(fpos) - 0.5f - (m_offset.y - filterRadius));

            // Determine the affected range of pixels 
            const Point2i min(std::max((int) std::ceil (pos.x - filterRadius), 0),
                              std::max((int) std::ceil (pos.y - filterRadius), 0)),
                          max(std::min((int) std::floor(pos.x + filterRadius), size.x - 1),
                              std::min((int) std::floor(pos.y + filterRadius), size.y - 1));
		
            std::tuple<int, int, int> discrete_pt = std::make_tuple(pos.x, pos.y, std::get<2>(fpos));
	    
            //if(!m_data.count(discrete_pt))  m_data[discrete_pt] = Spectrum(0.0);
	    //m_data[discrete_pt] += value;

            // Lookup values from the pre-rasterized filter 
            for (int x=min.x, idx = 0; x<=max.x; ++x)
                m_weightsX[idx++] = m_filter->evalDiscretized(x-pos.x);
            for (int y=min.y, idx = 0; y<=max.y; ++y)
                m_weightsY[idx++] = m_filter->evalDiscretized(y-pos.y);

            // Rasterize the filtered sample into the framebuffer 
            for (int y=min.y, yr=0; y<=max.y; ++y, ++yr) {
                const Float weightY = m_weightsY[yr];
                //Float *dest = m_bitmap->getFloatData()
                //    + (y * (size_t) size.x + min.x) * channels;
                

                for (int x=min.x, xr=0; x<=max.x; ++x, ++xr) {
                    const Float weight = m_weightsX[xr] * weightY;
                    std::tuple<int, int, int> discrete_pt = std::make_tuple(x, y, std::get<2>(fpos));
                    
                    if(!m_data.count(discrete_pt))  m_data[discrete_pt] = Spectrum(0.0);

                    m_data[discrete_pt] += weight * value;
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
            Log(EWarn, "%s", oss.str().c_str());
        }
        return false;
    }

    /// Create a clone of the entire image block
    ref<SparseImageBlock> clone() const {
        ref<SparseImageBlock> clone = new SparseImageBlock(m_pixelFormat,
            m_bitmap->getSize() - Vector2i(2*m_borderSize, 2*m_borderSize), m_filter, 3);
        copyTo(clone);
        return clone;
    }

    /// Copy the contents of this image block to another one with the same configuration
    void copyTo(SparseImageBlock *copy) const {
        /*memcpy(copy->getBitmap()->getUInt8Data(), m_bitmap->getUInt8Data(), m_bitmap->getBufferSize());*/

        for(auto p : m_data) {
            copy->m_data[p.first] = p.second;
        }

        copy->m_size = m_size;
        copy->m_offset = m_offset;
        copy->m_warn = m_warn;
        //copy->m_size = m_size;

        //Log(EError, "EError is not permitted for SparseImageBlock\n");
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
    virtual ~SparseImageBlock();
protected:
    ref<Bitmap> m_bitmap;
    Point2i m_offset;
    Vector2i m_size;
    int m_borderSize;
    const ReconstructionFilter *m_filter;
    Float *m_weightsX, *m_weightsY;
    bool m_warn;
    Bitmap::EPixelFormat m_pixelFormat;
    std::map<std::tuple<int, int, int>, Spectrum> m_data;
};


MTS_NAMESPACE_END

#endif /* __MITSUBA_RENDER_IMAGEBLOCK_H_ */
