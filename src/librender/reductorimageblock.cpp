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

#include <mitsuba/render/reductorimageblock.h>

MTS_NAMESPACE_BEGIN

ReductorImageBlock::ReductorImageBlock(Bitmap::EPixelFormat fmt, const Vector2i &size,
        const ReconstructionFilter *filter, 
        const Sampler *sampler, int channels, bool warn, const Float *reductor, Vector2i reductorSize, int ignoreIndex) : m_offset(0),
        m_size(size), m_filter(filter), 
        m_sampler(sampler), m_weightsX(NULL), m_weightsY(NULL), m_warn(warn), m_pixelFormat(fmt), m_ignoreIndex(ignoreIndex) {
    m_borderSize = filter ? filter->getBorderSize() : 0;

    /* Allocate a small bitmap data structure for the block */
    //m_bitmap = new Bitmap(fmt, Bitmap::EFloat,
    //    size + Vector2i(2 * m_borderSize), channels);
    //m_reductor_texture = reductor;
    m_reductorSize = reductorSize;
    if(reductor)
    {
        m_reductor = new Float[m_reductorSize.x * m_reductorSize.y];
        Log(EInfo, "Creating reductor image block.. %dx%d", m_reductorSize.x, m_reductorSize.y);
        memcpy(m_reductor, reductor, sizeof(Float) * m_reductorSize.x * m_reductorSize.y);
        Log(EInfo, "Finished reading from file. %f, %dx%d", m_reductor[m_reductorSize.x * m_reductorSize.y - 1], m_reductorSize.x, m_reductorSize.y);
        printf("ReductorImageBlock DATA: %f\n", m_reductor[0]);
    } else {
        m_reductor = nullptr;
        Log(EInfo, "Reductor running in raw mode. Greatly increased memory consumption and reduced performance");
    }

    Log(EInfo, "ReductorImageBlock created with dimensions %d, %d: Reductor texture size: %d, %d", size.x, size.y, m_reductorSize.x, m_reductorSize.y);
    if (filter) {
        /* Temporary buffers used in put() */
        int tempBufferSize = (int) std::ceil(2*filter->getRadius()) + 1;
        m_weightsX = new Float[2*tempBufferSize];
        m_weightsY = m_weightsX + tempBufferSize;
    }

    m_hnResolution = 64;
    m_hvResolution = 64;

    m_hnUnitSize = 1.0 / m_hnResolution;
    m_hvUnitSize = 1.0 / m_hvResolution;

}

ReductorImageBlock::~ReductorImageBlock() {
    Log(EInfo, "Destroying ReductorImageBlock: %d elements", m_data.size());
    m_data.clear();
    if (m_weightsX)
        delete[] m_weightsX;
    if (m_reductor)
        delete[] m_reductor;
}

void ReductorImageBlock::load(Stream *stream) {
    m_offset = Point2i(stream);
    m_size = Vector2i(stream);
    m_paddedSize = Vector2i(stream);

    uint64_t numElements = stream->readLong();    
    
    //m_reductor.load(stream);
    m_data.clear(); 
    //Log(EError, "load() not implemented for SparseImageBlock");
    for(uint64_t l = 0; l < numElements; l++) {
	    auto n = stream->readInt();
	    auto dx = stream->readFloat();
	    auto dy = stream->readFloat();
	    auto dz = stream->readFloat();
	    Float v[3] = {dx, dy, dz};
	    m_data[n] = Spectrum(v);
    }

    uint64_t numElementsRaw = stream->readLong();    

    //m_reductor.load(stream);
    m_dataUncompressed.clear(); 
    //Log(EError, "load() not implemented for SparseImageBlock");
    for(uint64_t l = 0; l < numElementsRaw; l++) {
	    auto x = stream->readInt();
        auto y = stream->readInt();
        auto n = stream->readInt();
	    auto dx = stream->readFloat();
	    auto dy = stream->readFloat();
	    auto dz = stream->readFloat();
	    Float v[3] = {dx, dy, dz};
	    m_dataUncompressed[std::make_tuple(x,y,n)] = Spectrum(v);
    }

    uint64_t numHistogramElements = stream->readLong();

    //m_reductor.load(stream);
    m_histogramData.clear(); 
    //m_histogramData.resize(numHistogramElements, Spectrum(0.0));
    //Log(EError, "load() not implemented for SparseImageBlock");
    for(uint64_t l = 0; l < numHistogramElements; l++) {
        auto is = stream->readInt();
	    auto it = stream->readInt();
	    auto ihn = stream->readInt();
        auto ihv = stream->readInt();
	    auto sx = stream->readFloat();
	    auto sy = stream->readFloat();
	    auto sz = stream->readFloat();
	    Float v[3] = {sx, sy, sz};
	    m_histogramData[std::make_tuple(is, it, ihn, ihv)] = Spectrum(v);
    }

    stream->readFloatArray(
        m_reductor,
        (size_t) m_reductorSize.x *
        (size_t) m_reductorSize.y);

}

void ReductorImageBlock::save(Stream *stream) const {
    m_offset.serialize(stream);
    m_size.serialize(stream);
    m_paddedSize.serialize(stream);

    stream->writeLong(m_data.size());
    for(auto item : m_data) {
        stream->writeInt(item.first);
	    stream->writeFloat(item.second[0]);
	    stream->writeFloat(item.second[1]);
	    stream->writeFloat(item.second[2]);
    }

    stream->writeLong(m_dataUncompressed.size());
    for(auto item : m_dataUncompressed) {
        stream->writeInt(std::get<0>(item.first));
        stream->writeInt(std::get<1>(item.first));
        stream->writeInt(std::get<2>(item.first));
	    stream->writeFloat(item.second[0]);
	    stream->writeFloat(item.second[1]);
	    stream->writeFloat(item.second[2]);
    }

    stream->writeLong(m_histogramData.size());
    for(auto item : m_histogramData) {
        stream->writeInt(std::get<0>(item.first));
        stream->writeInt(std::get<1>(item.first));
        stream->writeInt(std::get<2>(item.first));
        stream->writeInt(std::get<3>(item.first));
        stream->writeFloat(item.second[0]);
        stream->writeFloat(item.second[1]);
        stream->writeFloat(item.second[2]);
    }

    stream->writeFloatArray(
        m_reductor,
        (size_t) m_reductorSize.x *
        (size_t) m_reductorSize.y);
}

std::string ReductorImageBlock::toString() const {
    std::ostringstream oss;
    oss << "ReductorImageBlock[" << endl
        << "  offset = " << m_offset.toString() << "," << endl
        << "  size = " << m_size.toString() << "," << endl
        << "  borderSize = " << m_borderSize << endl
        << "]";
    return oss.str();
}

MTS_IMPLEMENT_CLASS(ReductorImageBlock, false, WorkResult)
MTS_NAMESPACE_END
