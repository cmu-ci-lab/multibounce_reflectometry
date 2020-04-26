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

#include <mitsuba/render/sparseimageblock.h>

MTS_NAMESPACE_BEGIN

SparseImageBlock::SparseImageBlock(Bitmap::EPixelFormat fmt, const Vector2i &size,
        const ReconstructionFilter *filter, int channels, bool warn) : m_offset(0),
        m_size(size), m_filter(filter), m_weightsX(NULL), m_weightsY(NULL), m_warn(warn), m_pixelFormat(fmt) {
    m_borderSize = filter ? filter->getBorderSize() : 0;

    /* Allocate a small bitmap data structure for the block */
    //m_bitmap = new Bitmap(fmt, Bitmap::EFloat,
    //    size + Vector2i(2 * m_borderSize), channels);
    Log(EInfo, "SparseImageBlock created with dimensions %d, %d", size.x, size.y);
    if (filter) {
        /* Temporary buffers used in put() */
        int tempBufferSize = (int) std::ceil(2*filter->getRadius()) + 1;
        m_weightsX = new Float[2*tempBufferSize];
        m_weightsY = m_weightsX + tempBufferSize;
    }
}

SparseImageBlock::~SparseImageBlock() {
    Log(EInfo, "Destroying SparseImageBlock: %d elements", m_data.size());
    m_data.clear(); 
    if (m_weightsX)
        delete[] m_weightsX;
}

void SparseImageBlock::load(Stream *stream) {
    m_offset = Point2i(stream);
    m_size = Vector2i(stream);

    uint64_t numElements = stream->readLong();    
    
    m_data.clear(); 
    //Log(EError, "load() not implemented for SparseImageBlock");
    for(uint64_t l = 0; l < numElements; l++) {
	auto x = stream->readInt();
	auto y = stream->readInt();
	auto n = stream->readInt();
	auto dx = stream->readFloat();
	auto dy = stream->readFloat();
	auto dz = stream->readFloat();
	Float v[3] = {dx, dy, dz};
	m_data[std::make_tuple(x, y, n)] = Spectrum(v);
    }
    /*stream->readFloatArray(
        m_bitmap->getFloatData(),
        (size_t) m_bitmap->getSize().x *
        (size_t) m_bitmap->getSize().y * m_bitmap->getChannelCount());*/

    // ERROR.
}

void SparseImageBlock::save(Stream *stream) const {
    m_offset.serialize(stream);
    m_size.serialize(stream);

    // ERROR.
    //Log(EError, "save() not implemented for SparseImageBlock");
    stream->writeLong(m_data.size());
    for(auto item : m_data) {
        stream->writeInt(std::get<0>(item.first));
        stream->writeInt(std::get<1>(item.first));
        stream->writeInt(std::get<2>(item.first));
	stream->writeFloat(item.second[0]);
	stream->writeFloat(item.second[1]);
	stream->writeFloat(item.second[2]);
    }
    /*stream->writeFloatArray(
        m_bitmap->getFloatData(),
        (size_t) m_bitmap->getSize().x *
        (size_t) m_bitmap->getSize().y * m_bitmap->getChannelCount());*/
}


std::string SparseImageBlock::toString() const {
    std::ostringstream oss;
    oss << "SparseImageBlock[" << endl
        << "  offset = " << m_offset.toString() << "," << endl
        << "  size = " << m_size.toString() << "," << endl
        << "  borderSize = " << m_borderSize << endl
        << "]";
    return oss.str();
}

MTS_IMPLEMENT_CLASS(SparseImageBlock, false, WorkResult)
MTS_NAMESPACE_END
