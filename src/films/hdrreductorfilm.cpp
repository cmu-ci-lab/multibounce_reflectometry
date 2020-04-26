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

#include <mitsuba/render/film.h>
#include <mitsuba/render/reductorimageblock.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/statistics.h>
#include <boost/algorithm/string.hpp>
#include "banner.h"
#include "annotations.h"
#include <cmath>
#include <fstream>
MTS_NAMESPACE_BEGIN

/*!\plugin{hdrsparsefilm}{High dynamic range sparse 4D film}
 * \order{1}
 * \parameters{
 *     \parameter{width, height}{\Integer}{
 *       Width and height of the camera sensor in pixels
 *       \default{768, 576}
 *     }
 *     \parameter{fileFormat}{\String}{
 *       Denotes the desired output file format. The options
 *       are \code{openexr} (for ILM's OpenEXR format),
 *       \code{rgbe} (for Greg Ward's RGBE format),
 *       or \code{pfm} (for the Portable Float Map format)
 *       \default{\code{openexr}}
 *     }
 *     \parameter{pixelFormat}{\String}{Specifies the desired pixel format
 *         of output images. The options are \code{luminance},
 *         \code{luminanceAlpha}, \code{rgb}, \code{rgba}, \code{xyz},
 *         \code{xyza}, \code{spectrum}, and \code{spectrumAlpha}.
 *         For the \code{spectrum*} options, the number of written channels depends on
 *         the value assigned to \code{SPECTRUM\_SAMPLES} during compilation
 *         (see \secref{compiling} for details)
 *         \default{\code{rgb}}
 *     }
 *     \parameter{componentFormat}{\String}{Specifies the desired floating
 *         point component format of output images. The options are
 *         \code{float16}, \code{float32}, or \code{uint32}.
 *         \default{\code{float16}}
 *     }
 *     \parameter{cropOffsetX, cropOffsetY, cropWidth, cropHeight}{\Integer}{
 *       These parameters can optionally be provided to select a sub-rectangle              
 *       of the output. In this case, Mitsuba will only render the requested
 *       regions. \default{Unused}
 *     }
 *     \parameter{attachLog}{\Boolean}{Mitsuba can optionally attach
 *         the entire rendering log file as a metadata field so that this
 *         information is permanently saved.
 *         \default{\code{true}, i.e. attach it}
 *     }
 *     \parameter{banner}{\Boolean}{Include a small Mitsuba banner in the
 *         output image? \default{\code{true}}
 *     }
 *     \parameter{highQualityEdges}{\Boolean}{
 *        If set to \code{true}, regions slightly outside of the film
 *        plane will also be sampled. This may improve the image
 *        quality at the edges, especially when using very large
 *        reconstruction filters. In general, this is not needed though.
 *        \default{\code{false}, i.e. disabled}
 *     }
 *     \parameter{\Unnamed}{\RFilter}{Reconstruction filter that should
 *     be used by the film. \default{\code{gaussian}, a windowed Gaussian filter}}
 * }
 *
 * This is the default film plugin that is used when none is explicitly
 * specified. It stores the captured image as a high dynamic range OpenEXR file
 * and tries to preserve the rendering as much as possible by not performing any
 * kind of post processing, such as gamma correction---the output file
 * will record linear radiance values.
 *
 * When writing OpenEXR files, the film will either produce a luminance, luminance/alpha,
 * RGB(A), XYZ(A) tristimulus, or spectrum/spectrum-alpha-based bitmap having a
 * \code{float16}, \code{float32}, or \code{uint32}-based internal representation
 * based on the chosen parameters.
 * The default configuration is RGB with a \code{float16} component format,
 * which is appropriate for most purposes. Note that the spectral output options
 * only make sense when using a custom build of Mitsuba that has spectral
 * rendering enabled (this is not the case for the downloadable release builds).
 * For OpenEXR files, Mitsuba also supports fully general multi-channel output;
 * refer to the \pluginref{multichannel} plugin for details on how this works.
 *
 * The plugin can also write RLE-compressed files in the Radiance RGBE format
 * pioneered by Greg Ward (set \code{fileFormat=rgbe}), as well as the
 * Portable Float Map format (set \code{fileFormat=pfm}).
 * In the former case,
 * the \code{componentFormat} and \code{pixelFormat} parameters are ignored,
 * and the output is ``\code{float8}''-compressed RGB data.
 * PFM output is restricted to \code{float32}-valued images using the
 * \code{rgb} or \code{luminance} pixel formats.
 * Due to the superior accuracy and adoption of OpenEXR, the use of these
 * two alternative formats is discouraged however.
 *
 * When RGB(A) output is selected, the measured spectral power distributions are
 * converted to linear RGB based on the CIE 1931 XYZ color matching curves and
 * the ITU-R Rec. BT.709-3 primaries with a D65 white point.
 *
 * \begin{xml}[caption=Instantiation of a film that writes a full-HD RGBA OpenEXR file without the Mitsuba banner]
 * <film type="hdrfilm">
 *     <string name="pixelFormat" value="rgba"/>
 *     <integer name="width" value="1920"/>
 *     <integer name="height" value="1080"/>
 *     <boolean name="banner" value="false"/>
 * </film>
 * \end{xml}
 *
 * \subsubsection*{Render-time annotations:}
 * \label{sec:film-annotations}
 * The \pluginref{ldrfilm} and \pluginref{hdrfilm} plugins support a
 * feature referred to as \emph{render-time annotations} to facilitate
 * record keeping.
 * Annotations are used to embed useful information inside a rendered image so
 * that this information is later available to anyone viewing the image.
 * Exemplary uses of this feature might be to store the frame or take number,
 * rendering time, memory usage, camera parameters, or other relevant scene
 * information.
 *
 * Currently, two different types are supported: a \code{metadata} annotation
 * creates an entry in the metadata table of the image, which is preferable
 * when the image contents should not be touched. Alternatively, a \code{label}
 * annotation creates a line of text that is overlaid on top of the image. Note
 * that this is only visible when opening the output file (i.e. the line is not
 * shown in the interactive viewer).
 * The syntax of this looks as follows:
 *
 * \begin{xml}
 * <film type="hdrfilm">
 *  <!-- Create a new metadata entry 'my_tag_name' and set it to the
 *       value 'my_tag_value' -->
 *  <string name="metadata['key_name']" value="Hello!"/>
 *
 *  <!-- Add the label 'Hello' at the image position X=50, Y=80 -->
 *  <string name="label[50, 80]" value="Hello!"/>
 * </film>
 * \end{xml}
 *
 * The \code{value="..."} argument may also include certain keywords that will be
 * evaluated and substituted when the rendered image is written to disk. A list all available
 * keywords is provided in Table~\ref{tbl:film-keywords}.
 *
 * Apart from querying the render time,
 * memory usage, and other scene-related information, it is also possible
 * to `paste' an existing parameter that was provided to another plugin---for instance,
 * the camera transform matrix would be obtained as \code{\$sensor['toWorld']}. The name of
 * the active integrator plugin is given by \code{\$integrator['type']}, and so on.
 * All of these can be mixed to build larger fragments, as following example demonstrates.
 * The result of this annotation is shown in Figure~\ref{fig:annotation-example}.
 * \begin{xml}[mathescape=false]
 * <string name="label[10, 10]" value="Integrator: $integrator['type'],
 *   $film['width']x$film['height'], $sampler['sampleCount'] spp,
 *   render time: $scene['renderTime'], memory: $scene['memUsage']"/>
 * \end{xml}
 * \vspace{1cm}
 * \renderings{
 * \fbox{\includegraphics[width=.8\textwidth]{images/annotation_example}}\hfill\,
 * \caption{\label{fig:annotation-example}A demonstration of the label annotation feature
 *  given the example string shown above.}
 * }
 * \vspace{2cm}
 * \begin{table}[htb]
 * \centering
 * \begin{savenotes}
 * \begin{tabular}{ll}
 * \toprule
 * \code{\$scene['renderTime']}& Image render time, use \code{renderTimePrecise} for more digits.\\
 * \code{\$scene['memUsage']}& Mitsuba memory usage\footnote{The definition of this quantity unfortunately
 * varies a bit from platform to platform. On Linux and Windows, it denotes the total
 * amount of allocated RAM and disk-based memory that is private to the process (i.e. not
 * shared or shareable), which most intuitively captures the amount of memory required for
 * rendering. On OSX, it denotes the working set size---roughly speaking, this is the
 * amount of RAM apportioned to the process (i.e. excluding disk-based memory).}.
 * Use \code{memUsagePrecise} for more digits.\\
 * \code{\$scene['coreCount']}& Number of local and remote cores working on the rendering job\\
 * \code{\$scene['blockSize']}& Block size used to parallelize up the rendering workload\\
 * \code{\$scene['sourceFile']}& Source file name\\
 * \code{\$scene['destFile']}& Destination file name\\
 * \code{\$integrator['..']}& Copy a named integrator parameter\\
 * \code{\$sensor['..']}& Copy a named sensor parameter\\
 * \code{\$sampler['..']}& Copy a named sampler parameter\\
 * \code{\$film['..']}& Copy a named film parameter\\
 * \bottomrule
 * \end{tabular}
 * \end{savenotes}
 * \caption{\label{tbl:film-keywords}A list of all special
 * keywords supported by the annotation feature}
 * \end{table}
 *
 */

class HDRReductorFilm : public Film {
public:
    HDRReductorFilm(const Properties &props) : Film(props) {
        /* Should an Mitsuba banner be added to the output image? */
        m_banner = props.getBoolean("banner", true);
        /* Attach the log file as the EXR comment attribute? */
        m_attachLog = props.getBoolean("attachLog", true);

        std::string fileFormat = boost::to_lower_copy(
            props.getString("fileFormat", "openexr"));
        std::vector<std::string> pixelFormats = tokenize(boost::to_lower_copy(
            props.getString("pixelFormat", "rgb")), " ,");
        std::vector<std::string> channelNames = tokenize(
            props.getString("channelNames", ""), ", ");
        std::string componentFormat = boost::to_lower_copy(
            props.getString("componentFormat", "float16"));
        std::string reductorFile = props.getString("reductorFile", "");
        int ignoreIndex = props.getInteger("ignoreIndex", -1);
        m_ignoreIndex = ignoreIndex;

        if(reductorFile.size() == 0) {
            Log(EWarn, "'reductorFile' not specified. Operating in RAW mode. Note that this format is incompatible with the tensorflow plugins");
            m_rawMode = true;
        } else {
            m_rawMode = false;
        }

        if (!m_rawMode){
            std::ifstream freductor(reductorFile);
            int w,h,c;
            freductor.read(reinterpret_cast<char*>(&w), sizeof(int));
            freductor.read(reinterpret_cast<char*>(&h), sizeof(int));
            freductor.read(reinterpret_cast<char*>(&c), sizeof(int));
            m_reductorSize.x = w;
            m_reductorSize.y = h;
            if (c != 1) {
                Log(EError, "Reductor file must be single channel. Number of channels detected: %d", c);
            }

            m_reductor = new Float[w * h * c];
            freductor.read(reinterpret_cast<char*>(m_reductor), sizeof(Float) * w * h * c);
            Log(EInfo, "Finished reading from file. %f, %dx%d", m_reductor[w * h - 1], m_reductorSize.x, m_reductorSize.y);
        } else {
            m_reductor = NULL;
            m_reductorSize.x = m_size.x;
            m_reductorSize.y = m_size.y;
            //memset(m_reductor, 0, sizeof(Float) * m_size.x * m_size.y);
        }

        if (fileFormat == "shds") {
            m_fileFormat = Bitmap::ESparseHDRStream;
        } else {
            Log(EError, "The \"fileFormat\" parameter must be "
                "equal to \"shds\"!");
        }

        if (pixelFormats.empty())
            Log(EError, "At least one pixel format must be specified!");

        if ((pixelFormats.size() != 1 && channelNames.size() != pixelFormats.size()) ||
            (pixelFormats.size() == 1 && channelNames.size() > 1))
            Log(EError, "Number of channel names must match the number of specified pixel formats!");

        if (pixelFormats.size() != 1 && m_fileFormat != Bitmap::EOpenEXR)
            Log(EError, "General multi-channel output is only supported when writing OpenEXR files!");

        for (size_t i=0; i<pixelFormats.size(); ++i) {
            std::string pixelFormat = pixelFormats[i];
            std::string name = i < channelNames.size() ? (channelNames[i] + std::string(".")) : "";

            if (pixelFormat == "luminance") {
                m_pixelFormats.push_back(Bitmap::ELuminance);
                m_channelNames.push_back(name + "Y");
            } else { 
                Log(EError, "The \"pixelFormat\" parameter must be equal to "
                    "\"luminance\". Other formats are not supported for sparse rendering!");
            }
        }

        //if (componentFormat == "float16") {
            m_componentFormat = Bitmap::EFloat16;
        //} else 
        if (componentFormat == "float32") {
            m_componentFormat = Bitmap::EFloat32;
        } /*else if (componentFormat == "uint32") {
            m_componentFormat = Bitmap::EUInt32;
        }*/ else {
            Log(EError, "The \"componentFormat\" parameter must either be "
                "equal to \"float32\"! Other formats are not supported for Sparse 3D Stream");
        }

        std::vector<std::string> keys = props.getPropertyNames();
        for (size_t i=0; i<keys.size(); ++i) {
            std::string key = boost::to_lower_copy(keys[i]);
            key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());

            if ((boost::starts_with(key, "metadata['") && boost::ends_with(key, "']")) ||
                (boost::starts_with(key, "label[") && boost::ends_with(key, "]")))
                props.markQueried(keys[i]);
        }

        if (m_pixelFormats.size() == 1) {
            m_storage = new ReductorImageBlock(Bitmap::ESpectrumAlphaWeight, m_cropSize, NULL, NULL, 0, true, (m_rawMode ? NULL : m_reductor), m_reductorSize, m_ignoreIndex);
        } else {
            m_storage = new ReductorImageBlock(Bitmap::EMultiSpectrumAlphaWeight, m_cropSize,
                            NULL, NULL, (int) (SPECTRUM_SAMPLES * m_pixelFormats.size() + 2), true, (m_rawMode ? NULL : m_reductor), m_reductorSize, m_ignoreIndex);
        }

        Log(EInfo, "Initialized reductor film.");
        //m_reductor = NULL;
    }

    HDRReductorFilm(Stream *stream, InstanceManager *manager)
        : Film(stream, manager) {
        m_banner = stream->readBool();
        m_attachLog = stream->readBool();
        m_fileFormat = (Bitmap::EFileFormat) stream->readUInt();
        m_pixelFormats.resize((size_t) stream->readUInt());
        for (size_t i=0; i<m_pixelFormats.size(); ++i)
            m_pixelFormats[i] = (Bitmap::EPixelFormat) stream->readUInt();
        m_channelNames.resize((size_t) stream->readUInt());
        for (size_t i=0; i<m_channelNames.size(); ++i)
            m_channelNames[i] = stream->readString();
        m_componentFormat = (Bitmap::EComponentFormat) stream->readUInt();
        m_reductorSize.x = stream->readInt();
        m_reductorSize.y = stream->readInt();
        m_rawMode = stream->readBool();
        //m_reductor = static_cast<Texture*>(manager->getInstance(stream));
        Log(EInfo, "HDRFilm() from stream %dx%d", m_reductorSize.x, m_reductorSize.y);
        Log(EInfo, "HDRFilm: %s", this->toString().c_str());

        if(!m_rawMode){
            m_reductor = new Float[m_reductorSize.x * m_reductorSize.y];
            stream->readFloatArray(m_reductor, m_reductorSize.x * m_reductorSize.y);
        } else {
            m_reductor = nullptr;
        }
    }
    ~HDRReductorFilm(){
        Log(EInfo, "Deleting film");
        if (m_reductor)
            delete[] m_reductor;
    }
    void serialize(Stream *stream, InstanceManager *manager) const {
        Film::serialize(stream, manager);
        stream->writeBool(m_banner);
        stream->writeBool(m_attachLog);
        stream->writeUInt(m_fileFormat);
        stream->writeUInt((uint32_t) m_pixelFormats.size());
        for (size_t i=0; i<m_pixelFormats.size(); ++i)
            stream->writeUInt(m_pixelFormats[i]);
        stream->writeUInt((uint32_t) m_channelNames.size());
        for (size_t i=0; i<m_channelNames.size(); ++i)
            stream->writeString(m_channelNames[i]);
        stream->writeUInt(m_componentFormat);
        //m_reductor->serialize(stream, manager);
        stream->writeInt(m_reductorSize.x);
        stream->writeInt(m_reductorSize.y);
        stream->writeBool(m_rawMode);
        Log(EInfo, "HDRFilm: %s", this->toString().c_str());
        Log(EInfo, "HDRFilm::serialize %dx%d", m_reductorSize.x, m_reductorSize.y);
        if (!m_rawMode)
            stream->writeFloatArray(m_reductor, m_reductorSize.x * m_reductorSize.y);
    }

    void clear() {
        m_storage->clear();
    }

    void put(const ImageBlock *block) {
        //Log(EError, "SparseHDRFilm currently does not support standard ImageBlocks. Only use with sparse integrators (for example, 'bdptdiff')");
        const ReductorImageBlock *rd_block = reinterpret_cast<const ReductorImageBlock*>(block);
        if(m_reductor == NULL) {
            Log(EInfo, "Running in raw mode");
        }
        //Log(EInfo, "Accumulating a block %dx%d: %d with %dx%d: ", m_storage->getWidth(), m_storage->getHeight(), m_storage->getData().size(), sp_block->getWidth(), sp_block->getHeight());
        m_storage->put(rd_block);
    }

    //void putSparse(const SparseImageBlock *block) {
    //    m_storage->put(sp_block);
    //}

    void setBitmap(const Bitmap *bitmap, Float multiplier) {
        Log(EError, "setBitmap() not supported for Sparse films");
        //bitmap->convert(m_storage->getBitmap(), multiplier);
    }

    void addBitmap(const Bitmap *bitmap, Float multiplier) {
        /* Currently, only accumulating spectrum-valued floating point images
           is supported. This function basically just exists to support the
           somewhat peculiar film updates done by BDPT */

        Log(EError, "addBitmap(): Not supported for sparse films!");
    }

    bool develop(const Point2i &sourceOffset, const Vector2i &size,
            const Point2i &targetOffset, Bitmap *target) const {
        
        Log(EError, "develop() not supported for HDRSparseFilm");
        exit(1);
        return true;
    }

    void setDestinationFile(const fs::path &destFile, uint32_t blockSize) {
        m_destFile = destFile;
    }

    void develop(const Scene *scene, Float renderTime) {
        if(m_reductor == NULL) {
            Log(EInfo, "develop() running in raw mode");
        }

        if (m_destFile.empty())
            return;

        Log(EDebug, "Developing film ..");

        ref<Bitmap> bitmap;

        fs::path histogramFilename = fs::path(m_destFile);
        histogramFilename.replace_extension(".hist");
        ref<FileStream> hstream = new FileStream(histogramFilename, FileStream::ETruncWrite);

        const std::map<std::tuple<int, int, int, int>, Spectrum> &hdata = m_storage->getHistogramData();
        hstream->writeInt(m_storage->getHistogramHNResolution());
        hstream->writeInt(m_storage->getHistogramHVResolution());
        hstream->writeInt(hdata.size());
        for (auto item : hdata) {
            hstream->writeInt(std::get<0>(item.first));
            hstream->writeInt(std::get<1>(item.first));
            hstream->writeInt(std::get<2>(item.first));
            hstream->writeInt(std::get<3>(item.first));
            hstream->writeFloat(item.second[0]);
            hstream->writeFloat(item.second[1]);
            hstream->writeFloat(item.second[2]);
        }
        hstream->close();

        if (!m_rawMode) {
            fs::path filename = m_destFile;
            std::string properExtension;
            properExtension = ".shds";

            std::string extension = boost::to_lower_copy(filename.extension().string());
            if (extension != properExtension)
                filename.replace_extension(properExtension);

            auto storage_tmp = m_storage->getData();
            int width = m_storage->getWidth();
            int height = m_storage->getHeight();

            Log(EInfo, "Elements: %d", storage_tmp.size()); 
            Log(EInfo, "Width: %d", m_storage->getWidth()); 
            Log(EInfo, "Height: %d", m_storage->getHeight());

            Log(EInfo, "Writing sparse map to \"%s\" ..", filename.string().c_str());
            ref<FileStream> stream = new FileStream(filename, FileStream::ETruncWrite);

            // TODO: Potential issues.
            // Manually write this one.
            

	        std::map<int, Spectrum> storage;
            for(std::map<int, Spectrum>::iterator it = storage_tmp.begin(); it != storage_tmp.end(); it++ ) {

                auto t = it->first;
                auto v = it->second;
            
            //int x = std::get<0>(t);
            //int y = std::get<1>(t);
            //if (x < 0 || x >= width || y < 0 || y >= height) {
                // Discard values outside the frame.
            //    continue;
            //}
    	        storage[t] = v;
	        }
	        storage_tmp.clear();

            Log(EInfo, "Elements: %d", storage.size()); 
            Log(EInfo, "Width: %d", m_storage->getWidth()); 
            Log(EInfo, "Height: %d", m_storage->getHeight());
            Log(EInfo, "Writing Int: ", storage.size());
            stream->writeInt(storage.size());
            Log(EInfo, "Writing Int: ", m_storage->getWidth());
            stream->writeInt(m_storage->getWidth());
            Log(EInfo, "Writing Int: ", m_storage->getHeight());
            stream->writeInt(m_storage->getHeight());
            for(std::map<int, Spectrum>::iterator it = storage.begin(); it != storage.end(); it++ ) {
                auto t = it->first;
                auto v = it->second;

                if(std::isinf(v[0]) || std::isinf(v[1]) || std::isinf(v[2])) 
                    Log(EInfo, "%d -> %f %f %f", t, v[0], v[1], v[2]);
                if (t < 0) {
                    Log(EError, "Attempted to write to an invalid index: ", t);
                }

                stream->writeInt(t);

                stream->writeFloat(v[0]);
                stream->writeFloat(v[1]);
                stream->writeFloat(v[2]);
            }
            Log(EInfo, "Done writing sparse map.");
            Log(EInfo, "Clearing Storage.. %d elements", storage.size());
            m_storage->clear();
            storage.clear();
        } else {
            // Operating in raw mode
            fs::path filename = m_destFile;
            std::string properExtension;
            properExtension = ".raw";

            std::string extension = boost::to_lower_copy(filename.extension().string());
            if (extension != properExtension)
                filename.replace_extension(properExtension);

            Log(EInfo, "Writing sparse map to \"%s\" ..", filename.string().c_str());
            ref<FileStream> stream = new FileStream(filename, FileStream::ETruncWrite);

            // TODO: Potential issues.
            // Manually write this one.
            /*auto storage_tmp = m_storage->getUncompressedData();
            int width = m_storage->getWidth();
            int height = m_storage->getHeight();
	        std::map<std::tuple<int,int,int>, Spectrum> storage;
            for(std::map<std::tuple<int,int,int>, Spectrum>::iterator it = storage_tmp.begin(); it != storage_tmp.end(); it++ ) {

                auto t = it->first;
                auto v = it->second;

                int x = std::get<0>(t);
                int y = std::get<1>(t);
                if (x < 0 || x >= width || y < 0 || y >= height) {
                    // Discard values outside the frame.
                    continue;
                }
    	        storage[t] = v;
	        }
	        storage_tmp.clear();*/

            auto storage = m_storage->getUncompressedData();

            stream->writeInt(storage.size());
            stream->writeInt(m_storage->getWidth());
            stream->writeInt(m_storage->getHeight());
            Log(EInfo, "Elements: %d", storage.size()); 
            Log(EInfo, "Width: %d", m_storage->getWidth()); 
            Log(EInfo, "Height: %d", m_storage->getHeight());
            for(std::map<std::tuple<int,int,int>, Spectrum>::iterator it = storage.begin(); it != storage.end(); it++ ) {
                auto t = it->first;
                auto v = it->second;

                auto x = std::get<0>(t);
                auto y = std::get<1>(t);
                auto n = std::get<2>(t);

                if(std::isinf(v[0]) || std::isinf(v[1]) || std::isinf(v[2]))
                    Log(EInfo, "%d, %d, %d -> %f %f %f", x, y, n, v[0], v[1], v[2]);
                if((x == 114 && y == 109) || (x == 114 && y == 49))
                    Log(EInfo, "%d, %d, %d -> %f %f %f", x, y, n, v[0], v[1], v[2]);

                stream->writeInt(x);
                stream->writeInt(y);
                stream->writeInt(n);

                stream->writeFloat(v[0]);
                stream->writeFloat(v[1]);
                stream->writeFloat(v[2]);
            }
            Log(EInfo, "Done writing raw map.");
            Log(EInfo, "Clearing Storage.. %d elements", storage.size());
            m_storage->clear();
            storage.clear();
        }

    }

    bool hasAlpha() const {
        for (size_t i=0; i<m_pixelFormats.size(); ++i) {
            if (m_pixelFormats[i] == Bitmap::ELuminanceAlpha ||
                m_pixelFormats[i] == Bitmap::ERGBA ||
                m_pixelFormats[i] == Bitmap::EXYZA ||
                m_pixelFormats[i] == Bitmap::ESpectrumAlpha)
                return true;
        }
        return false;
    }

    bool destinationExists(const fs::path &baseName) const {
        std::string properExtension;
        if (m_fileFormat == Bitmap::EOpenEXR)
            properExtension = ".exr";
        else if (m_fileFormat == Bitmap::ERGBE)
            properExtension = ".rgbe";
        else
            properExtension = ".pfm";

        fs::path filename = baseName;
        if (boost::to_lower_copy(filename.extension().string()) != properExtension)
            filename.replace_extension(properExtension);
        return fs::exists(filename);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "HDRReductorFilm[" << endl
            << "  size = " << m_size.toString() << "," << endl
            << "  fileFormat = " << m_fileFormat << "," << endl
            << "  pixelFormat = ";
        for (size_t i=0; i<m_pixelFormats.size(); ++i)
            oss << m_pixelFormats[i] << ", ";
        oss << endl
            << "  channelNames = ";
        for (size_t i=0; i<m_channelNames.size(); ++i)
            oss << "\"" << m_channelNames[i] << "\"" << ", ";
        oss << endl
            << "  componentFormat = " << m_componentFormat << "," << endl
            << "  cropOffset = " << m_cropOffset.toString() << "," << endl
            << "  cropSize = " << m_cropSize.toString() << "," << endl
            << "  banner = " << m_banner << "," << endl
            << "  filter = " << indent(m_filter->toString()) << endl
            << "]";
        return oss.str();
    }
    /*void addChild(const std::string &name, ConfigurableObject *child) {
            /*if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
                if (m_reductor != NULL)
                    Log(EError, "Only one texture can be specified as a reductor!");
                const Properties &props = child->getProperties();

                //m_reductor = static_cast<Texture *>(child);
                printf("addChild() DATA: %f\n", m_reductor[0]);

                if (m_pixelFormats.size() == 1) {
                    m_storage = new ReductorImageBlock(Bitmap::ESpectrumAlphaWeight, m_cropSize, NULL, 0, true, m_reductor.get());
                } else {
                    m_storage = new ReductorImageBlock(Bitmap::EMultiSpectrumAlphaWeight, m_cropSize,
                                    NULL, (int) (SPECTRUM_SAMPLES * m_pixelFormats.size() + 2), true, m_reductor.get());
                }
                
                Log(EInfo, "Found reductor texture: Resolution %dx%d", m_reductorSize.x, m_reductorSize.y);
            } else {
            //    Film::addChild(name, child);
            //}
    }*/

    virtual bool isReductor() { return true; }

    virtual const float* getReductor() { 
        Log(EInfo, "getting reductor");
        return m_reductor; 
    }

    virtual Vector2i getReductorSize() { return m_reductorSize; }

    MTS_DECLARE_CLASS()
protected:
    Bitmap::EFileFormat m_fileFormat;
    std::vector<Bitmap::EPixelFormat> m_pixelFormats;
    std::vector<std::string> m_channelNames;
    Bitmap::EComponentFormat m_componentFormat;
    bool m_banner;
    bool m_attachLog;
    bool m_rawMode;
    fs::path m_destFile;
    Float* m_reductor;
    Vector2i m_reductorSize;
    ref<ReductorImageBlock> m_storage;
    int m_ignoreIndex;
};

MTS_IMPLEMENT_CLASS_S(HDRReductorFilm, false, Film)
MTS_EXPORT_PLUGIN(HDRReductorFilm, "High dynamic range film with reductor texture");
MTS_NAMESPACE_END
