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

#include <mitsuba/render/util.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/plugin.h>
#include <boost/algorithm/string.hpp>
#if defined(WIN32)
#include <mitsuba/core/getopt.h>
#endif

MTS_NAMESPACE_BEGIN

class BSDFExport : public Utility
{
  public:
    void help()
    {
        cout << endl;
        cout << "Synopsis: Exports computed BSDF properties" << endl;
        cout << endl;
        cout << "Usage: mtsutil bsdfexport [options] <Scene XML file or PLY file>" << endl;
        cout << "Options/Arguments:" << endl;
        cout << "   -h             Display this help text" << endl
             << endl;
        cout << "   -n             Export NDF" << endl
             << endl;
        cout << "   -o file        BSDF properties target file" << endl
             << endl;
        cout << "   -t value       Theta_o samples between 0 and pi/2" << endl
             << endl;
        cout << "   -p value       Phi_o samples between 0 and 2.pi" << endl
             << endl;
        cout << "   -k value       Theta_i samples between 0 and pi/2" << endl
             << endl;
    }

    int run(int argc, char **argv)
    {
        ref<FileResolver> fileResolver = Thread::getThread()->getFileResolver();
        int optchar;
        char *end_ptr = NULL;
        char outputfile[200];
        char title[200];
        Float intersectionCost = -1, traversalCost = -1, emptySpaceBonus = -1;
        int stopPrims = -1, maxDepth = -1, exactPrims = -1, minMaxBins = -1;
        bool clip = true, parallel = true, retract = true, fitParameters = false;
        optind = 1;

        int thetaInSamples = 32;
        int thetaOutSamples = 32;
        int phiOutSamples = 9;

        bool exportNDF = false;

        std::map<std::string, std::string, SimpleStringOrdering> parameters;

        /* Parse command-line arguments */
        while ((optchar = getopt(argc, argv, "t:p:o:k:D:l:n")) != -1)
        {
            switch (optchar)
            {
            case 'h':
            {
                help();
                return 0;
            }
            break;
            case 'n':
                exportNDF = true;
                break;
            case 't':
                thetaOutSamples = strtol(optarg, &end_ptr, 10);
                break;
            case 'p':
                phiOutSamples = strtol(optarg, &end_ptr, 10);
                break;
            case 'k':
                thetaInSamples = strtol(optarg, &end_ptr, 10);
                break;
            case 'l':
                strcpy(title, optarg);
                break;
            case 'D':
            {
                std::vector<std::string> param = tokenize(optarg, "=");
                if (param.size() != 2)
                    SLog(EError, "Invalid parameter specification \"%s\"", optarg);
                parameters[param[0]] = param[1];
                SLog(EInfo, "%s\n", optarg);
                SLog(EInfo, "Got a param %s", param.at(0).c_str());
            }
            break;
            case 'o':
                strcpy(outputfile, optarg);
                break;
            };
        }

        if (optind == argc || optind + 1 < argc)
        {
            help();
            return 0;
        }

        ref<Scene> scene;
        ref<ShapeKDTree> kdtree;
        ref<BSDF> bsdf;

        std::string lowercase = boost::to_lower_copy(std::string(argv[optind]));

        if (boost::ends_with(lowercase, ".xml"))
        {
            fs::path
                filename = fileResolver->resolve(argv[optind]),
                filePath = fs::absolute(filename).parent_path(),
                baseName = filename.stem();
            ref<FileResolver> frClone = fileResolver->clone();
            frClone->prependPath(filePath);
            Thread::getThread()->setFileResolver(frClone);
            scene = loadScene(argv[optind], parameters);
            //kdtree = scene->getKDTree();
            //ref<mitsuba::TriMesh> mesh = scene->getMeshes()[0];
            //bsdf = mesh->getBSDF();
            ref<mitsuba::Shape> shape = scene->getShapes()[0];
            bsdf = shape->getBSDF();
        }
        else
        {
            Log(EError, "The supplied scene filename must end in either PLY or XML!");
        }

        /* Show some statistics, and make sure it roughly fits in 80cols */
        Logger *logger = Thread::getThread()->getLogger();
        DefaultFormatter *formatter = ((DefaultFormatter *)logger->getFormatter());
        logger->setLogLevel(EDebug);
        formatter->setHaveDate(false);

        const size_t nRays = 5000000;

        Float best = 0;
        //ostream << thetaInSamples << std::endl;
        //ostream << thetaOutSamples << std::endl;
        //ostream << phiOutSamples << std::endl;

        if (!exportNDF)
        {
            for (int i = 0; i < thetaInSamples; ++i)
            {
                float u = ((float)i / thetaInSamples);
                float theta_i = u * M_PI_2;

                std::stringstream ss;
                ss << outputfile << "-" << i << ".txt";
                std::ofstream ostream(ss.str().c_str());

                ostream << "#phi_in" << 0.0f << std::endl;
                ostream << "#theta_in" << theta_i * (M_1_PI * 180) << std::endl;
                ostream << "#datapoints_in_file" << thetaOutSamples * phiOutSamples << std::endl;
                ostream << "#sample_name \"" << title << " " << theta_i * (M_1_PI * 180) << "\"" << std::endl;

                for (int j = 0; j < thetaOutSamples; ++j)
                {
                    float u = ((float)j / thetaOutSamples);
                    float theta_o = u * u * M_PI_2;

                    for (int k = 0; k < phiOutSamples; ++k)
                    {
                        float u = ((float)k / phiOutSamples);
                        float phi_o = M_PI * 2 * u - M_PI;

                        Vector3f wi = Vector3f(0, sinf(theta_i), cosf(theta_i));
                        Vector3f wo = Vector3f(sinf(theta_o) * sinf(phi_o), sinf(theta_o) * cosf(phi_o), cosf(theta_o));

                        // Warp to specular reflection angle.
                        //Vector3f wm = (wi + wo) / 2;
                        //wm = wm / wm.length();
                        //wo = 2 * dot(wi, wm) * wm - wi;

                        Intersection its;
                        its.p = mitsuba::Point3f(0.f, 0.f, 0.f);
                        its.t = 0.0f;
                        its.uv = mitsuba::Point2f(0.f, 0.f);
                        its.wi = wo;
                        its.shape = scene->getShapes()[0];

                        BSDFSamplingRecord bRec(its, wi, wo);
                        bRec.wi = wi;
                        bRec.wo = wo;

                        Spectrum s = bsdf->eval(bRec);
                        Float lumval = s.getLuminance();

                        ostream << theta_o * (M_1_PI * 180) << " " << phi_o * (M_1_PI * 180) << " " << lumval << std::endl;

                        std::cout << wo.toString() << " " << wi.toString() << " ";

                        std::cout << theta_o * (M_1_PI * 180) << " " << phi_o * (M_1_PI * 180) << " " << lumval * cosf(theta_i) << std::endl;
                    }
                }
                ostream.close();
            }
        } else {
            // Export NDF
            std::stringstream ss;
            ss << outputfile << "-ndf.txt";
            std::ofstream ostream(ss.str().c_str());

            for (int i = 0; i < thetaOutSamples; i++) {
                float u = ((float)i / thetaOutSamples);
                float theta_i = u * M_PI_2;

                Vector3f wi = Vector3f(0, sinf(theta_i), cosf(theta_i));
                Vector3f wo = Vector3f(sinf(theta_i) * sinf(0), sinf(theta_i) * cosf(0), cosf(0));
                Intersection its;

                its.p = mitsuba::Point3f(0.f, 0.f, 0.f);
                its.t = 0.0f;
                its.uv = mitsuba::Point2f(0.f, 0.f);
                its.wi = wo;
                its.shape = scene->getShapes()[0];

                BSDFSamplingRecord bRec(its, wi, wo);
                bRec.wi = wi;
                bRec.wo = wo;

                Spectrum s = bsdf->eval(bRec);
                Float lumval = s.getLuminance();

                ostream << theta_i * (M_1_PI * 180) << " " << lumval << std::endl;

            }

            ostream.close();
        }

        Log(EInfo, "Version 1.1.1", best);
        Thread::getThread()->getLogger()->setLogLevel(EInfo);
        return 0;
    }

    MTS_DECLARE_UTILITY()
};

MTS_EXPORT_UTILITY(BSDFExport, "Exports effective NDF")
MTS_NAMESPACE_END
