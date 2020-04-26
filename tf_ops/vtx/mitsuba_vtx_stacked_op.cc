/* nclude <sys/stat.h> Copyright 2015 The TensorFlow Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// An example Op.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>

#include <stdio.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <mutex>
#include <sys/file.h>

#include "ply_utils.h"

namespace tensorflow
{

REGISTER_OP("Mitsuba")
    .Input("params: float")
    .Input("bsdf: float")
    .Input("tabular_bsdf: float")
    .Input("samplewts: float")
    //.Input("weight: float")
    .Input("depth: float")
    .Input("samples: float")
    .Input("parameter_map: float")
    .Input("serverindex: int32")
    .Input("unitindex: int32")

    .Output("image: float")
    .Doc(R"doc(
BDPT algorithm renders a scene file with given parameters.
)doc");

class MitsubaOp : public OpKernel
{

  public:
    explicit MitsubaOp(OpKernelConstruction *context) : OpKernel(context)
    {
        //std::cout << "Creating Mitsuba operator" << std::endl;
        //printf("Lock address: %08x\n", &glock);
    }

    // Opens a connection to a mitsuba tensorflow server on a given port no.
    void Connect(int port)
    {

        struct sockaddr_in address;
        int valread;
        struct sockaddr_in serv_addr;
        if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
        {
            printf("\n Socket creation error \n");
            exit(-1);
        }

        memset(&serv_addr, '0', sizeof(serv_addr));

        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port);
        //glock.lock();

        //flock(lockfd, LOCK_EX);

        // Convert IPv4 and IPv6 addresses from text to binary form
        if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0)
        {
            printf("\nInvalid address/ Address not supported \n");
            //glock.unlock();
            exit(-1);
        }

        if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
        {
            printf("\nConnection Failed to port %d\n", port);
            //glock.unlock();
            exit(-1);
        }
    }

    // Close connection.
    void Disconnect()
    {
        close(sock);
        //flock(lockfd, LOCK_UN);
    }

    void Compute(OpKernelContext *context) override
    {

        // Output a scalar string.
        Tensor *output_tensor = nullptr;
        std::vector<int> v = {256, 256};

        const Tensor &params = context->input(0);
        const Tensor &bsdf = context->input(1);
        const Tensor &tabular_bsdf = context->input(2);
        const Tensor &samplewts = context->input(3);
        const Tensor &depth = context->input(4);
        const Tensor &samples = context->input(5);
        const Tensor &map_tensor = context->input(6);
        const Tensor &serverindex = context->input(7);
        const Tensor &unitindex = context->input(8);

        // TODO: WARN: Currently the shape is fixed at 256x256.
        // Implementation will break if the mitsuba scene files render at
        // any other resolution.
        auto paramvec = params.matrix<float>();
        auto bsdfvec = bsdf.vec<float>();
        auto tabularbsdfvec = tabular_bsdf.tensor<float, 3>();
        auto samplewtsvec = samplewts.vec<float>();
        auto dval = depth.scalar<float>();
        auto sval = samples.scalar<float>();

        auto unitIndex = unitindex.scalar<int>();
        auto serverIndex = serverindex.matrix<int>();

        //std::stringstream lss; 
        //lss << "/tmp/mtsintlock-" << serverIndex(0) << ".lock";
        //lockfd = open(lss.str().c_str(), O_RDONLY);
        //std::cout << "Waiting for lock " << lss.str() << std::endl;

        std::vector<tfply::Triangle> tris;
        std::vector<tfply::Vertex> vtxs;
        std::vector<tfply::Normal> normals;
        std::vector<tfply::Normal> modded_normals;

        auto mapvec = map_tensor.matrix<float>();
        int num_normals = map_tensor.dim_size(0);
        int numImages = params.dim_size(1);
        int numParameters = params.dim_size(0);

        // Ensure independent parameters and server indices.
        // We assume for now that we use the same set of normals
        // as well as BSDFs.
        assert(params.dim_size(1) == serverindex.dim_size(1));

        for (int i = 0; i < num_normals; i++)
        {
            modded_normals.push_back(tfply::Normal(mapvec(i, 0), mapvec(i, 1), mapvec(i, 2)));
        }

        std::stringstream plyfiless;

        //plyfiless << "/tmp/mts_mesh_intensity_slot_" << unitIndex(0) << ".ply";
        // Use only the first slot. Share the ply.
        plyfiless << "/tmp/mts_mesh_intensity_slot_0.ply";

        // Important. Check to make sure it's working as expected.
        tfply::ReadPLY("/tmp/mts_srcmesh.ply", vtxs, normals, tris);
        tfply::WritePLY(plyfiless.str().c_str(), vtxs, modded_normals, tris);

        assert(paramvec(0) == paramvec(1)); // Make the icky assumption that paramvec[0,1] is W and H.
                                            // and the assumption that W == H

        int W = static_cast<int>(paramvec(0, 0));
        int H = static_cast<int>(paramvec(1, 0));

        TensorShape shape;
        shape.AddDim(W);
        shape.AddDim(H);
        shape.AddDim(numImages);

        OP_REQUIRES_OK(context,
                       context->allocate_output(0, shape, &output_tensor));

        for(int img = 0; img < numImages; img++){
            //std::cout << "Requesting image " << img << std::endl;
            // Make a list of parameter values to send.
            // Note, order is important.
            std::vector<float> rdata;
            rdata.push_back(img); // ply file index
            rdata.push_back(dval(0)); // depth
            rdata.push_back(sval(0)); // sampleCount

            // Push general parameters (non-gradient)
            // Specific to the image.
            for (int i = 0; i < numParameters; i++)
                rdata.push_back(paramvec(i, img));

            // Push BSDF parameters. (differentiable)
            // Independent of the image
            for (int i = 0; i < bsdfvec.size(); i++)
                rdata.push_back(bsdfvec(i));
            
            // Push BSDF sample weights.
            for (int i = 0; i < samplewtsvec.size(); i++)
                rdata.push_back(samplewtsvec(i));
            
            std::stringstream ss;
            ss << "/tmp/tabular-bsdf-" << img << ".binary";
            WriteTabularBSDF(tabular_bsdf, ss.str().c_str());

            std::cout << "[Sparse Op] Finished writing Tabular BSDF to " << std::endl;
            std::cout << ss.str().c_str() << std::endl;

            // Shoot a request for the image and the respective 
            // mtstensorflow server.
            RequestImage(img, rdata, serverIndex(0, img));
        }

        for(int img = 0; img < numImages; img++){
            //std::cout << "Fetching image " << img << std::endl;
            FetchImage(img, output_tensor, context, serverIndex(0, img));
        }
    }

    void RequestImage(int n, std::vector<float> &rdata, int port)
    {
        Connect(port); // Server indices.
        short t = rdata.size();
        // Send parameters to renderer.
        int a = send(sock, &t, sizeof(short), 0);
        for (int i = 0; i < rdata.size(); i++)
        {
            float k = rdata.at(i);
            int a = send(sock, &k, sizeof(float), 0);
        }
        Disconnect();
    }

    void WriteTabularBSDF(const Tensor &tensor, std::string fname)
    {
        std::ofstream outfile(fname);
        auto output_tensor = tensor.tensor<float, 3>();

        int resolutionThetaD = tensor.dim_size(0);
        int resolutionThetaH = tensor.dim_size(1);
        int resolutionPhiD = tensor.dim_size(2);
        outfile.write(reinterpret_cast<const char *>(&resolutionThetaD), sizeof(int));
        outfile.write(reinterpret_cast<const char *>(&resolutionThetaH), sizeof(int));
        outfile.write(reinterpret_cast<const char *>(&resolutionPhiD), sizeof(int));

        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < resolutionThetaD; i++){
                for (int j = 0; j < resolutionThetaH; j++){
                    for (int k = 0; k < resolutionPhiD; k++){
                        double f = static_cast<double>(output_tensor(i, j, k));
                        // Repeat thrice for RGB contents (this version only supports grayscale values)
                        outfile.write(reinterpret_cast<const char *>(&f), sizeof(double));
                    }
                }
            }
        }

        outfile.close();
    }

    void FetchImage(int n, Tensor *output_tensor, OpKernelContext *context, int port){
        // Read the image from a FIFO pipe.
        std::stringstream mtsout;
        mtsout << "/tmp/mtsout-" << port << ".hds";

        int imgfile = open(mtsout.str().c_str(), O_RDONLY);

        int r, c, dims;
        int k;
        k = read(imgfile, &r, sizeof(int));
        k = read(imgfile, &c, sizeof(int));
        k = read(imgfile, &dims, sizeof(int));

        if (dims != 1)
        {
            std::cerr << "[Sparse Op] Instance:" << n << " Dimensions not equal to 1 Dims: " << dims << std::endl;
            Disconnect();
            exit(1);
        }

        //TODO: Free memory.
        float *data = new float[r * c * dims];

        if (!data)
        {
            std::cerr << "[Sparse Op] Out of memory" << std::endl;
            Disconnect();
            exit(1);
        }

        int sofar = 0;
        int totalbytes = sizeof(float) * r * c;
        while (true)
        {
            int k = read(imgfile, ((char *)data) + sofar, totalbytes - sofar);
            if (k == -1)
            {
                std::cerr << "[Sparse Op] Instance:" << n << " ERROR while reading: " << strerror(errno) << std::endl;
                Disconnect();
                exit(1);
            }

            if (k == 0)
                break;
            sofar += k;
            if (sofar == totalbytes)
                break;
        }

        // TODO: Temporary.
        if (r != c)
        {
            std::cerr << "[Mitsuba Tensorflow Operator] WARNING: Currently no support for R != C image dimensions" << std::endl;
        }

        auto output = output_tensor->tensor<float, 3>();
        // Set output values to data obtained.
        // TODO: Check if the values are row major or column major.
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                output(i, j, n) = data[i * c + j];
            }
        }

        delete[] data;
        close(imgfile);

    }
    static std::mutex glock;
    int lockfd;
  private:
    int sock;
};
std::mutex MitsubaOp::glock;

REGISTER_KERNEL_BUILDER(Name("Mitsuba").Device(DEVICE_CPU), MitsubaOp);

} // namespace tensorflow
