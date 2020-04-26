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

        flock(lockfd, LOCK_EX);

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
        //std::cerr << "[Sparse Op] [Connect] Closing port: " << std::endl;
        //glock.unlock();
        close(sock);
        //glock.unlock();
        //std::cout << "Releasing lock " << std::endl;
        flock(lockfd, LOCK_UN);
    }

    void Compute(OpKernelContext *context) override
    {
        
        // Output a scalar string.
        Tensor *output_tensor = nullptr;
        std::vector<int> v = {256, 256};

        //std::cout << "Compute" << std::endl;

        //const Tensor& alpha = context->input(0);
        //const Tensor& weight = context->input(1);
        const Tensor &params = context->input(0);
        const Tensor &bsdf = context->input(1);
        const Tensor &samplewts = context->input(2);
        const Tensor &depth = context->input(3);
        const Tensor &samples = context->input(4);
        const Tensor &map_tensor = context->input(5);
        const Tensor &serverindex = context->input(6);
        const Tensor &unitindex = context->input(7);


        // TODO: WARN: Currently the shape is fixed at 256x256.
        // Implementation will break if the mitsuba scene files render at
        // any other resolution.

        //auto output = output_tensor->tensor<float, 3>();
        //auto alphaval = alpha.scalar<float>();
        //auto wval = weight.scalar<float>();
        auto paramvec = params.vec<float>();
        auto bsdfvec = bsdf.vec<float>();
        auto samplewtsvec = samplewts.vec<float>();
        auto dval = depth.scalar<float>();
        auto sval = samples.scalar<float>();

        auto unitIndex = unitindex.scalar<int>();
        auto serverIndex = serverindex.vec<int>();

        //std::cout << "Compute called. " << unitIndex(0) << " " << serverIndex(0) << std::endl;

        std::stringstream lss; 
        lss << "/tmp/mtsintlock-" << serverIndex(0) << ".lock";
        lockfd = open(lss.str().c_str(), O_RDONLY);
        std::cout << "Waiting for lock " << lss.str() << std::endl;

        std::vector<tfply::Triangle> tris;
        std::vector<tfply::Vertex> vtxs;
        std::vector<tfply::Normal> normals;
        std::vector<tfply::Normal> modded_normals;

        auto mapvec = map_tensor.matrix<float>();
        int num_normals = map_tensor.dim_size(0);
        for (int i = 0; i < num_normals; i++)
        {
            modded_normals.push_back(tfply::Normal(mapvec(i, 0), mapvec(i, 1), mapvec(i, 2)));
        }

        //map_tensor.dim_size(0);
        //map_tensor.dim_size(1);
        // Write the PLY file with modified normals.

        //WriteParameterMap("/tmp/mts_nmap.hds", map_tensor);
        //WriteIndexMap("/tmp/mts_idx.hds", map_tensor);

        // Make a list of parameter values to send.
        // Note, order is important.
        std::vector<float> rdata;
        //rdata.push_back(alphaval(0));   // alpha
        //rdata.push_back(wval(0));       // weight1
        //rdata.push_back(1-wval(0));     // weight2

        // Put all the new parameters into the list.
        //std::cout << "PARAMS: " << paramvec.size() << std::endl;
        //for(int i = 0; i < paramvec.size(); i++){
        //    rdata.push_back(paramvec(i));
        //}
        rdata.push_back(unitIndex(0)); // ply file index
        rdata.push_back(dval(0)); // depth
        rdata.push_back(sval(0)); // sampleCount

        // Push general parameters (non-gradient)
        for (int i = 0; i < paramvec.size(); i++)
        {
            rdata.push_back(paramvec(i));
        }

        // Push BSDF parameters. (differentiable)
        for (int i = 0; i < bsdfvec.size(); i++)
        {
            rdata.push_back(bsdfvec(i));
        }

        // Push sample weights.
        for (int i = 0; i < samplewtsvec.size(); i++)
        {
            rdata.push_back(samplewtsvec(i));
        }

        Connect(serverIndex(0)); // Server indices.
        std::stringstream plyfiless;

        // TODO: WARN: PROBLEM.
        plyfiless << "/tmp/mts_mesh_intensity_slot_" << unitIndex(0) << ".ply";

        tfply::ReadPLY("/tmp/mts_srcmesh.ply", vtxs, normals, tris);
        tfply::WritePLY(plyfiless.str().c_str(), vtxs, modded_normals, tris);
        FetchImage(0, rdata, output_tensor, context, serverIndex(0));
        Disconnect();
        //}
    }

    void FetchImage(int n, std::vector<float> &rdata, Tensor *output_tensor, OpKernelContext *context, int port)
    {
        //std::cout << "MTS_OP" << std::endl;
        short t = rdata.size();
        //std::cerr << "[Sparse Op] Sending size: " << t << std::endl;
        // Send parameters to renderer.
        int a = send(sock, &t, sizeof(short), 0);
        for (int i = 0; i < rdata.size(); i++)
        {
            float k = rdata.at(i);
            //std::cerr << "[Sparse Op] Sending: " << k << std::endl;
            int a = send(sock, &k, sizeof(float), 0);
        }

        // Read the image from a FIFO pipe.
        std::stringstream mtsout;
        mtsout << "/tmp/mtsout-" << port << ".hds";
        //std::cout << "[Sparse Op] Instance " << n << ", Reading from " << mtsout.str().c_str() << std::endl;
        int imgfile = open(mtsout.str().c_str(), O_RDONLY);

        int r, c, dims;
        int k;
        k = read(imgfile, &r, sizeof(int));
        k = read(imgfile, &c, sizeof(int));
        k = read(imgfile, &dims, sizeof(int));

        TensorShape shape;
        shape.AddDim(r);
        shape.AddDim(c);
        shape.AddDim(1);

        OP_REQUIRES_OK(context,
                       context->allocate_output(0, shape, &output_tensor));

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
    /*void WriteParameterMap(std::string fname, const Tensor& map_tensor) {

                    FILE* mapfile = fopen(fname.c_str(), "w");
                    if(!mapfile) {
                        std::cout << "Failed to write normal map: " << fname.c_str() << std::endl;
                        exit(1);
                    }

                    std::cout << "Writing to file"
                    
                    int dimx = map_tensor.dim_size(0);
                    int dimy = map_tensor.dim_size(1);
                    int dimz = map_tensor.dim_size(2);

                    float *fdata = new float[dimx * dimy * dimz];
                    auto tdata = map_tensor.tensor<float, 3>();
                    int curr = 0;
                    for(int x = 0; x < dimx; x++) {
                        for(int y = 0; y < dimy; y++) {
                            for(int z = 0; z < dimz; z++) {
                                fdata[curr++] = tdata(x, y, z);
                            }
                        }
                    }
                    
                    int k;
                    k = fwrite(&dimx, sizeof(int), 1, mapfile);
                    k = fwrite(&dimy, sizeof(int), 1, mapfile);
                    k = fwrite(&dimz, sizeof(int), 1, mapfile);
                    
                    int bytesToWrite = sizeof(float) * dimx * dimy * dimz;
                    int bytesWritten = fwrite(fdata, sizeof(float), dimx * dimy * dimz, mapfile);

                    if(bytesToWrite != bytesWritten) {
                        std::cout << "ERROR: Couldn't write all teh bytes 0_0" << std::endl;
                        exit(1);
                    }

                    fclose(mapfile);

                }*/

    /*void WriteIndexMap(std::string fname, const Tensor& map_tensor) {
                    FILE* mapfile = fopen(fname.c_str(), "w");
                    if(!mapfile) {
                        std::cout << "Failed to write index map: " << fname.c_str() << std::endl;
                        exit(1);
                    }
                    
                    int dimx = map_tensor.dim_size(0);
                    int dimy = map_tensor.dim_size(1);
                    int dimz = map_tensor.dim_size(2);

                    float *fdata = new float[dimx * dimy * dimz];
                    auto tdata = map_tensor.tensor<float, 3>();
                    int curr = 0;
                    for(int x = 0; x < dimx; x++) {
                        for(int y = 0; y < dimy; y++) {
                            for(int z = 0; z < dimz; z++) {
                                fdata[curr++] = static_cast<float>(y * dimx + x);
                            }
                        }
                    }
                    
                    int k;
                    k = fwrite(&dimx, sizeof(int), 1, mapfile);
                    k = fwrite(&dimy, sizeof(int), 1, mapfile);
                    k = fwrite(&dimz, sizeof(int), 1, mapfile);
                    
                    int bytesToWrite = sizeof(float)  * dimx * dimy * dimz;
                    int bytesWritten = fwrite(fdata, sizeof(float), dimx * dimy * dimz, mapfile);

                    if(bytesToWrite != bytesWritten) {
                        std::cout << "ERROR: Couldn't write all teh bytes 0_0" << std::endl;
                        exit(1);
                    }

                    fclose(mapfile);

                }*/
    static std::mutex glock;
    int lockfd;
  private:
    int sock;
};
std::mutex MitsubaOp::glock;

REGISTER_KERNEL_BUILDER(Name("Mitsuba").Device(DEVICE_CPU), MitsubaOp);

} // namespace tensorflow
