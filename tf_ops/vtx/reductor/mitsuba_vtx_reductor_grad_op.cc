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

class FloatSet3
{
  public:
    float x;
    float y;
    float z;
    FloatSet3(float x, float y, float z) : x(x), y(y), z(z)
    {
    }
    FloatSet3() : x(0), y(0), z(0)
    {
    }
};
REGISTER_OP("MitsubaGrad")
    .Input("grad: float")
    //.Input("alpha: float")
    //.Input("weight: float")
    .Input("params: float") // Linear parameter set.
    .Input("bsdf: float")   // BSDF parameter set.
    .Input("samplewts: float")   // BSDF mixture sample weights.

    .Input("depth: float")
    .Input("samples: float")
    .Input("parameter_map: float") // N-dimensional parameter set.
    .Input("serverindex: int32") // Index used to select a server.
    .Input("unitindex: int32") // Index used to differentiate this instance from others when writing debug output.
    //.Output("grad_alpha: float")
    //.Output("grad_weight: float")
    .Output("grad_params: float")
    .Output("grad_bsdf: float")
    .Output("grad_samplewts: float")
    .Output("grad_depth: float")
    .Output("grad_samples: float")
    .Output("grad_parameter_map: float")
    .Output("grad_serverindex: int32")
    .Output("grad_unitindex: int32")
    //.Attr("desc: string")
    //.Attr("T: float")
    .Doc(R"doc(
BDPT gradient algorithm
)doc");

class MitsubaGradOp : public OpKernel
{
  public:
    explicit MitsubaGradOp(OpKernelConstruction *context) : OpKernel(context)
    {
    }

    void Connect(int port)
    {
        //glock.lock();
        flock(lockfd, LOCK_EX);
        //std::cerr << "[Sparse Grad Op] Opening a port: " << port << std::endl;
        struct sockaddr_in address;
        //int sock = 0, valread;
        int valread;
        struct sockaddr_in serv_addr;
        //char *hello = "Hello from client";
        //char buffer[1024] = {0};
        if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
        {
            std::cerr << "\n Socket creation error \n";
            //return -1;
            exit(1);
        }

        memset(&serv_addr, '0', sizeof(serv_addr));

        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port);

        // Convert IPv4 and IPv6 addresses from text to binary form
        if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0)
        {
            std::cerr << "\nInvalid address/ Address not supported \n";
            //return -1;
            exit(1);
        }

        if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
        {
            std::cerr << "\nConnection Failed \n";
            exit(1);
        }

        //std::cout << "Finished connecting." << std::endl;
    }

    // Takes no attr ID argument as an input because
    // it handles every attr.
    void ComputeAttrSetGrad(int i, std::vector<float> attrs, const Tensor &grad_output_tensor, const Tensor &map_tensor, const Tensor &bsdf_tensor, std::vector<FloatSet3> &dI, std::vector<FloatSet3> &dBSDF, int port, int uidx)
    {

        std::stringstream reductorfile;
        reductorfile << "/tmp/reductor-" << uidx << ".hds";
        // Write to output.
        WriteHDSImage(grad_output_tensor, reductorfile.str().c_str());

        short numvals = attrs.size();
        int a = send(sock, &numvals, sizeof(short), 0);

        for (int i = 0; i < attrs.size(); i++)
        {
            float k = attrs.at(i);
            // stream them out.
            int a = send(sock, &k, sizeof(float), 0);
        }

        // Read a Sparse HDR Stream.
        std::stringstream mtsgradout;

        mtsgradout << "/tmp/mtsgradout-" << port << ".shds";
        //std::cout << "[Sparse Grad Op] Instance" << i << ", Reading from file: " << mtsgradout.str() << std::endl;
        imgfile = open(mtsgradout.str().c_str(), O_RDONLY);

        int r, c, n;
        int k;
        k = read(imgfile, &n, sizeof(int));
        k = read(imgfile, &r, sizeof(int));
        k = read(imgfile, &c, sizeof(int));

        //std::cerr << "[Sparse Grad Op] SHDS: " << n << " elements, " << r << "x" << c << " resolution\n";
        if (sizeof(int) != sizeof(float))
        {
            // ERROR.
            std::cerr << "[Sparse Grad Op] Expected size of int to be equal to size of float." << std::endl;
            exit(1);
        }
        struct DataPoint
        {
            int n;
            float dx;
            float dy;
            float dz;
        };

        if (sizeof(DataPoint) != 4 * sizeof(int))
        {
            std::cerr << "[Sparse Grad Op] Expected size of DataPoint struct to be exactly 6 times the size of an int (and a float.)" << std::endl;
            exit(1);
        }

        DataPoint *data = new DataPoint[n];

        if (!data)
        {
            std::cerr << "[Sparse Grad Op] Out of memory" << std::endl;
            exit(1);
        }

        int sofar = 0;
        int totalbytes = sizeof(DataPoint) * n;
        while (true)
        {
            int k = read(imgfile, ((char *)data) + sofar, totalbytes - sofar);
            if (k == -1)
            {
                std::cerr << "[Sparse Grad Op] ERROR while reading: " << strerror(errno) << std::endl;
                exit(1);
            }

            if (k == 0)
                break;
            sofar += k;
            if (sofar >= totalbytes)
                break;
        }

        //auto dI = grad_map_tensor->matrix<float>();
        int diff_size = map_tensor.dim_size(0);
        int numBsdfParameters = bsdf_tensor.dim_size(0);
        //int diff_y_size = map_tensor.dim_size(1);

        if (map_tensor.dim_size(1) != 3)
        {
            std::cerr << "[Sparse Grad Op] ERROR: Expected 2nd dimension to be 3" << std::endl;
            exit(1);
        }

        for (int d = 0; d < n; d++)
        {
            DataPoint p = data[d];
            // TODO: Check if x and y aren't accidentally flipped.
            //for(int j = 0; j < c; j++) {
            //int dy = p.n / diff_x_size;
            //int dx = p.n % diff_x_size;

            if (p.n < 0 || p.n >= diff_size + numBsdfParameters)
            {
                std::cerr << "[AttrSetGrad] Illegal parameter index: '" << p.n << "'" << std::endl;
                exit(1);
            }

            if (std::isinf(p.dx) || std::isinf(p.dy) || std::isinf(p.dz))
            {
                std::cerr << "p INF at " << p.n << std::endl;
                exit(1);
            }

            if (p.n >= numBsdfParameters)
            {
                dI[p.n - numBsdfParameters].x = p.dx;
                dI[p.n - numBsdfParameters].y = p.dy;
                dI[p.n - numBsdfParameters].z = p.dz;
            }
            else
            {
                dBSDF[p.n].x = p.dx;
                dBSDF[p.n].y = p.dy;
                dBSDF[p.n].z = p.dz;
            }
        }

        delete[] data;
        close(imgfile);
        //return dI;
    }

    void Disconnect()
    {

        close(sock);
        //glock.unlock();
        //std::cout << "Releasing lock " << std::endl;
        flock(lockfd, LOCK_UN);

    }

    void WriteHDSImage(const Tensor &tensor, std::string image)
    {
        std::ofstream outfile(image);
        auto output_tensor = tensor.tensor<float, 3>();

        int width = tensor.dim_size(0);
        int height = tensor.dim_size(1);
        int channels = 1;

        outfile.write(reinterpret_cast<const char *>(&width), sizeof(int));
        outfile.write(reinterpret_cast<const char *>(&height), sizeof(int));
        outfile.write(reinterpret_cast<const char *>(&channels), sizeof(int));

        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                float f = output_tensor(i, j, 0);
                outfile.write(reinterpret_cast<const char *>(&f), sizeof(float));
            }
        }

        outfile.close();
    }

    void Compute(OpKernelContext *context) override
    {

        //std::cout << "Computing " << std::endl;
        //std::cout << "Gradient render job" << std::endl;

        // Output a scalar string.
        //Tensor* output_tensor = nullptr;
        std::vector<int> v = {256, 256};

        const Tensor &grad_output_tensor = context->input(0);
        //const Tensor& alpha = context->input(1);
        //const Tensor& weight = context->input(2);
        const Tensor &params = context->input(1);
        const Tensor &bsdf = context->input(2);
        const Tensor &samplewts = context->input(3);
        const Tensor &depth = context->input(4);
        const Tensor &samples = context->input(5);
        const Tensor &map_tensor = context->input(6);
        const Tensor &serverindex = context->input(7);
        const Tensor &unitindex = context->input(8);

        int numBsdfParameters = bsdf.shape().dim_size(0);
        //printf("[TF Grad Op] BSDF parameters: %d\n", numBsdfParameters);

        //Tensor* grad_alpha_tensor;
        //Tensor* grad_weight_tensor;
        Tensor *grad_params_tensor;
        Tensor *grad_depth_tensor;
        Tensor *grad_bsdf_tensor;
        Tensor *grad_samplewts_tensor;
        Tensor *grad_samples_tensor;
        Tensor *grad_map_tensor;
        Tensor *grad_serverindex_tensor;
        Tensor *grad_unitindex_tensor;

        //OP_REQUIRES_OK(context,
        //        context->allocate_output(0, alpha.shape(), &grad_alpha_tensor));
        //OP_REQUIRES_OK(context,
        //        context->allocate_output(1, weight.shape(), &grad_weight_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, params.shape(), &grad_params_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(1, bsdf.shape(), &grad_bsdf_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(2, samplewts.shape(), &grad_samplewts_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(3, depth.shape(), &grad_depth_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(4, samples.shape(), &grad_samples_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(5, map_tensor.shape(), &grad_map_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(6, serverindex.shape(), &grad_serverindex_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(7, unitindex.shape(), &grad_unitindex_tensor));

        auto paramsval = params.vec<float>();
        auto bsdfval = bsdf.vec<float>();
        auto samplewtval = samplewts.vec<float>();
        auto depthval = depth.scalar<float>();
        auto samplesval = samples.scalar<float>();
        auto serverindexval = serverindex.vec<int>();
        auto unitindexval = unitindex.scalar<int>();

        std::stringstream lss; 
        lss << "/tmp/mtsgradlock-" << serverindexval(1) << ".lock";
        lockfd = open(lss.str().c_str(), O_RDONLY);
        std::cout << "Waiting for lock " << lss.str() << std::endl;

        // Given input.
        auto grad_output = grad_output_tensor.tensor<float, 3>();

        // Outputs that need to be calculated.
        //auto grad_alpha = grad_alpha_tensor->scalar<float>();
        //auto grad_weight = grad_weight_tensor->scalar<float>();
        auto grad_params = grad_params_tensor->vec<float>();
        auto grad_bsdf = grad_bsdf_tensor->vec<float>();
        auto grad_samplewts = grad_samplewts_tensor->vec<float>();
        auto grad_depth = grad_depth_tensor->scalar<float>();
        auto grad_samples = grad_samples_tensor->scalar<float>();
        auto grad_serverindex = grad_serverindex_tensor->vec<int>();
        auto grad_unitindex = grad_unitindex_tensor->scalar<float>();

        // Attribute list that needs to be sent to the
        // renderer.
        std::vector<float> attrs;
        //attrs.push_back(alphaval(0));
        //attrs.push_back(weightval(0));
        //attrs.push_back(1.0f - weightval(0));
        //for(int i = 0; i < paramsval.size(); i++) {
        //    attrs.push_back(paramsval(i));
        //}
        attrs.push_back(unitindexval(0)); // Push the index code to pick the right ply file.
        attrs.push_back(depthval(0));
        attrs.push_back(samplesval(0));
        for (int i = 0; i < paramsval.size(); i++)
        {
            attrs.push_back(paramsval(i));
        }

        for (int i = 0; i < bsdfval.size(); i++)
        {
            attrs.push_back(bsdfval(i));
        }

        for (int i = 0; i < samplewtval.size(); i++)
        {
            attrs.push_back(samplewtval(i));
        }


        /*(for(int i = 0; i < paramsval.size(); i++) {
                        Connect(7555);
                        grad_params(i) = ComputeAttrGrad(i, attrs, grad_output_tensor);
                        Disconnect();
                    }*/

        /*Connect(7555);
                    grad_alpha(0) = ComputeAttrGrad(1, attrs, grad_output_tensor);
                    Disconnect();

                    Connect(7555);
                    grad_weight(0) = ComputeAttrGrad(0, attrs, grad_output_tensor);
                    Disconnect();*/

        for (int i = 0; i < paramsval.size(); i++)
        {
            grad_params(i) = 0.f;
        }

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
        // Write the parameter map out to HDS files and also
        // write out the reference HDS files.
        //WriteParameterMap("/tmp/mts_nmap.hds", map_tensor);
        //WriteIndexMap("/tmp/mts_idx.hds", map_tensor);

        // Create an array to accumulate differentials for each normal
        std::vector<FloatSet3> dI(num_normals);
        // Create an array to accumulate differentials for each BSDF parameter
        std::vector<FloatSet3> dBSDF(numBsdfParameters);

        // Initialize differentials.
        for (int i = 0; i < num_normals; i++)
        {
            dI[i] = FloatSet3(0, 0, 0);
        }
        for (int i = 0; i < numBsdfParameters; i++)
        {
            dBSDF[i] = FloatSet3(0, 0, 0);
        }

        //for (int i = 0; i < instanceCount(0); i++)
        //{
        // Accumulate multiple values.
        std::stringstream plyfiless;
        plyfiless << "/tmp/mts_mesh_gradient_slot_" << unitindexval(0) << ".ply";

        Connect(serverindexval(1));
        tfply::ReadPLY("/tmp/mts_srcmesh.ply", vtxs, normals, tris);
        tfply::WritePLY(plyfiless.str().c_str(), vtxs, modded_normals, tris);
        ComputeAttrSetGrad(0, attrs, grad_output_tensor, map_tensor, bsdf, dI, dBSDF, serverindexval(1), unitindexval(0));
        Disconnect();
        //}

        auto dI_tensor = grad_map_tensor->matrix<float>();
        auto dBSDF_tensor = grad_bsdf_tensor->vec<float>();
        //std::cerr << "Tensor index count: " << num_normals << std::endl;
        //std::cerr << "Instance count: " << instanceCount(0) << std::endl;
        for (int i = 0; i < dI.size(); i++)
        {
            // Compute maginitude and deviation
            float nx = dI[i].x;
            float ny = dI[i].y;
            float nz = dI[i].z;

            float ox = modded_normals[i].x;
            float oy = modded_normals[i].y;
            float oz = modded_normals[i].z;

            float mag = sqrt(nx * nx + ny * ny + nz * nz) + 5e-6; // Padding epsilon to prevent div-by-0 errors.
            float nnx = nx / mag;
            float nny = ny / mag;
            float nnz = nz / mag;

            float dev = nnx * ox + nny * oy + nnz * oz;

            // Deliver adjusted gradients for much faster convergence.
            //dI_tensor(i,0) = (nnx - ox);
            //dI_tensor(i,1) = (nny - oy);
            //dI_tensor(i,2) = (nnz - oz);

            // Original gradients.
            dI_tensor(i, 0) = nx;
            dI_tensor(i, 1) = ny;
            dI_tensor(i, 2) = nz;
        }

        for (int i = 0; i < dBSDF.size(); i++)
        {
            // Compute maginitude and deviation
            float bx = dBSDF[i].x;
            float by = dBSDF[i].y;
            float bz = dBSDF[i].z;

            // Use only one channel since they all must be equal (duplicated to fit 3-channel requirements)
            dBSDF_tensor(i) = bx;
        }

        std::stringstream ss;
        ss << "/tmp/mts_mesh_gradients-" << unitindexval(0) << ".ply";
        std::vector<tfply::Normal> gradients;
        for (int i = 0; i < dI.size(); i++)
        {
            gradients.push_back(tfply::Normal(dI_tensor(i, 0), dI_tensor(i, 1), dI_tensor(i, 2)));
        }

        tfply::WritePLY(ss.str(), vtxs, gradients, tris);

        // std::cerr << grad_output_tensor
        //int dbgnormal = 12345;
        //std::cerr << dI_tensor(dbgnormal, 0) << "," << dI_tensor(dbgnormal,1) << "," << dI_tensor(dbgnormal,2) << std::endl;
        /*for(int z = 0; z < 3; z++) { 
                        for(int x = 0; x < dI.size(); x++) {
                            //for(int y = 0; y < diff_y_size; y++) {
                                std::cerr << dI_tensor(x, z);
                            //}
                            std::cerr << std::endl;
                        }
                        std::cerr << std::endl << std::endl;
                    } */

        // Gradients for regular stuff.
        grad_depth(0) = 0.f;
        grad_samples(0) = 0.f;
        //grad_instances(0) = 0.f;
        grad_unitindex(0) = 0.f;

        for(int i = 0; i < samplewtval.size(); i++){
            grad_samplewts(i) = 0.f;
        }

        grad_serverindex(0) = 0.f;
        grad_serverindex(1) = 0.f;
    }
    void Check(int k, size_t sz)
    {
        if (k != sz)
        {
            std::cout << "Didn't write the correct bytes. Actual:" << sz << " Written:" << k << std::endl;
            exit(1);
        }
    }

    static std::mutex glock;
    int lockfd;
  private:
    int sock;
    int imgfile;
};
std::mutex MitsubaGradOp::glock;
REGISTER_KERNEL_BUILDER(Name("MitsubaGrad").Device(DEVICE_CPU), MitsubaGradOp);

} // namespace tensorflow
