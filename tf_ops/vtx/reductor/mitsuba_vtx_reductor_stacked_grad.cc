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
    .Input("tabular_bsdf: float")   // BSDF parameter set.
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
    .Output("grad_tabular_bsdf: float")
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
        //flock(lockfd, LOCK_EX);
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
    void RequestGrad(int i, 
                std::vector<float> attrs, 
                const Tensor &grad_output_tensor, 
                const Tensor &map_tensor, 
                const Tensor &bsdf_tensor, 
                const Tensor &tabular_bsdf_tensor,
                int port, 
                int uidx)
    {

        Connect(port);
        std::stringstream reductorfile;
        reductorfile << "/tmp/reductor-" << uidx << ".hds";
        std::stringstream tabularfile;
        tabularfile << "/tmp/tabular-bsdf-" << uidx << ".binary";
        std::cout << "[Sparse Grad Op] Finished writing Tabular BSDF to " << std::endl;
        std::cout << tabularfile.str().c_str() << std::endl;

        // Write to output.
        WriteHDSImage(grad_output_tensor, reductorfile.str().c_str(), i);
        WriteTabularBSDF(tabular_bsdf_tensor, tabularfile.str().c_str());

        short numvals = attrs.size();
        int a = send(sock, &numvals, sizeof(short), 0);

        for (int j = 0; j < attrs.size(); j++)
        {
            float k = attrs.at(j);
            // stream them out.
            int a = send(sock, &k, sizeof(float), 0);
        }
        Disconnect();
    }

    void FetchGrad(int i,
        const Tensor &map_tensor,
        const Tensor &bsdf_tensor,
        const Tensor &tabular_bsdf_tensor,
        std::vector<FloatSet3> &dI,
        std::vector<FloatSet3> &dBSDF,
        std::vector<FloatSet3> &dTabularBSDF,
        int port,
        int uidx){

        // Read a Sparse HDR Stream.
        std::stringstream mtsgradout;

        mtsgradout << "/tmp/mtsgradout-" << port << ".shds";
        imgfile = open(mtsgradout.str().c_str(), O_RDONLY);

        int r, c, n;
        int k;
        k = read(imgfile, &n, sizeof(int));
        std::cout << "[SHDS READ " << port << "] N: " << n << ", k:" << k << std::endl;
        k = read(imgfile, &r, sizeof(int));
        std::cout << "[SHDS READ " << port << "] R: " << r << ", k:" << k << std::endl;
        k = read(imgfile, &c, sizeof(int));
        std::cout << "[SHDS READ " << port << "] C: " << c << ", k:" << k << std::endl;

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

        int numNormals = map_tensor.dim_size(0);
        int numBsdfParameters = bsdf_tensor.dim_size(0);
        int numTabularBsdfParameters = 
                    tabular_bsdf_tensor.dim_size(0) *
                    tabular_bsdf_tensor.dim_size(1) *
                    tabular_bsdf_tensor.dim_size(2);

        if (map_tensor.dim_size(1) != 3)
        {
            std::cerr << "[Sparse Grad Op] ERROR: Expected 2nd dimension to be 3" << std::endl;
            exit(1);
        }

        for (int d = 0; d < n; d++)
        {
            DataPoint p = data[d];
            //std::cout << p.n << ":" << p.dx << "," << p.dy << "," << p.dz << std::endl;

            if (p.n < 0 || p.n >= numNormals + numBsdfParameters + numTabularBsdfParameters)
            {
                std::cerr << "[AttrSetGrad " << port << "] Illegal parameter index: '" << p.n << "'" << std::endl;
                exit(1);
            }

            if (std::isinf(p.dx) || std::isinf(p.dy) || std::isinf(p.dz))
            {
                std::cerr << "p INF at " << p.n << std::endl;
                exit(1);
            }

            if (p.n >= numBsdfParameters + numTabularBsdfParameters)
            {
                dI[p.n - (numBsdfParameters + numTabularBsdfParameters)].x = p.dx;
                dI[p.n - (numBsdfParameters + numTabularBsdfParameters)].y = p.dy;
                dI[p.n - (numBsdfParameters + numTabularBsdfParameters)].z = p.dz;
            }
            else if (p.n >= numBsdfParameters) 
            {
                dTabularBSDF[p.n - (numBsdfParameters)].x = p.dx;
                dTabularBSDF[p.n - (numBsdfParameters)].y = p.dy;
                dTabularBSDF[p.n - (numBsdfParameters)].z = p.dz;
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
    }

    void Disconnect()
    {

        close(sock);

        //flock(lockfd, LOCK_UN);

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

    void WriteHDSImage(const Tensor &tensor, std::string image, int idx)
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
                float f = output_tensor(i, j, idx);
                outfile.write(reinterpret_cast<const char *>(&f), sizeof(float));
            }
        }

        outfile.close();
    }

    void Compute(OpKernelContext *context) override
    {


        // Output a scalar string.

        std::vector<int> v = {256, 256};

        const Tensor &grad_output_tensor = context->input(0);

        const Tensor &params = context->input(1);
        const Tensor &bsdf = context->input(2);
        const Tensor &tabular_bsdf = context->input(3);
        const Tensor &samplewts = context->input(4);
        const Tensor &depth = context->input(5);
        const Tensor &samples = context->input(6);
        const Tensor &map_tensor = context->input(7);
        const Tensor &serverindex = context->input(8);
        const Tensor &unitindex = context->input(9);

        int numBsdfParameters = bsdf.shape().dim_size(0);
        int numTabularBsdfParameters = tabular_bsdf.shape().dim_size(0) *
                                       tabular_bsdf.shape().dim_size(1) *
                                       tabular_bsdf.shape().dim_size(2);

        Tensor *grad_params_tensor;
        Tensor *grad_depth_tensor;
        Tensor *grad_bsdf_tensor;
        Tensor *grad_tabular_bsdf_tensor;
        Tensor *grad_samplewts_tensor;
        Tensor *grad_samples_tensor;
        Tensor *grad_map_tensor;
        Tensor *grad_serverindex_tensor;
        Tensor *grad_unitindex_tensor;

        OP_REQUIRES_OK(context,
                       context->allocate_output(0, params.shape(), &grad_params_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(1, bsdf.shape(), &grad_bsdf_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(2, tabular_bsdf.shape(), &grad_tabular_bsdf_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(3, samplewts.shape(), &grad_samplewts_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(4, depth.shape(), &grad_depth_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(5, samples.shape(), &grad_samples_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(6, map_tensor.shape(), &grad_map_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(7, serverindex.shape(), &grad_serverindex_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(8, unitindex.shape(), &grad_unitindex_tensor));

        auto paramsval = params.matrix<float>();
        auto bsdfval = bsdf.vec<float>();
        auto tabularbsdfval = tabular_bsdf.tensor<float, 3>();
        auto samplewtval = samplewts.vec<float>();
        auto depthval = depth.scalar<float>();
        auto samplesval = samples.scalar<float>();
        auto serverindexval = serverindex.matrix<int>();
        auto unitindexval = unitindex.scalar<int>();

        //std::stringstream lss; 
        //lss << "/tmp/mtsgradlock-" << serverindexval(1) << ".lock";
        //lockfd = open(lss.str().c_str(), O_RDONLY);
        //std::cout << "Waiting for lock " << lss.str() << std::endl;

        assert(params.dim_size(1) == serverindex.dim_size(1));

        int numImages = params.dim_size(1);
        int numParameters = params.dim_size(0);

        // Given input.
        auto grad_output = grad_output_tensor.tensor<float, 3>();

        // Outputs that need to be calculated.
        auto grad_params = grad_params_tensor->matrix<float>();
        auto grad_bsdf = grad_bsdf_tensor->vec<float>();
        auto grad_tabular_bsdf = grad_tabular_bsdf_tensor->tensor<float, 3>();
        auto grad_samplewts = grad_samplewts_tensor->vec<float>();
        auto grad_depth = grad_depth_tensor->scalar<float>();
        auto grad_samples = grad_samples_tensor->scalar<float>();
        auto grad_serverindex = grad_serverindex_tensor->matrix<int>();
        auto grad_unitindex = grad_unitindex_tensor->scalar<float>();

        for (int img = 0; img < numImages; img++){
            for (int i = 0; i < params.dim_size(0); i++)
            {
                grad_params(i, img) = 0.f;
            }
        }

        std::vector<tfply::Triangle> tris;
        std::vector<tfply::Vertex> vtxs;
        std::vector<tfply::Normal> normals;
        std::vector<tfply::Normal> modded_normals;

        auto mapvec = map_tensor.matrix<float>();
        int numNormals = map_tensor.dim_size(0);
        for (int i = 0; i < numNormals; i++)
        {
            modded_normals.push_back(tfply::Normal(mapvec(i, 0), mapvec(i, 1), mapvec(i, 2)));
        }


        // Accumulate multiple values.
        std::stringstream plyfiless;
        plyfiless << "/tmp/mts_mesh_gradient_slot_0.ply";
        tfply::ReadPLY("/tmp/mts_srcmesh.ply", vtxs, normals, tris);
        tfply::WritePLY(plyfiless.str().c_str(), vtxs, modded_normals, tris);

        for (int img = 0; img < numImages; img++){
            //printf("Requesting grad %d\n", img);

            // Attribute list that needs to be sent to the
            // renderer.
            std::vector<float> attrs;

            attrs.push_back(img); // Push the index code to pick the right ply file.
            attrs.push_back(depthval(0));
            attrs.push_back(samplesval(0));

            for (int i = 0; i < numParameters; i++)
            {
                attrs.push_back(paramsval(i, img));
            }

            for (int i = 0; i < bsdfval.size(); i++)
            {
                attrs.push_back(bsdfval(i));
            }

            for (int i = 0; i < samplewtval.size(); i++)
            {
                attrs.push_back(samplewtval(i));
            }

            RequestGrad(img, attrs, grad_output_tensor, map_tensor, bsdf, tabular_bsdf, serverindexval(1, img), img);
        }

        // Create an array to accumulate differentials for each normal
        std::vector<FloatSet3> dI_total(numNormals, FloatSet3(0, 0, 0));
        // Create an array to accumulate differentials for each BSDF parameter
        std::vector<FloatSet3> dBSDF_total(numBsdfParameters, FloatSet3(0, 0, 0));
        // Create an array to accumulate differentials for each tabular BSDF parameter
        std::vector<FloatSet3> dTabularBSDF_total(numTabularBsdfParameters, FloatSet3(0, 0, 0));

        for(int img = 0; img < numImages; img++){
            //printf("Fetching grad %d\n", img);
            // Create an array to accumulate differentials for each normal
            std::vector<FloatSet3> dI(numNormals, FloatSet3(0, 0, 0));
            // Create an array to accumulate differentials for each BSDF parameter
            std::vector<FloatSet3> dBSDF(numBsdfParameters, FloatSet3(0, 0, 0));
            // Create an array to accumulate differentials for each Tabular BSDF parameter
            std::vector<FloatSet3> dTabularBSDF(numTabularBsdfParameters, FloatSet3(0, 0, 0));

            FetchGrad(img, map_tensor, bsdf, tabular_bsdf, dI, dBSDF, dTabularBSDF, serverindexval(1, img), img);

            for(int i = 0; i < dI.size(); i++) {
                FloatSet3& fs = dI_total[i];
                fs.x += dI[i].x;
                fs.y += dI[i].y;
                fs.z += dI[i].z;
                dI_total[i] = fs;
            }

            for(int i = 0; i < dBSDF.size(); i++) {
                FloatSet3& fs = dBSDF_total[i];
                fs.x += dBSDF[i].x;
                fs.y += dBSDF[i].y;
                fs.z += dBSDF[i].z;
                dBSDF_total[i] = fs;
            }

            for(int i = 0; i < dTabularBSDF.size(); i++) {
                FloatSet3& fs = dTabularBSDF_total[i];
                fs.x += dTabularBSDF[i].x;
                fs.y += dTabularBSDF[i].y;
                fs.z += dTabularBSDF[i].z;
                dTabularBSDF_total[i] = fs;
            }

            printf("Finished accumulating grad %d\n", img);

            std::stringstream ss;
            ss << "/tmp/mts_mesh_gradients-" << img << ".ply";
            std::vector<tfply::Normal> gradients;
            for (int i = 0; i < dI.size(); i++)
            {
                //gradients.push_back(tfply::Normal(dI_tensor(i, 0), dI_tensor(i, 1), dI_tensor(i, 2)));
                gradients.push_back(tfply::Normal(dI[i].x, dI[i].y, dI[i].z));
            }

            tfply::WritePLY(ss.str(), vtxs, gradients, tris);

        }

        auto dI_tensor = grad_map_tensor->matrix<float>();
        auto dBSDF_tensor = grad_bsdf_tensor->vec<float>();
        auto dTabularBSDF_tensor = grad_tabular_bsdf_tensor->tensor<float, 3>();

        int resolutionThetaD = tabular_bsdf.dim_size(0);
        int resolutionThetaH = tabular_bsdf.dim_size(1);
        int resolutionPhiD = tabular_bsdf.dim_size(2);

        for (int i = 0; i < dI_total.size(); i++)
        {
            // Compute maginitude and deviation
            float nx = dI_total[i].x;
            float ny = dI_total[i].y;
            float nz = dI_total[i].z;

            // Original gradients.
            dI_tensor(i, 0) = nx;
            dI_tensor(i, 1) = ny;
            dI_tensor(i, 2) = nz;
        }

        for (int i = 0; i < dBSDF_total.size(); i++)
        {
            // Compute maginitude and deviation
            float bx = dBSDF_total[i].x;
            float by = dBSDF_total[i].y;
            float bz = dBSDF_total[i].z;

            // Use only one channel since ghey all must be equal (duplicated to fit 3-channel requirements)
            dBSDF_tensor(i) = bx;
        }

        for (int i = 0; i < dTabularBSDF_total.size(); i++)
        {
            // Compute maginitude and deviation
            float bx = dTabularBSDF_total[i].x;
            float by = dTabularBSDF_total[i].y;
            float bz = dTabularBSDF_total[i].z;

            int indexPhiD = i % resolutionPhiD;
            int indexPhiDRemoved = static_cast<int>((i - indexPhiD) / resolutionPhiD);
            int indexThetaH = indexPhiDRemoved % resolutionThetaH;
            int indexThetaD = static_cast<int>((indexPhiDRemoved - indexThetaH) / resolutionThetaH);

            // Use only one channel since they all must be equal (duplicated to fit 3-channel requirements)
            dTabularBSDF_tensor(indexThetaD, indexThetaH, indexPhiD) = bx;
        }

        // Gradients for regular stuff.
        grad_depth(0) = 0.f;
        grad_samples(0) = 0.f;
        //grad_instances(0) = 0.f;
        grad_unitindex(0) = 0.f;

        for(int i = 0; i < samplewtval.size(); i++){
            grad_samplewts(i) = 0.f;
        }

        for (int img = 0; img < numImages; img++){
            grad_serverindex(0, img) = 0.f;
            grad_serverindex(1, img) = 0.f;
        }
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
