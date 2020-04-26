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

#include "ply_utils.h"

namespace tensorflow {
    
    class FloatSet3{
        public:
        float x;
        float y;
        float z;
        FloatSet3(float x, float y, float z) : x(x), y(y), z(z) {
        
        }
        FloatSet3() : x(0), y(0), z(0) {
        
        }
    };
    REGISTER_OP("MitsubaGrad")
        .Input("grad: float")
        //.Input("alpha: float")
        //.Input("weight: float")
        .Input("params: float") // Linear parameter set.

        .Input("depth: float")
        .Input("samples: float")
        .Input("parameter_map: float") // N-dimensional parameter set.
        .Input("instances: float")
        //.Output("grad_alpha: float")
        //.Output("grad_weight: float")
        .Output("grad_params: float")
        .Output("grad_depth: float")
        .Output("grad_samples: float")
        .Output("grad_parameter_map: float")
        .Output("grad_instances: float")

        //.Attr("desc: string")
        //.Attr("T: float")
        .Doc(R"doc(
BDPT gradient algorithm
)doc");

        class MitsubaGradOp : public OpKernel {
            public:
                explicit MitsubaGradOp(OpKernelConstruction* context) : OpKernel(context) {
                }

                void Connect(int port) {
                    glock.lock();
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
                    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0) 
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
                void ComputeAttrSetGrad(int i, std::vector<float> attrs, const Tensor& grad_output_tensor, const Tensor& map_tensor, std::vector<FloatSet3>& dI) {
                    
                    auto grad_output = grad_output_tensor.tensor<float, 3>();
                    //std::cout << "MTS_GRAD_OP" << std::endl;

                    short numvals = attrs.size();
                    int a = send(sock, &numvals, sizeof(short), 0);
                    //std::cerr << "[Sparse Grad Op] Sending size: " << numvals << std::endl;
                    //float attrid = attr;
                    //int b = send(sock, &attrid, sizeof(float), 0);
                    for(int i = 0; i < attrs.size(); i++) {
                        //std::cout << "datagrad: " << attrs.at(i) << std::endl;
                        float k = attrs.at(i);
                        //std::cerr << "[Sparse Grad Op] Sending: " << k << std::endl;
                        // stream them out.
                        int a = send(sock, &k, sizeof(float), 0);
                        //std::cout << "datagrad: " << k << std::endl;
                    }
                    
                    
                    //std::cout << "Done sending data" << std::endl;
                    
                    //std::cout << "Reading from FIFO pipe" << std::endl;

                    // Read a Sparse HDR Stream.
                    std::stringstream mtsgradout;

                    mtsgradout << "/tmp/mtsgradout-" << i << ".shds";
                    //std::cout << "[Sparse Grad Op] Instance" << i << ", Reading from file: " << mtsgradout.str() << std::endl;
                    imgfile = open(mtsgradout.str().c_str(),O_RDONLY);
                    
                    int r,c,n;
                    int k;
                    k = read(imgfile, &n, sizeof(int));
                    k = read(imgfile, &r, sizeof(int));
                    k = read(imgfile, &c, sizeof(int));
                    
                    //std::cerr << "[Sparse Grad Op] SHDS: " << n << " elements, " << r << "x" << c << " resolution\n";
                    if(sizeof(int) != sizeof(float)) {
                        // ERROR.
                        std::cerr << "[Sparse Grad Op] Expected size of int to be equal to size of float." << std::endl;
                        exit(1);
                    }
                    struct DataPoint{
                        int x;
                        int y;
                        int n;

                        float dx;
                        float dy;
                        float dz;
                    };

                    if( sizeof(DataPoint) != 6 * sizeof(int) ){
                        std::cerr << "[Sparse Grad Op] Expected size of DataPoint struct to be exactly 6 times the size of an int (and a float.)" << std::endl;
                        exit(1); 
                    }
                    //TODO: Free memory.
                    
                    std::cerr << "Allocating data: " << sizeof(DataPoint) * n << std::endl;
                    //DataPoint* data = (DataPoint*) malloc(sizeof(DataPoint) * n);
                    DataPoint* data = new DataPoint[n];
                    
                    if (!data) {
                        std::cerr << "[Sparse Grad Op] Out of memory" << std::endl;
                        exit(1);
                    }

                    int sofar = 0;
                    int totalbytes = sizeof(DataPoint) * n;
                    while(true){
                        int k = read(imgfile, ((char*)data) + sofar, totalbytes - sofar);
                        if(k == -1) {
                            std::cerr << "[Sparse Grad Op] ERROR while reading: " << strerror(errno) << std::endl;
                            exit(1);
                        }
                        std::cerr << "Read: " << k << " ;Sofar: " << sofar << ";Totalbytes: " << totalbytes << "; Address: "<< static_cast<void*>(((char*)data) + sofar) << std::endl;
                        if(k == 0) break;
                        sofar += k;
                        if(sofar >= totalbytes) break;
                    }

                    //auto dI = grad_map_tensor->matrix<float>();
                    int diff_size = map_tensor.dim_size(0);
                    //int diff_y_size = map_tensor.dim_size(1);

                    if(map_tensor.dim_size(1) != 3) {
                        std::cerr << "[Sparse Grad Op] ERROR: Expected 2nd dimension to be 3" << std::endl;
                        exit(1);
                    }

                    // Compute the differential by computing the sum of the products of the 
                    // dInput/dOutput with dOutput that is provided as input.
                    /*for(int x = 0; x < diff_size; x++) {
                            dI[x].x = 0;
                            dI(x, 1) = 0;
                            dI(x, 2) = 0;
                    } */

                    for(int d = 0; d < n; d++) {
                        DataPoint p = data[d];
                        // TODO: Check if x and y aren't accidentally flipped.
                        //for(int j = 0; j < c; j++) {
                        //int dy = p.n / diff_x_size;
                        //int dx = p.n % diff_x_size;

                        if(p.x < 0 || p.x >= c) {
                            std::cerr << "[AttrSetGrad] Received X coordinate '" << p.x << "': out of bounds" << std::endl;
                            exit(1);
                        }

                        if(p.y < 0 || p.y >= r) {
                            std::cerr << "[AttrSetGrad] Received Y coordinate '" << p.y << "': out of bounds" << std::endl;
                            exit(1);
                        }

                        if(p.n < 0 || p.n >= diff_size) {
                            std::cerr << "[AttrSetGrad] Illegal parameter index: '" << p.n << "'" << std::endl;
                            exit(1);
                        }

                        /*if(dy > diff_y_size) { 
                            std::cerr << "[AttrSetGrad] Received Y parameter index '" << dy << "': out of bounds" << std::endl;
                            exit(1);
                        }*/
                        
                        //if((p.x % 10 == 0) && (p.y % 10 == 0))
                        //    std::cerr << "grad_output at " << p.y << " " << p.x << " " << i << ": " << grad_output(p.y, p.x, i) <<  " D: " << p.dx << ", " << p.dy << ", " << p.dz << std::endl;
                        
                        if(std::isinf(grad_output(p.y, p.x, i))) {
                            std::cerr << "grad_output INF at " << p.y << "," << p.x << ", " << p.n << std::endl;
                            exit(1);
                        }
                        
                        if(std::isinf(p.dx) || std::isinf(p.dy) || std::isinf(p.dz)) {
                            std::cerr << "p INF at " << p.y << ", " << p.x << ", " << p.n << std::endl;
                            exit(1);
                        }

                        dI[p.n].x += grad_output(p.y, p.x, i) * p.dx;
                        dI[p.n].y += grad_output(p.y, p.x, i) * p.dy;
                        dI[p.n].z += grad_output(p.y, p.x, i) * p.dz;
                        /*if((p.n == 12345)){
			    std::cerr << "grad_output at " << p.x << ", " << p.y << ", " << i << ": " << grad_output(p.y, p.x,i);
                            std::cerr << "differential at " << p.dx << ", " << p.dy << ", " << p.dz << std::endl;
                        }*/
                    }

                    delete[] data;
                    close(imgfile);
                    //return dI;
                }

                void Disconnect() {

                    close(sock);
                    glock.unlock();
                }
                void Compute(OpKernelContext* context) override {

                    //std::cout << "Computing " << std::endl;
                    //std::cout << "Gradient render job" << std::endl;
                    
                    
                    // Output a scalar string.
                    //Tensor* output_tensor = nullptr;
                    std::vector<int> v = {256, 256};
                    
                    const Tensor& grad_output_tensor = context->input(0);
                    //const Tensor& alpha = context->input(1);
                    //const Tensor& weight = context->input(2);
                    const Tensor& params = context->input(1);
                    const Tensor& depth = context->input(2);
                    const Tensor& samples = context->input(3);
                    const Tensor& map_tensor = context->input(4);
                    const Tensor& instances = context->input(5);
                    
                    //Tensor* grad_alpha_tensor;
                    //Tensor* grad_weight_tensor;
                    Tensor* grad_params_tensor;
                    Tensor* grad_depth_tensor;
                    Tensor* grad_samples_tensor;
                    Tensor* grad_map_tensor;
                    Tensor* grad_instances_tensor;

                    //OP_REQUIRES_OK(context, 
                    //        context->allocate_output(0, alpha.shape(), &grad_alpha_tensor));
                    //OP_REQUIRES_OK(context, 
                    //        context->allocate_output(1, weight.shape(), &grad_weight_tensor));
                    OP_REQUIRES_OK(context,
                              context->allocate_output(0, params.shape(), &grad_params_tensor));
                    OP_REQUIRES_OK(context, 
                            context->allocate_output(1, depth.shape(), &grad_depth_tensor));
                    OP_REQUIRES_OK(context, 
                            context->allocate_output(2, samples.shape(), &grad_samples_tensor));
                    OP_REQUIRES_OK(context, 
                            context->allocate_output(3, map_tensor.shape(), &grad_map_tensor));
                    OP_REQUIRES_OK(context, 
                            context->allocate_output(4, instances.shape(), &grad_instances_tensor));
                    
                    //auto alphaval = alpha.scalar<float>();
                    //auto weightval = weight.scalar<float>();
                    auto paramsval = params.vec<float>();
                    auto depthval = depth.scalar<float>();
                    auto samplesval = samples.scalar<float>();
                    auto instanceCount = instances.scalar<float>();

                    // Given input.
                    auto grad_output = grad_output_tensor.tensor<float, 3>();
                    
                    // Outputs that need to be calculated.
                    //auto grad_alpha = grad_alpha_tensor->scalar<float>();
                    //auto grad_weight = grad_weight_tensor->scalar<float>();
                    auto grad_params = grad_params_tensor->vec<float>();
                    auto grad_depth = grad_depth_tensor->scalar<float>();
                    auto grad_samples = grad_samples_tensor->scalar<float>();
                    auto grad_instances = grad_instances_tensor->scalar<float>();
                    
                    // Attribute list that needs to be sent to the 
                    // renderer.
                    std::vector<float> attrs;
                    //attrs.push_back(alphaval(0));
                    //attrs.push_back(weightval(0));
                    //attrs.push_back(1.0f - weightval(0));
                    //for(int i = 0; i < paramsval.size(); i++) {
                    //    attrs.push_back(paramsval(i));
                    //}
                    attrs.push_back(depthval(0));
                    attrs.push_back(samplesval(0));
                    for(int i = 0; i < paramsval.size(); i++) {
                        attrs.push_back(paramsval(i));
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
                    
                    
                    for(int i = 0; i < paramsval.size(); i++) {
                        grad_params(i) = 0.f;
                    }
                    
                    
                    std::vector<tfply::Triangle> tris;
                    std::vector<tfply::Vertex> vtxs;
                    std::vector<tfply::Normal> normals;
                    std::vector<tfply::Normal> modded_normals;

                    auto mapvec = map_tensor.matrix<float>();
                    int num_normals = map_tensor.dim_size(0);
                    for(int i = 0; i < num_normals; i++) {
                        modded_normals.push_back(tfply::Normal(mapvec(i, 0), mapvec(i, 1), mapvec(i, 2)));
                    }

                    //map_tensor.dim_size(0);
                    //map_tensor.dim_size(1);
                    // Write the PLY file with modified normals.
                    tfply::ReadPLY("/tmp/mts_srcmesh.ply", vtxs, normals, tris);
                    tfply::WritePLY("/tmp/mts_mesh.ply", vtxs, modded_normals, tris);
                    //WriteParameterMap("/tmp/mts_nmap.hds", map_tensor);
                    // Write the parameter map out to HDS files and also 
                    // write out the reference HDS files.
                    //WriteParameterMap("/tmp/mts_nmap.hds", map_tensor);
                    //WriteIndexMap("/tmp/mts_idx.hds", map_tensor);
                    
                    // Create an array to accumulate differentials for each normal
                    std::vector<FloatSet3> dI(num_normals);
                    for(int i = 0; i < num_normals; i++) {
                        dI[i] = FloatSet3(0, 0, 0); 
                    }
                    for(int i = 0; i < instanceCount(0); i++) {
                        // Accumulate multiple values.
                        Connect(7555);
                        ComputeAttrSetGrad(0, attrs, grad_output_tensor, map_tensor, dI);
                        Disconnect();
                    }

                    auto dI_tensor = grad_map_tensor->matrix<float>();
                    //std::cerr << "Tensor index count: " << num_normals << std::endl;
                    //std::cerr << "Instance count: " << instanceCount(0) << std::endl;
                    for(int i = 0; i < dI.size(); i++ ){
                         dI_tensor(i,0) = dI[i].x;
                         dI_tensor(i,1) = dI[i].y;
                         dI_tensor(i,2) = dI[i].z;
                    }
		    // std::cerr << grad_output_tensor 
		    int dbgnormal = 12345;
		    std::cerr << dI_tensor(dbgnormal, 0) << "," << dI_tensor(dbgnormal,1) << "," << dI_tensor(dbgnormal,2) << std::endl;
                    /*for(int z = 0; z < 3; z++) { 
                        for(int x = 0; x < dI.size(); x++) {
                            //for(int y = 0; y < diff_y_size; y++) {
                                std::cerr << dI_tensor(x, z);
                            //}
                            std::cerr << std::endl;
                        }
                        std::cerr << std::endl << std::endl;
                    } */

                    grad_depth(0) = 0.f;
                    grad_samples(0) = 0.f;
                    grad_instances(0) = 0.f;
                }
                void Check(int k, size_t sz) {
                    if(k != sz) {
                        std::cout<<"Didn't write the correct bytes. Actual:" << sz << " Written:" << k << std::endl;
                        exit(1);
                    }
                }
                
                static std::mutex glock;
            private:
                int sock;
                int imgfile;
        };
    std::mutex MitsubaGradOp::glock;   
    REGISTER_KERNEL_BUILDER(Name("MitsubaGrad").Device(DEVICE_CPU), MitsubaGradOp);

}  // namespace tensorflow
