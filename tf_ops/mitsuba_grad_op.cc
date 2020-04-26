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

namespace tensorflow {

    REGISTER_OP("MitsubaGrad")
        .Input("grad: float")
        //.Input("alpha: float")
        //.Input("weight: float")
        .Input("params: float")
        .Input("depth: float")
        .Input("samples: float")
        //.Output("grad_alpha: float")
        //.Output("grad_weight: float")
        .Output("grad_params: float")
        .Output("grad_depth: float")
        .Output("grad_samples: float")

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
                    
                    struct sockaddr_in address;
                    //int sock = 0, valread;
                    int valread;
                    struct sockaddr_in serv_addr;
                    //char *hello = "Hello from client";
                    //char buffer[1024] = {0};
                    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
                    {
                        printf("\n Socket creation error \n");
                        //return -1;
                        exit(-1);
                    }

                    memset(&serv_addr, '0', sizeof(serv_addr));

                    serv_addr.sin_family = AF_INET;
                    serv_addr.sin_port = htons(port);

                    // Convert IPv4 and IPv6 addresses from text to binary form
                    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0) 
                    {
                        printf("\nInvalid address/ Address not supported \n");
                        //return -1;
                        exit(-1);
                    }

                    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
                    {
                        printf("\nConnection Failed \n");
                        exit(-1);
                    }

                    //std::cout << "Finished connecting." << std::endl;
                }
                float ComputeAttrGrad(int attr, std::vector<float> attrs, const Tensor& grad_output_tensor) {
                    
                    auto grad_output = grad_output_tensor.matrix<float>();
                    //std::cout << "MTS_GRAD_OP" << std::endl;

                    short numvals = attrs.size() + 1;
                    int a = send(sock, &numvals, sizeof(short), 0);
                    //std::cout << "Sending: " << numvals << std::endl;
                    float attrid = attr;
                    int b = send(sock, &attrid, sizeof(float), 0);
                    for(int i = 0; i < attrs.size(); i++) {
                        //std::cout << "datagrad: " << attrs.at(i) << std::endl;
                        float k = attrs.at(i);
                        //std::cout << "Sending: " << k << std::endl;
                        // stream them out.
                        int a = send(sock, &k, sizeof(float), 0);
                        //std::cout << "datagrad: " << k << std::endl;
                    }
                    
                    
                    //std::cout << "Done sending data" << std::endl;
                    
                    //std::cout << "Reading from FIFO pipe" << std::endl;

                    imgfile = open("/tmp/mtsgradout.hds",O_RDONLY);
                    
                    int r,c;
                    read(imgfile, &r, sizeof(int));
                    read(imgfile, &c, sizeof(int));

                     
                    //TODO: Free memory.
                    float *data = (float*) malloc(sizeof(float) * r * c);
                    
                    int sofar = 0;
                    int totalbytes = sizeof(float) * r * c;
                    while(true){
                        int k = read(imgfile, ((char*)data) + sofar, totalbytes - sofar);
                        if(k == -1) {
                            std::cout << "ERROR while reading: " << strerror(errno) << std::endl ;
                            exit(1);
                        }
                        //std::cout << "Read: " << k << std::endl;
                        if(k == 0) break;
                        sofar += k;
                        if(sofar >= totalbytes) break;
                    }

                    float dI = 0.f;
                    // Compute the differential by computing the sum of the products of the 
                    // dInput/dOutput with dOutput that is provided as input.
                    for(int i = 0; i < r; i++) {
                        for(int j = 0; j < c; j++) {
                            dI += grad_output(i, j) * data[i * 256 + j];
                        }
                    }
                    
                    close(imgfile);
                    return dI;
                }

                void Disconnect() {
                    close(sock);
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
                    
                    //Tensor* grad_alpha_tensor;
                    //Tensor* grad_weight_tensor;
                    Tensor* grad_params_tensor;
                    Tensor* grad_depth_tensor;
                    Tensor* grad_samples_tensor;


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
                    
                    //auto alphaval = alpha.scalar<float>();
                    //auto weightval = weight.scalar<float>();
                    auto paramsval = params.vec<float>();
                    auto depthval = depth.scalar<float>();
                    auto samplesval = samples.scalar<float>();

                    // Given input.
                    auto grad_output = grad_output_tensor.matrix<float>();
                    
                    // Outputs that need to be calculated.
                    //auto grad_alpha = grad_alpha_tensor->scalar<float>();
                    //auto grad_weight = grad_weight_tensor->scalar<float>();
                    auto grad_params = grad_params_tensor->vec<float>();
                    auto grad_depth = grad_depth_tensor->scalar<float>();
                    auto grad_samples = grad_samples_tensor->scalar<float>();
                    
                    // Attribute list that needs to be sent to the 
                    // renderer.
                    std::vector<float> attrs;
                    //attrs.push_back(alphaval(0));
                    //attrs.push_back(weightval(0));
                    //attrs.push_back(1.0f - weightval(0));
                    for(int i = 0; i < paramsval.size(); i++) {
                        attrs.push_back(paramsval(i));
                    }
                    attrs.push_back(depthval(0));
                    attrs.push_back(samplesval(0));
                    
                    for(int i = 0; i < paramsval.size(); i++) {
                        Connect(7555);
                        grad_params(i) = ComputeAttrGrad(i, attrs, grad_output_tensor);
                        Disconnect();
                    }

                    /*Connect(7555);
                    grad_alpha(0) = ComputeAttrGrad(1, attrs, grad_output_tensor);
                    Disconnect();

                    Connect(7555);
                    grad_weight(0) = ComputeAttrGrad(0, attrs, grad_output_tensor);
                    Disconnect();*/

                    grad_depth(0) = 0.f;
                    grad_samples(0) = 0.f;
                }
            private:
                int sock;
                int imgfile;
                
        };
        
    REGISTER_KERNEL_BUILDER(Name("MitsubaGrad").Device(DEVICE_CPU), MitsubaGradOp);

}  // namespace tensorflow
