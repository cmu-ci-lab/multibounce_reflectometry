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

    REGISTER_OP("Mitsuba")
        .Input("params: float")
        //.Input("weight: float")
        .Input("depth: float")
        .Input("samples: float")
        .Output("image: float") 
        .Doc(R"doc(
BDPT algorithm renders a scene file with given parameters.
)doc");

        class MitsubaOp : public OpKernel {
            public:
                explicit MitsubaOp(OpKernelConstruction* context) : OpKernel(context) {

                }

                // Opens a connection to a mitsuba tensorflow server on a given port no.
                void Connect(int port) {
                    
                    struct sockaddr_in address;
                    int valread;
                    struct sockaddr_in serv_addr;
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
                
                // Close connection.
                void Disconnect(){
                    close(sock);
                }

                void Compute(OpKernelContext* context) override {

                    Connect(7554);

                    // Output a scalar string.
                    Tensor* output_tensor = nullptr;
                    std::vector<int> v = {256, 256};
                    
                    //const Tensor& alpha = context->input(0);
                    //const Tensor& weight = context->input(1);
                    const Tensor& params = context->input(0);
                    const Tensor& depth = context->input(1);
                    const Tensor& samples = context->input(2);
                    

                    // TODO: WARN: Currently the shape is fixed at 256x256.
                    // Implementation will break if the mitsuba scene files render at
                    // any other resolution.
                    TensorShape shape;
                    shape.AddDim(256);
                    shape.AddDim(256);
                    OP_REQUIRES_OK(context,
                            context->allocate_output(0, shape, &output_tensor));

                    auto output = output_tensor->matrix<float>();
                    //auto alphaval = alpha.scalar<float>();
                    //auto wval = weight.scalar<float>();
                    auto paramvec = params.vec<float>();
                    auto dval = depth.scalar<float>();
                    auto sval = samples.scalar<float>();

                    // Make a list of parameter values to send.
                    // Note, order is important.
                    std::vector<float> rdata;
                    //rdata.push_back(alphaval(0));   // alpha
                    //rdata.push_back(wval(0));       // weight1
                    //rdata.push_back(1-wval(0));     // weight2

                    // Put all the new parameters into the list.
                    //std::cout << "PARAMS: " << paramvec.size() << std::endl;
                    for(int i = 0; i < paramvec.size(); i++){
                        rdata.push_back(paramvec(i));
                    }
                    
                    rdata.push_back(dval(0));       // depth
                    rdata.push_back(sval(0));       // sampleCount

                    //std::cout << "MTS_OP" << std::endl;
                    short t = rdata.size();
                    // Send parameters to renderer.
                    int a = send(sock, &t, sizeof(short), 0);
                    for(int i = 0; i < rdata.size(); i++) {
                        float k = rdata.at(i);
                        //std::cout << "Sending: " << k << std::endl;   
                        int a = send(sock, &k, sizeof(float), 0);  
                    }

                    // Read the image from a FIFO pipe.
                    
                    int imgfile = open("/tmp/mtsout.hds",O_RDONLY);
                    
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
                        
                        if(k == 0) break;
                        sofar += k;
                        if(sofar == totalbytes) break;
                    }
                    

                    // TODO: Temporary. 
                    if(r != c) {
                        std::cout << "[Mitsuba Tensorflow Operator] WARNING: Currently no support for R != C image dimensions" << std::endl;
                    }
                    
                    // Set output values to data obtained.
                    // TODO: Check if the values are row major or column major.
                    for(int i = 0; i < r; i++) {
                        for(int j = 0; j < c; j++) {
                            output(i, j) = data[i * c + j];
                        }
                    }
                    
                    close(imgfile);
                    Disconnect();
                }
            private:
                int sock;
                
        };

    REGISTER_KERNEL_BUILDER(Name("Mitsuba").Device(DEVICE_CPU), MitsubaOp);

}  // namespace tensorflow
