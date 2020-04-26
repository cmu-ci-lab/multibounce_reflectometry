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

#ifndef __TF_UTILS_H
#define __TF_UTILS_H
namespace tensorflow {
    void WriteParameterMap(std::string fname, const Tensor& map_tensor) {
        FILE* mapfile = fopen(fname.c_str(), "w");
        if(!mapfile) {
            std::cerr << "Failed to write normal map: " << fname.c_str() << std::endl;
            exit(1);
        }

        int dimx = map_tensor.dim_size(0);
        int dimy = map_tensor.dim_size(1);
        int dimz = map_tensor.dim_size(2);

        float *fdata = new float[dimx * dimy * dimz];
        auto tdata = map_tensor.tensor<float, 3>();
        int curr = 0;
        for(int y = 0; y < dimy; y++) {
            for(int x = 0; x < dimx; x++) {
                for(int z = 0; z < dimz; z++) {
                    fdata[curr++] = tdata(x, y, z);
                    //std::cout << "At " << x << "," << y << "," << z << ": " << tdata(x, y, z) << std::endl;
                }
            }
        }

        int k;
        k = fwrite(&dimx, sizeof(int), 1, mapfile);
        k = fwrite(&dimy, sizeof(int), 1, mapfile);
        k = fwrite(&dimz, sizeof(int), 1, mapfile);
        
        std::cerr << "Writing map file: " << dimx << "x" << dimy << "x" << dimz << std::endl;
        int bytesToWrite = dimx * dimy * dimz;
        int bytesWritten = fwrite(fdata, sizeof(float), dimx * dimy * dimz, mapfile);

        if(bytesToWrite != bytesWritten) {
            std::cerr << "ERROR: Couldn't write all teh bytes 0_0 " << bytesWritten << "/" << bytesToWrite << std::endl;
            exit(1);
        }

        fclose(mapfile);

    }

    void WriteIndexMap(std::string fname, const Tensor& map_tensor) {
        FILE* mapfile = fopen(fname.c_str(), "w");
        if(!mapfile) {
            std::cerr << "Failed to write index map: " << fname.c_str() << std::endl;
            exit(1);
        }

        int dimx = map_tensor.dim_size(0);
        int dimy = map_tensor.dim_size(1);
        int dimz = map_tensor.dim_size(2);


        float *fdata = new float[dimx * dimy * dimz];
        auto tdata = map_tensor.tensor<float, 3>();
        int curr = 0;
        for(int y = 0; y < dimy; y++) {
            for(int x = 0; x < dimx; x++) {
                for(int z = 0; z < dimz; z++) {
                    fdata[curr++] = static_cast<float>(y * dimx + x) / (dimy * dimx);
                    //std::cout << "Index At " << x << "," << y << "," << z << ": " << fdata[curr-1] << std::endl;
                }
            }
        }

        int k;
        k = fwrite(&dimx, sizeof(int), 1, mapfile);
        k = fwrite(&dimy, sizeof(int), 1, mapfile);
        k = fwrite(&dimz, sizeof(int), 1, mapfile);

        std::cerr << "Writing index file: " << dimx << "x" << dimy << "x" << dimz << std::endl;
        int bytesToWrite = dimx * dimy * dimz;
        int bytesWritten = fwrite(fdata, sizeof(float), dimx * dimy * dimz, mapfile);

        if(bytesToWrite != bytesWritten) {
            std::cerr << "ERROR: Couldn't write all teh bytes 0_0" << std::endl;
            exit(1);
        }

        fclose(mapfile);

    }

}
#endif
