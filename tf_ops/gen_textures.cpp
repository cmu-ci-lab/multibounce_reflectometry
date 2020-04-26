#include <iostream>
#include <fstream>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

#define WIDTH 4
#define HEIGHT 4

// Generate a specific texture and it's index grid for testing.
int main(int argc, char** argv) {
    
    //int nfile = open("/tmp/mts_nmap.hds", O_CREAT | O_WRONLY);
    FILE* nfile = fopen("/tmp/mts_nmap.hds", "w");
    if(nfile == NULL) {
        std::cout << "Could not open file /tmp/mts_nmap.hds" << std::endl;
    }
    std::cout << "Opened file /tmp/mts_nmap.hds" << std::endl;
    //int ifile = open("/tmp/mts_index.hds", O_CREAT | O_WRONLY);
    FILE* ifile = fopen("/tmp/mts_idx.hds", "w");
    if(ifile == NULL) {
        std::cout << "Could not open file /tmp/mts_idx.hds" << std::endl;
    }
    std::cout << "Opened file /tmp/mts_idx.hds" << std::endl;

    int w = WIDTH;
    int h = HEIGHT;
    int c = 3;
    int ci = 3;
    fwrite(&w, sizeof(int), 1, nfile);
    fwrite(&h, sizeof(int), 1, nfile);
    fwrite(&c, sizeof(int), 1, nfile);
    
    fwrite(&w, sizeof(int), 1, ifile);
    fwrite(&h, sizeof(int), 1, ifile);
    //c = 1;
    fwrite(&ci, sizeof(int), 1, ifile);


    float data[WIDTH * HEIGHT * 3];
    for(int x = 0; x < WIDTH; x++) {
        for(int y = 0; y < HEIGHT; y++) {
            if((x+y)%2){
            //if(){
                data[y * WIDTH * c + x * c + 0] = 0.5f;
                data[y * WIDTH * c + x * c + 1] = 0.5f;
                data[y * WIDTH * c + x * c + 2] = 1.0f;

            } else { 
                //data[y * WIDTH * c + x * c + 0] = 0.789f;
                //data[y * WIDTH * c + x * c + 1] = 0.211f;
                //data[y * WIDTH * c + x * c + 2] = 0.789f;
                data[y * WIDTH * c + x * c + 0] = 0.7962f;
                data[y * WIDTH * c + x * c + 1] = 0.223f;
                data[y * WIDTH * c + x * c + 2] = 0.7934f;
                
            }
        }
    }
    fwrite(data, sizeof(float), WIDTH * HEIGHT * 3, nfile);
     

    for(int x = 0; x < WIDTH; x++) {
        for(int y = 0; y < HEIGHT; y++) {
            data[y * WIDTH * ci + x * ci] = static_cast<float>(y * WIDTH + x) / (WIDTH * HEIGHT);
            data[y * WIDTH * ci + x * ci + 1] = static_cast<float>(y * WIDTH + x) / (WIDTH * HEIGHT);
            data[y * WIDTH * ci + x * ci + 2] = static_cast<float>(y * WIDTH + x) / (WIDTH * HEIGHT);
        }
    }
    fwrite(data, sizeof(float), WIDTH * HEIGHT * ci, ifile);
    
    fclose(nfile);
    fclose(ifile);
    //float data[WIDTH * HEIGHT * 3];
    
    return 0;
}
