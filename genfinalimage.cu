#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void genimage()
{

}

extern "C" char* genfinalimage(char** compimages, int numcompimages, int fullwidth, int fullheight, int compwidth, int compheight)
{
    int finalimagesize = fullwidth * fullheight;

    return host_finalimage;
}
