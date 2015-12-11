#include <stdio.h>
#include <sdtlib.h>
#include <cuda_runtime.h>

__global__ void histandcompval(unsigned char* compimages, int* compvals, int numcompimages, int size, int height, int width)
{
    __shared__ unsigned int privhistr[256];
    __shared__ unsigned int privhistg[256];
    __shared__ unsigned int privhistb[256];
    
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; //4 blocks of 256 threads

    quadrant_x = tx % width;
    quadrant_y = tx / width;
    quadrant_offset = 0;
    if(bx == 1){ quadrant_offset += width; }
    if(by == 1){ quadrant_offset += (size << 1); }

    for(int j = 0; j < numcompimages; j++){
    
    privhistr[tx] = 0;
    privhistg[tx] = 0;
    privhistb[tx] = 0;
    __syncthreads();

    int start = (j << 2) * size + quadrant_offset;
    
    for(int i = 0; i * 256 < size; i++){
        int index = tx + i * 256;
        if(quadrant_y < height){
            atomicAdd(&(privhistr[compimages[start + (quadrant_x + quadrant_y * width) * 3]]), 1);
            atomicAdd(&(privhistg[compimages[start + (quadrant_x + quadrant_y * width) * 3 + 1]), 1);
            atomicAdd(&(privhistb[compimages[start + (quadrant_x + quadrant_y * width) * 3 + 2]), 1);
        }
        __syncthreads();
    }

    privhistr[tx] *= tx;
    privhistg[tx] *= tx;
    privhistb[tx] *= tx;
    __syncthreads();

    for(stride = 128; stride > 0; stride >> 1){
        if(tx < stride){
            privhistr[tx] += privhistr[tx + stride];
            privhistg[tx] += privhistg[tx + stride];
            privhistb[tx] += privhistb[tx + stride];
        }
    __syncthreads();
    }

    if(tx == 0){
        compvals[3 * (bx + (by << 1) + (j << 2))] = privhistr[0];
        compvals[1 + 3 * (bx + (by << 1) + (j << 2))] = privhistg[0];
        compvals[2 + 3 * (bx + (by << 1) + (j << 2))] = privhistb[0];
    }
    __syncthreads();

    }
}

extern "C" int* evaluate(char* comparray, int numcompimages, int width, int height)
{
    int halfwidth = width >> 1;
    int halfheight = height >> 1;

    
}
