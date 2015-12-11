#include <stdio.h>
#include <sdtlib.h>
#include <cuda_runtime.h>

    /* comporfull should be zero when computing for the component image array
     *   and one when computing for sections of the full image
     * when comporfull is zero, numimages is the number of input component images
     * when comporfull is one, numimages is the number of sections in the full image 
     * numwide is used only for full image computation - holds number of sections
     *   needed to fill one row of the full image - input zero if comporfull is zero
__global__ void histandcompval(unsigned char* imagearray, int* compvals, int comporfull, int numimages, int size, int height, int width, int numwide)
{
    __shared__ unsigned int privhistr[256];
    __shared__ unsigned int privhistg[256];
    __shared__ unsigned int privhistb[256];
    
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; //4 blocks of 256 threads

    quadrant_offset = 0;
    if(bx == 1){ quadrant_offset += width; }
    if(by == 1){ 
        if(comporfull == 0){ quadrant_offset += (size << 1); }
        else{ quadrant_offset += (numwide * size << 1); }
    }

    for(int j = 0; j < numimages; j++){
    
    privhistr[tx] = 0;
    privhistg[tx] = 0;
    privhistb[tx] = 0;
    __syncthreads();

    if(comporfull == 0){ int start = (j << 2) * size + quadrant_offset; }
    else{
        int start = ((j << 1) * (size << 2) + (j << 1) % numwide) * size + quadrant_offset;
    }
    
    for(int i = 0; i * 256 < size; i++){
        quadrant_x = (tx + i * 256) % width;
        quadrant_y = (tx + i * 256) / width;
        if(comporfull == 1){ width *= (numwide << 1); }
        if(quadrant_y < height){
            atomicAdd(&(privhistr[imagearray[start + (quadrant_x + quadrant_y * width) * 3]]), 1);
            atomicAdd(&(privhistg[imagearray[start + (quadrant_x + quadrant_y * width) * 3 + 1]), 1);
            atomicAdd(&(privhistb[imagearray[start + (quadrant_x + quadrant_y * width) * 3 + 2]), 1);
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
