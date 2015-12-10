#include <stdio.h>
#include <sdtlib.h>
#include <cuda_runtime.h>

__global__ histandcompval(unsigned char* A, int size)
{
    __shared__ unsigned int privhist[256];
    
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    if(tx < 256){ privhist[tx] = 0; }
    __syncthreads();
    
    for(int i = 0; i * BLOCK_SIZE < size; ++i){
        int index = tx + i * BLOCK_SIZE;
        if(index < size){ atomicAdd(&(privhist[A[index]]), 1); }
        __syncthreads();
    }

    if(tx < 256){
        
    }
}

extern "C" float* evaluate()
