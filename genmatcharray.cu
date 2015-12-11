#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void genclosestarray(int* compcompvals, int* sectcompvals, int* closestfit, int numcompimages, int numsections)
{
    __shared__ int distances[numcompimages];

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    int stride = 32;
    while(stride < numcompimages){ stride << 1; }
    __shared__ int location[stride];

    for(int j = 0; j * 4 < numsections; ++j){
    int sectimageindex = bx + (j * 4);
    if(sectimageindex < numsections){

    if(tx < 504){
        for(int i = 0; i * 42 < numcompimages; ++i){
            int compimageindex = (tx / 12) + (i * 42);
            if(compimageindex < numcompimages){
                int sqrtaddval = compcompvals[(tx % 12) + (12 * compimageindex)] - sectcompvals[(tx % 12) + (12 * bx * j)];
                atomicAdd(&(distances[compimageindex]),sqrtaddval * sqrtaddval);
            }
        }
    }
    __syncthreads();

    //find smallest value in distances and put it in an int array at sectimageindex
    if(tx < stride){ location[tx] = 0; }
    for(; stride > 0; stride >> 1){
        if((tx + stride) <= numcompimages){
            if(distances[tx + stride] < distances[tx]){
                distances[tx] = distances [tx + stride];
                location[tx] = location[tx + stride] | stride;
            }
        }
    }
    __syncthreads

    if(tx == 0){ closestfit[bx+j*4] = location[0]; }

    } }
}

    /* closest match array kernel call */
    int numsections = 0; //assign a real value

    /* image evaluation stuff goes here
     * information stored as four RGB floats left to right, top to bottom
     * per section into sectcompvals array */

    cudaFree(dev_image);

    int* dev_closestfit;
    cudaMalloc((void**)&dev_closestfit, numsections * sizeof(int));
    genclosestarray<<<4,512>>>(dev_compcompvals, dev_sectcompvals, dev_closestfit, numcompimages, numsections);
    cudaDeviceSynchronize();

    cudaFree(dev_sectcompvals);
    cudaFree(dev_compcompvals);
