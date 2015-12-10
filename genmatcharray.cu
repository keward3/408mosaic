#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void genclosestarray(float* compcompvals, float* sectcompvals, int* closestfit, int numcompimages, int numsections, int numblocks)
{
    __shared__ float distances[numcompimages];

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    int stride = 32;
    while(stride < numcompimages){ stride << 1; }
    __shared__ int location[stride];

    for(int j = 0; j * numblocks < numsections; ++j){
    int sectimageindex = bx + (j * numblocks);
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

    if(tx == 0){ closestfit[bx+j*numblocks] = location[0]; }

    } }
}

extern "C" int* genmatcharray(char* dev_image, float* dev_compcompvals, int numcompimages, int width, int height, int sectwidth, int sectheight)
{
    /* determine total number of sections */
    int numwide = width / sectwidth;
    int numtall = height / sectheight;
    if((width % sectwidth) != 0){ ++numwide; }
    if((height % sectheight != 0){ ++numtall; }
    int numsections = numwide * numtall;

    /* allocate memory for section comparison value array */
    //float* host_sectcompvals = malloc(numsections * 12 * sizeof(float));
    float* dev_sectcompvals;
    cudaMalloc((void**)&dev_sectcompvals, numsections * 12 * sizeof(float));

    /* image evaluation stuff goes here
     * information stored as four RGB floats left to right, top to bottom
     * per section into sectcompvals array */

    cudaFree(dev_image);

    int* host_closestfit = (int*)malloc(numsections * sizeof(int));
    int* dev_closestfit;
    cudaMalloc((void**)&dev_closestfit, numsections * sizeof(int));

    genclosestarray<<<4,512>>>(dev_compcompvals, dev_sectcompvals, dev_closestfit, numcompimages, numsections, 4);
    cudaDeviceSynchronize();

    cudaMemcpy(host_closestfit, dev_closestfit, numsections * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_sectcompvals);
    cudaFree(dev_closestfit);
    cudaFree(dev_compcompvals);

    return host_closestfit;
}
