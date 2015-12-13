#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL.h>
#include <SDL_image.h>
#include <time.h>
#include <string>
#include <stdio.h>

#undef main
#define COMP_WIDTH 100
#define COMP_HEIGHT 100
#define FINAL_HEIGHT 2000
#define FINAL_WIDTH 2000
#define ORIGINAL_IMAGE "original.png"

void cudasafe(int error, char* message, char* file, int line) {
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s : %i. In %s line %d\n", message, error, file, line);
		exit(-1);
	}
}

int* serial_genclosestarray(int* compvals, int* sectvals, int numcompimages, int numsections)
{
    int* closestarray = (int*)malloc(numsections * sizeof(int));
    for(int i = 0; i < numsections; i++){
        int closestval = 0;
        int closestlocation = 0;
        for(int k = 0; k < 4; k++){
            closestval += (sectvals[i*4 + k] - compvals[k]) * (sectvals[i*4 + k] - compvals[k]);
        }
        for(int j = 1; j < numcompimages; j++){
            int closestcompare = 0;
            for(int k = 0; k < 4; k++){
                closestcompare += (sectvals[i*4 + k] - compvals[j*4 + k]) * (sectvals[i*4 + k] - compvals[j*4 + k]);
            }
            if(closestcompare < closestval){
                closestval = closestcompare;
                closestlocation = j;
            }
        }
        closestarray[i] = closestlocation;
    }
    return closestarray;
}

__global__ void genclosestarray(int* compcompvals, int* sectcompvals, int* closestfit, int numcompimages, int numsections, int stride_in)
{
	extern __shared__ int distances[];
	extern __shared__ int location[];

	int a = 0;

	int tx = threadIdx.x;
	int bx = blockIdx.x;

	for (int j = 0; j * 4 < numsections; j++){
		int sectimageindex = bx + (j * 4);
		if (sectimageindex < numsections){

			
			for (int i = 0; i * 128 < numcompimages; i++)
			{
				int compimageindex = (tx / 4) + (i * 128);
				if (compimageindex < numcompimages)
				{

					int sqrtaddval = compcompvals[(tx % 4) + (4 * compimageindex)] - sectcompvals[(tx % 4) + (4 * sectimageindex)];
					int addval = sqrtaddval * sqrtaddval;
					if (addval != 0)
						a = 1;
					atomicAdd(&(distances[compimageindex]), addval);
				}
			}
			__syncthreads();

			//find smallest value in distances and put it in an int array at sectimageindex
			if (tx < numcompimages){ location[tx + numcompimages] = tx; }

			int stride = stride_in;

			for (int i = 0; stride > 0; stride = stride >> 1){
				if ((tx + stride) <= numcompimages){
					if (distances[tx + stride] < distances[tx]){
						distances[tx] = distances[tx + stride];
						location[tx + numcompimages] = location[tx + stride + numcompimages];
						//location[tx + numcompimages] = 5;
						__syncthreads();
					}
				}
			}

			/*
			for (; stride > 0; stride = stride >> 1){
			if ((tx + stride) <= numcompimages){
			if (distances[tx + stride] < distances[tx]){
			distances[tx] = distances[tx + stride];
			location[tx + numcompimages] = location[tx + stride + numcompimages] | stride;
			__syncthreads();
			}
			}
			} */

			if (tx == 0){ closestfit[sectimageindex] = location[numcompimages]; }
			//if (tx == 0){ closestfit[sectimageindex] = a; }
		}
	}
}

int main(int argc, char *args[]) {

	srand(time(NULL));
	
	int numSections = 20;
	int numCompImages = 20;

	int * sectVals = (int*)malloc(4*numSections*sizeof(int));
	int * compVals = (int*)malloc(4*numCompImages*sizeof(int));
	int* closest = (int*)malloc(numSections*sizeof(int));

	int* dev_sectVals;
	int* dev_compVals;
	int* dev_closest;
	
	int a;

	for (int i = 0; i < 4*numCompImages; i++)
	{
		compVals[i] = rand() % 10;
	}

	for (int i = 0; i < 4*numSections; i++)
	{
		sectVals[i] = rand() % 10;
	}

    int* serial_closest = serial_genclosestarray(compVals, sectVals, numCompImages, numSections);

	cudasafe(cudaMalloc((void**)&dev_sectVals, 4*numSections*sizeof(int)), "Cuda malloc", __FILE__, __LINE__);
	cudasafe(cudaMemcpy(dev_sectVals, sectVals, 4*numSections*sizeof(int), cudaMemcpyHostToDevice), "Cuda memory copy", __FILE__, __LINE__);

	cudasafe(cudaMalloc((void**)&dev_compVals, 4 * numCompImages*sizeof(int)), "Cuda malloc", __FILE__, __LINE__);
	cudasafe(cudaMemcpy(dev_compVals, compVals, 4 * numCompImages*sizeof(int), cudaMemcpyHostToDevice), "Cuda memory copy", __FILE__, __LINE__);

	cudasafe(cudaMalloc((void**)&dev_closest, numSections*sizeof(int)), "Cuda malloc", __FILE__, __LINE__);
	
	int stride;
	for (stride = 1; stride < numCompImages; stride = stride << 1) { }
    stride = stride >> 1;
	printf("stride = %d\n", stride);
	scanf("%d", &a);

	genclosestarray <<< 4, 512, sizeof(int)*(numCompImages + stride)>>>(dev_compVals, dev_sectVals, dev_closest, numCompImages, numSections, stride);
	printf("genclosestarray finished\n");
	scanf("%d", &a);

	cudasafe(cudaMemcpy(closest, dev_closest, numSections *sizeof(int), cudaMemcpyDeviceToHost), "Cuda memory copy", __FILE__, __LINE__);
	
	for (int i = 0; i < numSections; i++) 
	{ 
		printf("Section %d\n\tValue = %d\n\tClosest Index = %d\n\tClosest Value = %d\n\tSerial Closest Index = %d\n\tSerial Closest Value = %d\n\n", i, sectVals[i], closest[i], compVals[closest[i]], serial_closest[i], compVals[serial_closest[i]]);
	}
	scanf("%d", &a);
}