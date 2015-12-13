#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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

__global__ void findsmallest(int* indistances, int* closestfit, int numcompimages, int numsections, int stride_in)
{
	extern __shared__ int location[];
    extern __shared__ int distances[];

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    for (int j = 0; j * 4 < numsections; j++){
        int sectimageindex = bx + (j * 4);
        if (sectimageindex < numsections){

			//find smallest value in distances and put it in an int array at sectimageindex
			if (tx < numcompimages){
                location[tx] = tx;
                distances[numcompimages + tx] = indistances[tx];
            }

			int stride = stride_in;

			for (; stride > 0; stride = stride >> 1){
				if ((tx + stride) < numcompimages){
					if (distances[numcompimages + tx + stride] < distances[numcompimages + tx]){
						distances[numcompimages + tx] = distances[numcompimages + tx + stride];
						location[tx] = location[tx + stride];
						//location[tx] = 5;
						__syncthreads();
					}
				}
			}

			if (tx == 0){ 
                closestfit[sectimageindex] = location[0];
            }
			//if (tx == 0){ closestfit[sectimageindex] = a; }
		}
	}
}

int main(int argc, char *args[]) {

	srand(time(NULL));
	
	int numSections = 20;
	int numCompImages = 20;

	int* closest = (int*)malloc(numSections*sizeof(int));
    int* distances = (int*)malloc(numCompImages*sizeof(int));
    int* afterdistances = (int*)malloc(numCompImages*sizeof(int));

	int* dev_closest;
    int* dev_distances;
	
	int a;

	for (int i = 0; i < 4*numCompImages; i++)
	{
		distances[i] = rand() % 10;
	}

	cudasafe(cudaMalloc((void**)&dev_distances, numCompImages*sizeof(int)), "Cuda malloc", __FILE__, __LINE__);
	cudasafe(cudaMemcpy(dev_distances, distances, numCompImages*sizeof(int), cudaMemcpyHostToDevice), "Cuda memory copy", __FILE__, __LINE__);

	cudasafe(cudaMalloc((void**)&dev_closest, numSections*sizeof(int)), "Cuda malloc", __FILE__, __LINE__);
	
	int stride;
	for (stride = 1; stride < numCompImages; stride = stride << 1) { }
    stride = stride >> 1;
	printf("stride = %d\n", stride);
	scanf("%d", &a);

	findsmallest <<< 4, 512, sizeof(int)*(numCompImages)>>>(dev_distances, dev_closest, numCompImages, numSections, stride);
	printf("genclosestarray finished\n");
	scanf("%d", &a);

	cudasafe(cudaMemcpy(afterdistances, dev_distances, numCompImages *sizeof(int), cudaMemcpyDeviceToHost), "Cuda memory copy", __FILE__, __LINE__);
	cudasafe(cudaMemcpy(closest, dev_closest, numSections *sizeof(int), cudaMemcpyDeviceToHost), "Cuda memory copy", __FILE__, __LINE__);
	for (int i = 0; i < numSections; i++) 
	{ 
		printf("Section %d\n\tDistance = %d\n\tAfter Distance = %d\n\tSmallest Index = %d\n\tSmallest Value = %d\n\n", i, distances[i], afterdistances[i], closest[i], distances[closest[i]]);
	}
	scanf("%d", &a);
}
