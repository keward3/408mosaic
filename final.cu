#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL.h>
#include <SDL_image.h>
#include <dirent.h>
#include <string>
#include <stdio.h>

//FINAL_HEIGHT, FINAL_WIDTH must be evenly divisible by COMP_WIDTH, COMP_HEIGHT

#undef main
#define COMP_WIDTH 200
#define COMP_HEIGHT 200
#define FINAL_HEIGHT 2000
#define FINAL_WIDTH 2000
#define ORIGINAL_IMAGE "original.png"

using namespace std;




Uint8* serialBuildFinalImg(int* closestFit, Uint8* components, int sectionsPerRow, int pixelsPerSection)
{
	Uint8* finalImage = (Uint8*)malloc(FINAL_HEIGHT * FINAL_WIDTH * 4 * sizeof(Uint8));
	for (int i = 0; i < FINAL_HEIGHT * FINAL_WIDTH; i++){
		int xIndex = i % FINAL_WIDTH;
		int yIndex = i / FINAL_WIDTH;

		int sectionIndex = (xIndex / COMP_WIDTH) + (yIndex / COMP_HEIGHT) * sectionsPerRow;
		int closest = closestFit[sectionIndex];

		int sectionIndexX = xIndex % COMP_WIDTH;
		int sectionIndexY = yIndex % COMP_HEIGHT;

		finalImage[i * 4 + 0] = components[3 * (closest * pixelsPerSection + sectionIndexY * COMP_WIDTH + sectionIndexX) + 2];
		finalImage[i * 4 + 1] = components[3 * (closest * pixelsPerSection + sectionIndexY * COMP_WIDTH + sectionIndexX) + 1];
		finalImage[i * 4 + 2] = components[3 * (closest * pixelsPerSection + sectionIndexY * COMP_WIDTH + sectionIndexX)];
		finalImage[i * 4 + 3] = 0xff;
	}
	return finalImage;
}

int* genclosestarray(int* compvals, int* sectvals, int numcompimages, int numsections)
{
	int* closestarray = (int*)malloc(numsections * sizeof(int));
	for (int i = 0; i < numsections; i++){
		int closestval = 0;
		int closestlocation = 0;
		for (int k = 0; k < 12; k++){
			closestval += (sectvals[i * 12 + k] - compvals[k]) * (sectvals[i * 12 + k] - compvals[k]);
		}
		for (int j = 1; j < numcompimages; j++){
			int closestcompare = 0;
			for (int k = 0; k < 12; k++){
				closestcompare += (sectvals[i * 12 + k] - compvals[j * 12 + k]) * (sectvals[i * 12 + k] - compvals[j * 12 + k]);
			}
			if (closestcompare < closestval){
				closestval = closestcompare;
				closestlocation = j;
			}
		}
		closestarray[i] = closestlocation;
	}
	return closestarray;
}


void cudasafe(int error, char* message, char* file, int line) {
	if (error != cudaSuccess) {
		if (error == cudaErrorInvalidDevicePointer)
		{
			fprintf(stderr, "CUDA Invalid Device Pointer Error: %s : %i. In %s line %d\n", message, error, file, line);
			printf("CUDA Error: %s : %i. In %s line %d\n", message, error, file, line);
			int a;
			scanf("%d", &a);
			exit(-1);
		}
		else if (error == cudaErrorInitializationError)
		{
			fprintf(stderr, "CUDA Initialization Error: %s : %i. In %s line %d\n", message, error, file, line);
			printf("CUDA Error: %s : %i. In %s line %d\n", message, error, file, line);
			int a;
			scanf("%d", &a);
			exit(-1);
		}
		else
		{
			fprintf(stderr, "CUDA Error: %s : %i. In %s line %d\n", message, error, file, line);
			printf("CUDA Error: %s : %i. In %s line %d\n", message, error, file, line);
			int a;
			scanf("%d", &a);
			exit(-1);
		}
	}
}

__global__ void BuildFinalImg(int* closestFit, Uint8* components, Uint8* finalImage, int sectionsPerRow, int pixelsPerSection)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int xIndex = tx + bx*blockDim.x;
	int yIndex = ty + by*blockDim.y;

	int xthreadsingrid = gridDim.x * blockDim.x;
	int ythreadsingrid = gridDim.y * blockDim.y;

	int xStrides = FINAL_WIDTH / xthreadsingrid;
	if (FINAL_WIDTH % xthreadsingrid != 0){ xStrides++; }
	int yStrides = FINAL_HEIGHT / ythreadsingrid;
	if (FINAL_HEIGHT % ythreadsingrid != 0){ yStrides++; }

	for (int i = 0; i < xStrides; i++)
	{
		for (int j = 0; j < yStrides; j++)
		{
			if (xIndex < FINAL_WIDTH && yIndex < FINAL_HEIGHT)
			{
				int sectionX = xIndex / COMP_WIDTH;
				int sectionY = yIndex / COMP_HEIGHT;

				int sectionIndex = sectionY*sectionsPerRow + sectionX;
				int closest = closestFit[sectionIndex];

				Uint8* start = components + closest*pixelsPerSection*3;

				int sectionIndexX = xIndex % COMP_WIDTH;
				int sectionIndexY = yIndex % COMP_HEIGHT;

				Uint8 red = start[(sectionIndexX + sectionIndexY*COMP_WIDTH) * 3];
				Uint8 blue = start[(sectionIndexX + sectionIndexY*COMP_WIDTH) * 3 + 1];
				Uint8 green = start[(sectionIndexX + sectionIndexY*COMP_WIDTH) * 3 + 2];

				//*(Uint32*)finalImage[(xIndex + yIndex*FINAL_WIDTH) * 4] = 0xff000000 | ((((int)red) << 16)) | ((((int)green) << 8)) | ((int)blue);

				//*(Uint32*)finalImage[(xIndex + yIndex*FINAL_WIDTH) * 4] = 0xffffffff;
				finalImage[xIndex + yIndex * FINAL_WIDTH * 4] = green;
				finalImage[xIndex + yIndex * FINAL_WIDTH * 4 + 1] = blue;
				finalImage[xIndex + yIndex * FINAL_WIDTH * 4 + 2] = red;
				finalImage[xIndex + yIndex * FINAL_WIDTH * 4 + 3] = 0xff;
			}
			yIndex += ythreadsingrid;
		}
		yIndex = ty + by*blockDim.y;
		xIndex += xthreadsingrid;
	}
}

__global__ void genclosestarray(int* compcompvals, int* sectcompvals, int* closestfit, int numcompimages, int numsections, int stride)
{
	extern __shared__ int distances[];
	extern __shared__ int location[];

	int tx = threadIdx.x;
	int bx = blockIdx.x;

	for (int j = 0; j * 4 < numsections; j++){
		int sectimageindex = bx + (j * 4);
		if (sectimageindex < numsections){

			if (tx < 504)
			{
				for (int i = 0; i * 42 < numcompimages; i++)
				{
					int compimageindex = (tx / 12) + (i * 42);
					if (compimageindex < numcompimages)
					{

						int sqrtaddval = compcompvals[(tx % 12) + (12 * compimageindex)] - sectcompvals[(tx % 12) + (12 * bx * j)];
						int addval = sqrtaddval * sqrtaddval;
						atomicAdd(&(distances[compimageindex]), addval);
					}
				}
			}
			__syncthreads();

			//find smallest value in distances and put it in an int array at sectimageindex
			if (tx < numcompimages){ location[tx+numcompimages] = tx; }

			for (; stride > 0; stride = stride >> 1){
				if ((tx + stride) <= numcompimages){
					if (distances[tx + stride] < distances[tx]){
						distances[tx] = distances[tx + stride];
						location[tx + numcompimages] = location[tx + stride + numcompimages];
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

			if (tx == 0){ closestfit[bx + j*4] = location[numcompimages]; }

		}
	}
}


__global__ void cudaTransform(Uint8 *resized, Uint8 *input, Uint16 pitchInput, Uint8 bytesPerPixelInput, Uint8 bytesPerPixelOutput, int componentIndex, int pixelsPerSection, float xRatio, float yRatio){
	int x = (int)(xRatio * blockIdx.x);
	int y = (int)(yRatio * blockIdx.y);

	Uint8 *point_1;
	Uint8 *point_2;
	Uint8 *point_3;
	Uint8 *point_4;
	float xDist, yDist, blue, red, green;

	xDist = (xRatio * blockIdx.x) - x;
	yDist = (yRatio * blockIdx.y) - y;

	point_1 = input + y * pitchInput + x * bytesPerPixelInput;
	point_2 = input + y * pitchInput + (x + 1) * bytesPerPixelInput;
	point_3 = input + (y + 1) * pitchInput + x * bytesPerPixelInput;
	point_4 = input + (y + 1) * pitchInput + (x + 1) * bytesPerPixelInput;

	blue = (point_1[2])*(1 - xDist)*(1 - yDist) + (point_2[2])*(xDist)*(1 - yDist) + (point_3[2])*(yDist)*(1 - xDist) + (point_4[2])*(xDist * yDist);

	green = ((point_1[1]))*(1 - xDist)*(1 - yDist) + (point_2[1])*(xDist)*(1 - yDist) + (point_3[1])*(yDist)*(1 - xDist) + (point_4[1])*(xDist * yDist);

	red = (point_1[0])*(1 - xDist)*(1 - yDist) + (point_2[0])*(xDist)*(1 - yDist) + (point_3[0])*(yDist)*(1 - xDist) + (point_4[0])*(xDist * yDist);

	Uint8 *p = resized + pixelsPerSection*componentIndex*bytesPerPixelOutput+ blockIdx.y * COMP_WIDTH * bytesPerPixelOutput + blockIdx.x * bytesPerPixelOutput;
	p[0] = (Uint8)red;
	p[1] = (Uint8)green;
	p[2] = (Uint8)blue;
}


/* comporfull should be zero when computing for the component image array
*   and one when computing for sections of the full image
* when comporfull is zero, numimages is the number of input component images
* when comporfull is one, numimages is the number of sections in the full image
* numwide is used only for full image computation - holds number of sections
*   needed to fill one row of the full image - input zero if comporfull is zero
*/
__global__ void histandcompval(unsigned char* imagearray, float* compvals, int comporfull, int numimages, int size, int height, int width, int numwide)
{
	__shared__ unsigned int privhist[64];
    if(tx < 64){ privhist[tx] = 0; }
    __syncthreads();

	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int by = blockIdx.y; //4 blocks of 256 threads

	int quadrant_offset = 0;
	if (bx == 1){ quadrant_offset += width; }
	if (by == 1){
		if (comporfull == 0){ quadrant_offset += (size << 1); }
		else{ quadrant_offset += (numwide * size << 1); }
	}

	for (int j = 0; j < numimages; j++){
		if (comporfull == 0){ int start = j * (size << 2) + quadrant_offset; }
		else{
			int start = (j / numwide) * (size << 2) * numwide + (j % numwide) * (width << 1) + quadrant_offset;
		}

		for (int i = 0; i * 256 < size; i++){
			int quadrant_x = (tx + i * 256) % width;
			int quadrant_y = (tx + i * 256) / width;
			if (comporfull == 1){ width *= (numwide << 1); }
			if (quadrant_y < height){
                float grayval = imagearray[start + (quadrant_x + quadrant_y * width) * 3] * 0.07;
                grayval += imagearray[start + (quadrant_x + quadrant_y * width) * 3 + 1] * 0.71;
				grayval += imagearray[start + (quadrant_x + quadrant_y * width) * 3 + 2] * 0.21;
				atomicAdd(&(privhist[((int)grayval >> 2)];
			}
		}

		privhist[tx] *= tx;
		__syncthreads();

		for (int stride = 32; stride > 0; stride = stride >> 1){
			if (tx < stride){
				privhist[tx] += privhist[tx + stride];
			}
			__syncthreads();
		}

		if (tx == 0){
			compvals[bx + (by << 1) + (j << 2)] = privhist[0];
		}
		__syncthreads();
	}
}


    

int main(int argc, char *args[]) {

	SDL_Init(SDL_INIT_EVERYTHING);

	DIR *dir;

	struct dirent *picture;

	Uint32 amask = 0xff000000;
	Uint32 rmask = 0x00ff0000;
	Uint32 gmask = 0x0000ff00;
	Uint32 bmask = 0x000000ff;

	float b = 32.41;
	printf("size of int: %d\n", sizeof(int));
	printf("b: %f\n", b);
	printf("Uint8(b): %d\n", Uint8(b));
	printf("int(b): %d\n", int(b));

	int numcompimages = -2;

	if ((dir = opendir("./Images")) != NULL)
	{
		while ((picture = readdir(dir)) != NULL)
		{
			numcompimages++;
		}
	}
	else
	{
		int a;
		printf("COULD NOT OPEN DIRECTORY");
		scanf("%d", &a);
	}
	
	closedir(dir);

	Uint8 *dev_compimagearray;
	int resizedCompSize = sizeof(Uint8)*numcompimages * 3 * COMP_HEIGHT*COMP_WIDTH;
	cudasafe(cudaMalloc((void **)&dev_compimagearray, resizedCompSize), "New image allocation ", __FILE__, __LINE__);


	if ((dir = opendir("./Images")) != NULL)
	{
		int compIdx = 0;
		while ((picture = readdir(dir)) != NULL)
		{

			char * name = picture->d_name;

			if (string(name).compare(".") != 0 && string(name).compare("..")!=0)
			{
				string loadname = "./Images/" + string(name);
				SDL_Surface *image = IMG_Load(loadname.c_str());
				int imageByteLength = image->w * image->h * sizeof(Uint8)*image->format->BytesPerPixel;

				if (!image){
				printf("IMG_Load: %s\n", IMG_GetError());
				return 1;
				}

				dim3 grid(COMP_HEIGHT, COMP_HEIGHT);

				float xRatio = ((float)(image->w - 1)) / COMP_WIDTH;
				float yRatio = ((float)(image->h - 1)) / COMP_HEIGHT;

				// Create pointer to device and host pixels
				Uint8 *pixels = (Uint8*)image->pixels;
				Uint8 *pixels_dyn;

				cudaEvent_t start, stop;
				float time;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				// Copy original image
				cudasafe(cudaMalloc((void **)&pixels_dyn, imageByteLength), "Original image allocation ", __FILE__, __LINE__);
				cudasafe(cudaMemcpy(pixels_dyn, pixels, imageByteLength, cudaMemcpyHostToDevice), "Copy original image to device ", __FILE__, __LINE__);

				// Start measuring time
				cudaEventRecord(start, 0);

				// Do the bilinear transform on CUDA device
				cudaTransform <<< grid, 1 >>>(dev_compimagearray, pixels_dyn, image->pitch, image->format->BytesPerPixel, 3, compIdx, COMP_WIDTH*COMP_HEIGHT, xRatio, yRatio);

				// Stop the timer
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);

				// Free memory
				cudaFree(pixels_dyn);

				cudaEventElapsedTime(&time, start, stop);
				printf("Time for the kernel: %f ms\n", time);
				

				// Free surfaces
				SDL_FreeSurface(image);

				compIdx++;
			}

		}

		printf("Got past image load\n");
		int a;
		scanf("%d", &a);

		closedir(dir);

		SDL_Surface *original = IMG_Load(ORIGINAL_IMAGE);

		Uint8* dev_origimage;
		Uint8* dev_fullimage;
		int imageByteLength = original->w * original->h * sizeof(Uint8) * 4;
		cudasafe(cudaMalloc((void **)&dev_origimage, imageByteLength), "Original image allocation ", __FILE__, __LINE__);
		cudasafe(cudaMemcpy(dev_origimage, original->pixels, imageByteLength, cudaMemcpyHostToDevice), "Copy original image to device ", __FILE__, __LINE__);

		cudasafe(cudaMalloc((void **)&dev_fullimage, 3*FINAL_HEIGHT*FINAL_WIDTH*sizeof(Uint8)), "Original image allocation ", __FILE__, __LINE__);

		float orig_xRatio = ((float)(original->w - 1)) / FINAL_WIDTH;
		float orig_yRatio = ((float)(original->h - 1)) / FINAL_HEIGHT;

		dim3 gridOrig(FINAL_WIDTH, FINAL_HEIGHT);

		cudaTransform <<< gridOrig, 1 >>>(dev_fullimage, dev_origimage, original->pitch, original->format->BytesPerPixel, 3, 0, FINAL_WIDTH*FINAL_HEIGHT, orig_xRatio, orig_yRatio);

		int numsections = ((FINAL_WIDTH / COMP_WIDTH) * (FINAL_HEIGHT / COMP_HEIGHT));

		dim3 grid(2, 2);

		/* kernel call for component image evaluation */
		int* dev_compvals;
		cudasafe(cudaMalloc((void**)&dev_compvals, 4 * numsections * sizeof(int)), "cudaMalloc", __FILE__, __LINE__);
		int half_compheight = COMP_HEIGHT >> 1;
		int half_compwidth = COMP_WIDTH >> 1;
		int comp_quadrantsize = half_compheight * half_compwidth;

		printf("Got to component hist and compval\n");
		scanf("%d", &a);

		histandcompval <<<grid, 256 >>>(dev_compimagearray, dev_compvals, 0, numcompimages, comp_quadrantsize, half_compheight, half_compwidth, 0);
			//eventually free devcompvals

		printf("Got past componenthist and compval\n");
		scanf("%d", &a);

			/*kernel call for full image evaluation */
		//int fullimageheight = original->h; //assign a real value
		int* dev_sectvals;
		cudasafe(cudaMalloc((void**)&dev_sectvals, 4*numsections * sizeof(int)), "cudaMalloc", __FILE__, __LINE__);
		int halfsectionheight = COMP_HEIGHT >> 1;
		int sectionswide = FINAL_WIDTH / COMP_WIDTH;
		int halfsectionwidth = COMP_WIDTH >> 1;
		int full_quadrantsize = halfsectionheight * halfsectionwidth;

		printf("Got to original hist and compval\n");
		scanf("%d", &a);

		histandcompval <<<grid, 256 >>>(dev_fullimage, dev_sectvals, 1, numsections, full_quadrantsize, halfsectionheight, halfsectionwidth, sectionswide);
		printf("Got past original hist and compval\n");
		scanf("%d", &a);
		//eventually free sectvals

		/* closest match array kernel call */

		/* image evaluation stuff goes here
		* information stored as four RGB floats left to right, top to bottom
		* per section into sectvals array */
		cudasafe(cudaFree(dev_origimage), "cudaMalloc", __FILE__, __LINE__);
		cudasafe(cudaFree(dev_fullimage), "cudaMalloc", __FILE__, __LINE__);

		int* sectvals = (int*)malloc(4*numsections * sizeof(int));
		cudaMemcpy(sectvals, dev_sectvals, 4*numsections*sizeof(int), cudaMemcpyDeviceToHost);
		int* compvals = (int*)malloc(4*numcompimages * sizeof(int));
		cudaMemcpy(compvals, dev_compvals, 4*numcompimages*sizeof(int), cudaMemcpyDeviceToHost);

		int* closestarray = genclosestarray(compvals, sectvals, numcompimages, numsections);

		for (int i = 0; i < numsections; i++){ printf("%d ", closestarray[i]); }

		int* dev_closestfit;
		cudasafe(cudaMalloc((void**)&dev_closestfit, numsections * sizeof(int)), "cudaMalloc", __FILE__, __LINE__);

		int stride = 32;
		while (stride < numcompimages) { stride = stride << 1; }
		stride = stride >> 1;

		printf("Got to genclosestarray\n");
		scanf("%d", &a);
		size_t shared_mem = (numcompimages << 1)*sizeof(int);
		genclosestarray <<<4, 512, shared_mem>>>(dev_compvals, dev_sectvals, dev_closestfit, numcompimages, numsections, stride);
		cudaDeviceSynchronize();
		printf("Got past genclosestarray\n");
		scanf("%d", &a);

		cudasafe(cudaFree(dev_sectvals), "cudaFree", __FILE__, __LINE__);
		cudasafe(cudaFree(dev_compvals), "cudaFree", __FILE__, __LINE__);

		//Uint8* dev_finalImage;
		//cudasafe(cudaMalloc((void**)&dev_finalImage, FINAL_HEIGHT*FINAL_WIDTH * 4 * sizeof(Uint8)), "cudaMalloc", __FILE__, __LINE__);

		dim3 block(16, 16, 1);
		
		printf("Got to BuildFinalImg\n");
		scanf("%d", &a);
		//BuildFinalImg <<<grid, block >>>(dev_closestfit, dev_compimagearray, dev_finalImage, sectionswide, COMP_WIDTH*COMP_HEIGHT);
		// start serial buildfinalimg
		Uint8* host_compimagearray = (Uint8*)malloc(COMP_HEIGHT * COMP_WIDTH * numcompimages * 3 * sizeof(Uint8));
		cudasafe(cudaMemcpy(host_compimagearray, dev_compimagearray, COMP_HEIGHT * COMP_WIDTH * numcompimages * 3 * sizeof(Uint8), cudaMemcpyDeviceToHost), "from device to host", __FILE__, __LINE__);
		int* host_closestfit = (int*)malloc(numsections * sizeof(int));
		cudasafe(cudaMemcpy(host_closestfit, dev_closestfit, numsections * sizeof(int), cudaMemcpyDeviceToHost), "from device to host", __FILE__, __LINE__);
		for (int i = 0; i < numsections; i++){ printf("%d ", host_closestfit[i]); }
		Uint8* finalPixels = serialBuildFinalImg(host_closestfit, host_compimagearray, sectionswide, COMP_WIDTH*COMP_HEIGHT);
		// end serial buildfinalimg
		printf("Got past BuildFinalImg\n");
		scanf("%d", &a);

		// these free calls are for serial buildfinalimg
		free(host_compimagearray);
		free(host_closestfit);
		cudaFree(dev_compimagearray);
		cudaFree(dev_closestfit);
		// these free calls are for serial buildfinalimg

		SDL_Surface *finalImage = SDL_CreateRGBSurface(SDL_SWSURFACE, FINAL_WIDTH, FINAL_HEIGHT, 32, rmask, gmask, bmask, amask);
		
		//Uint8* finalPixels = (Uint8*)malloc(FINAL_HEIGHT*FINAL_WIDTH*sizeof(Uint8)*4);

		printf("Got to 411\n");
		scanf("%d", &a);

		//cudasafe(cudaMemcpy(finalPixels, dev_finalImage, FINAL_WIDTH*FINAL_HEIGHT*sizeof(Uint8)*4, cudaMemcpyDeviceToHost), "from device to host", __FILE__, __LINE__);

		printf("Got to 416\n");
		scanf("%d", &a);

		finalImage->pixels = finalPixels;


		printf("Got to 418\n");
		scanf("%d", &a);
		// Free memory

		printf("Got to save\n");
		scanf("%d", &a);
		SDL_SaveBMP(finalImage, "FinalImage.bmp");
		printf("Got past save\n");
		scanf("%d", &a);


		// Free surfaces
		SDL_FreeSurface(original);
		SDL_FreeSurface(finalImage);
		free(finalPixels);
	}
	else
	{
		int a;
		printf("COULD NOT OPEN DIRECTORY");
		scanf("%d", &a);
	}
	SDL_Quit();
}
