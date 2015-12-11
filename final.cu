
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL.h>
#include <SDL_image.h>
#include <dirent.h>
#include <string>
#include <stdio.h>

//FINAL_HEIGHT, FINAL_WIDTH must be evenly divisible by COMP_WIDTH, COMP_HEIGHT

#undef main
#define COMP_WIDTH 20
#define COMP_HEIGHT 20
#define FINAL_HEIGHT 2000
#define FINAL_WIDTH 2000
#define ORIGINAL_IMAGE "original.png"

using namespace std;




void cudasafe(int error, char* message, char* file, int line) {
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s : %i. In %s line %d\n", message, error, file, line);
		exit(-1);
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

	int xStrides = FINAL_WIDTH / gridDim.x + 1;
	int yStrides = FINAL_HEIGHT / gridDim.y + 1;

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

				int sectionIndexX = xIndex - sectionX*FINAL_WIDTH;
				int sectionIndexY = yIndex - sectionY*FINAL_HEIGHT;

				Uint8 red = start[(sectionIndexX + sectionIndexY*FINAL_WIDTH) * 3];
				Uint8 blue = start[(sectionIndexX + sectionIndexY*FINAL_WIDTH) * 3 + 1];
				Uint8 green = start[(sectionIndexX + sectionIndexY*FINAL_WIDTH) * 3 + 2];

				*(Uint32*)finalImage[(xIndex + yIndex*FINAL_WIDTH) * 4] = 0xff000000 | ((((int)red) << 16)) | ((((int)green) << 8)) | ((int)blue);
			}
			yIndex += gridDim.y;
		}
		yIndex = ty + by*blockDim.y;
		xIndex += gridDim.x;
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
						atomicAdd(&(distances[compimageindex]), sqrtaddval * sqrtaddval);
					}
				}
			}
			__syncthreads();

			//find smallest value in distances and put it in an int array at sectimageindex
			if (tx < stride){ location[tx+numcompimages] = 0; }
			for (; stride > 0; stride = stride >> 1){
				if ((tx + stride) <= numcompimages){
					if (distances[tx + stride] < distances[tx]){
						distances[tx] = distances[tx + stride];
						location[tx + numcompimages] = location[tx + stride + numcompimages] | stride;
					}
				}
			}
			__syncthreads();

			if (tx == 0){ closestfit[bx + j*4] = location[0 + numcompimages]; }

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
__global__ void histandcompval(unsigned char* imagearray, int* compvals, int comporfull, int numimages, int size, int height, int width, int numwide)
{
	__shared__ unsigned int privhistr[256];
	__shared__ unsigned int privhistg[256];
	__shared__ unsigned int privhistb[256];

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

		privhistr[tx] = 0;
		privhistg[tx] = 0;
		privhistb[tx] = 0;
		__syncthreads();
		int start = 0;

		if (comporfull == 0){ start = (j << 2) * size + quadrant_offset; }
		else{
			start = ((j << 1) * (size << 2) + (j << 1) % numwide) * size + quadrant_offset;
		}

		for (int i = 0; i * 256 < size; i++){
			int quadrant_x = (tx + i * 256) % width;
			int quadrant_y = (tx + i * 256) / width;
			if (comporfull == 1){ width *= (numwide << 1); }
			if (quadrant_y < height){
				atomicAdd(&(privhistr[imagearray[start + (quadrant_x + quadrant_y * width) * 3]]), 1);
				atomicAdd(&(privhistg[imagearray[start + (quadrant_x + quadrant_y * width) * 3 + 1]]), 1);
				atomicAdd(&(privhistb[imagearray[start + (quadrant_x + quadrant_y * width) * 3 + 2]]), 1);
			}
			__syncthreads();
		}

		privhistr[tx] *= tx;
		privhistg[tx] *= tx;
		privhistb[tx] *= tx;
		__syncthreads();

		for (int stride = 128; stride > 0; stride = stride >> 1){
			if (tx < stride){
				privhistr[tx] += privhistr[tx + stride];
				privhistg[tx] += privhistg[tx + stride];
				privhistb[tx] += privhistb[tx + stride];
			}
			__syncthreads();
		}

		if (tx == 0){
			compvals[3 * (bx + (by << 1) + (j << 2))] = privhistr[0];
			compvals[1 + 3 * (bx + (by << 1) + (j << 2))] = privhistg[0];
			compvals[2 + 3 * (bx + (by << 1) + (j << 2))] = privhistb[0];
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

			printf("%s", name);
			int a;
			scanf("%d", &a);

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
		closedir(dir);

		SDL_Surface *original = IMG_Load(ORIGINAL_IMAGE);

		int numsections = ((FINAL_WIDTH / COMP_WIDTH) * (FINAL_HEIGHT / COMP_HEIGHT));

		/* kernel call for component image evaluation */
		int* dev_compvals;
		cudaMalloc((void**)&dev_compvals, numsections * sizeof(int));
		int half_compheight = COMP_HEIGHT >> 1;
		int half_compwidth = COMP_WIDTH >> 1;
		int comp_quadrantsize = half_compheight * half_compwidth;
		histandcompval <<<4, 256 >> >(dev_compimagearray, dev_compvals, 0, numcompimages, comp_quadrantsize, half_compheight, half_compwidth, 0);
			//eventually free devcompvals

			/*kernel call for full image evaluation */
		//int fullimageheight = original->h; //assign a real value
		int fullimagewidth = original->w; //assign a real value
		int* dev_sectvals;
		cudaMalloc((void**)&dev_sectvals, numsections * sizeof(int));
		Uint8* dev_fullimage;
		int imageByteLength = original->w * original->h * sizeof(Uint8) * 4;
		cudasafe(cudaMalloc((void **)&dev_fullimage, imageByteLength), "Original image allocation ", __FILE__, __LINE__);
		cudasafe(cudaMemcpy(dev_fullimage, original->pixels, imageByteLength, cudaMemcpyHostToDevice), "Copy original image to device ", __FILE__, __LINE__);
		int halfsectionheight = original->h / (FINAL_HEIGHT / COMP_HEIGHT) >> 1;
		int sectionswide = FINAL_WIDTH / COMP_WIDTH;
		int halfsectionwidth = fullimagewidth / sectionswide >> 1;
		int full_quadrantsize = halfsectionheight * halfsectionwidth;
		histandcompval <<<4, 256 >>>(dev_fullimage, dev_sectvals, 1, numsections, full_quadrantsize, halfsectionheight, halfsectionwidth, sectionswide);
		//eventually free sectcompvals

		/* closest match array kernel call */

		/* image evaluation stuff goes here
		* information stored as four RGB floats left to right, top to bottom
		* per section into sectcompvals array */

		cudaFree(dev_fullimage);

		int* dev_closestfit;
		cudaMalloc((void**)&dev_closestfit, numsections * sizeof(int));

		int stride = 32;
		while (stride > numcompimages) { stride = stride << 1; }

		genclosestarray <<<4, 512 >>>(dev_compvals, dev_sectvals, dev_closestfit, numcompimages, numsections, stride);
		cudaDeviceSynchronize();

		cudaFree(dev_sectvals);
		cudaFree(dev_compvals);

		Uint8* dev_finalImage;
		cudaMalloc((void**)&dev_finalImage, FINAL_HEIGHT*FINAL_WIDTH*4*sizeof(Uint8));

		dim3 grid(2, 2, 0);
		dim3 block(16, 16, 0);
		
		BuildFinalImg <<<grid, block >>>(dev_closestfit, dev_compimagearray, dev_finalImage, sectionswide, COMP_WIDTH*COMP_HEIGHT);

		SDL_Surface *finalImage = SDL_CreateRGBSurface(SDL_SWSURFACE, FINAL_WIDTH, FINAL_HEIGHT, 32, rmask, gmask, bmask, amask);
		
		Uint8* finalPixels = (Uint8*)malloc(FINAL_HEIGHT*FINAL_WIDTH*sizeof(Uint8)*4);

		cudasafe(cudaMemcpy(finalPixels, dev_finalImage, FINAL_WIDTH*FINAL_HEIGHT*sizeof(Uint8)*4, cudaMemcpyDeviceToHost), "from device to host", __FILE__, __LINE__);
		finalImage->pixels = finalPixels;

		// Free memory
		cudaFree(finalPixels);

		SDL_SaveBMP(finalImage, "FinalImage.bmp");

		// Free surfaces
		SDL_FreeSurface(original);
		SDL_FreeSurface(finalImage);
	}
	else
	{
		int a;
		printf("COULD NOT OPEN DIRECTORY");
		scanf("%d", &a);
	}
	SDL_Quit();
}