__global__ void cudaTransform(Uint8 *output, Uint8 *input, Uint16 pitchOutput, Uint16 pitchInput, Uint8 bytesPerPixelInput, Uint8 bytesPerPixelOutput, float xRatio, float yRatio){
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

	Uint8 *p = output + blockIdx.y * pitchOutput + blockIdx.x * bytesPerPixelOutput;
	*(Uint32*)p = 0xff000000 | ((((int)red) << 16)) | ((((int)green) << 8)) | ((int)blue);
}

int main(int argc, char *args[]) {

	SDL_Init(SDL_INIT_EVERYTHING);

	DIR *dir;

	struct dirent *picture;

	Uint32 amask = 0xff000000;
	Uint32 rmask = 0x00ff0000;
	Uint32 gmask = 0x0000ff00;
	Uint32 bmask = 0x000000ff;

	int numcompimages = 0;

	
	

	if ((dir = opendir("./Images")) != NULL)
	{
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

				// New width of image
				int rWidth = 3000;
				int newWidth = image->w + (rWidth - image->w);
				int newHeight = image->h + (rWidth - image->w);
				dim3 grid(newWidth, newHeight);

				// Create scaled image surface
				SDL_Surface *newImage = SDL_CreateRGBSurface(SDL_SWSURFACE, newWidth, newHeight, 32, rmask, gmask, bmask, amask);
				int newImageByteLength = newImage->w * newImage->h * sizeof(Uint8)*newImage->format->BytesPerPixel;

				float xRatio = ((float)(image->w - 1)) / newImage->w;
				float yRatio = ((float)(image->h - 1)) / newImage->h;

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

				// Allocate new image on DEVICE
				Uint8 *newPixels_dyn;
				Uint8 *newPixels = (Uint8*)malloc(newImageByteLength);

				cudasafe(cudaMalloc((void **)&newPixels_dyn, newImageByteLength), "New image allocation ", __FILE__, __LINE__);

				// Start measuring time
				cudaEventRecord(start, 0);

				// Do the bilinear transform on CUDA device
				cudaTransform <<< grid, 1 >>>(newPixels_dyn, pixels_dyn, newImage->pitch, image->pitch, image->format->BytesPerPixel, newImage->format->BytesPerPixel, xRatio, yRatio);

				// Stop the timer
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);

				// Copy scaled image to host
				cudasafe(cudaMemcpy(newPixels, newPixels_dyn, newImageByteLength, cudaMemcpyDeviceToHost), "from device to host", __FILE__, __LINE__);
				newImage->pixels = newPixels;

				// Free memory
				cudaFree(pixels_dyn);
				cudaFree(newPixels_dyn);

				cudaEventElapsedTime(&time, start, stop);
				printf("Time for the kernel: %f ms\n", time);

				string savename = loadname.substr(0, loadname.length() - 4) + "2.bmp";
				//Save image
				SDL_SaveBMP(newImage, savename.c_str());

				// Free surfaces
				SDL_FreeSurface(image);
				SDL_FreeSurface(newImage);

				numcompimages++;
			}

		}
		closedir(dir);


	}
	else
	{
		int a;
		printf("COULD NOT OPEN DIRECTORY");
		scanf("%d", &a);
	}
	SDL_Quit();
}
