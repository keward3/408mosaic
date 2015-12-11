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
