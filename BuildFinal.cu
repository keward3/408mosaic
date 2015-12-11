__global__ void BuildFinalImg(int* closestFit, int* sections, int* finalImage, int sectionWidth, int sectionHeight, int sectionsPerRow, int height, int width, int pixelsPerSection)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int xIndex = tx + bx*blockDim.x;
	int yIndex = ty + by*blockDim.y;

	int xStrides = width / gridDim.x + 1;
	int yStrides = width / gridDim.y + 1;

	for (int i = 0; i < xStrides; i++)
	{
		for (int j = 0; j < yStrides; j++)
		{
			if (xIndex < width && yIndex < height)
			{
				int sectionX = xIndex / sectionWidth;
				int sectionY = yIndex / sectionHeight;

				int sectionIndex = sectionY*sectionsPerRow + sectionX;
				int closest = closestFit[sectionIndex];

				int* start = sections + closest*pixelsPerSection;

				int sectionIndexX = xIndex - sectionX*sectionWidth;
				int sectionIndexY = yIndex - sectionY*sectionWidth;

				finalImage[xIndex + yIndex*width] = start[sectionIndexX + sectionIndexY*sectionWidth];
			}
			yIndex += gridDim.y;
		}
		yIndex = ty + by*blockDim.y;
		xIndex += gridDim.x;
	}
}
