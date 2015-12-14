__global__ void histandcompval(unsigned char* imagearray, int* compvals, int comporfull, int numimages, int size, int height, int width, int numwide)
{
    extern __shared__ float grayvals[];

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    int by = bx / 2;
    bx = bx % 2;

    if (tx < 64){ privhist[tx] = 0; }
    __syncthreads();

    int quadrant_offset = 0;
    int ystep = 0;
    if(comporfull == 0){
        quadrant_offset = bx * width + by * size << 1;
        ystep = width;
    }
    else{
        quadrant_offset = bx * width + by * numwide * size << 1;
        ystep = width * numwide << 1;
    }

    int start = 0;

    for (int j = 0; j < numimages; j++){
        if (comporfull == 0){ start = j * (size << 2) + quadrant_offset; }
        else{
            start = (j / numwide) * (size << 2) * numwide + (j % numwide) * (width << 1) + quadrant_offset;
        }

        for (int i = 0; i * 256 < size; i++){
            int quadrant_x = (tx + i * 256) % width;
            int quadrant_y = (tx + i * 256) / width;
            if (quadrant_y < height){
                grayvals[tx + i * 256] = imagearray[start + (quadrant_x + quadrant_y * ystep) * 3] * 0.07;
                grayvals[tx + i * 256] += imagearray[start + (quadrant_x + quadrant_y * ystep) * 3 + 1] * 0.72;
                grayvals[tx + i * 256] += imagearray[start + (quadrant_x + quadrant_y * ystep) * 3 + 2] * 0.21;
            }
        }

        int stride = 32;
        for (; stride < size; stride = stride << 1; ){ }
        stride = stride >> 1;
        for (int stride = 32; stride > 0; stride = stride >> 1){
            if (tx < stride){
                grayvals[tx] += grayvals[tx + stride];
            }
            __syncthreads();
        }

        if (tx == 0){
            compvals[bx + (by << 1) + (j << 2)] = grayvals[0] / size;
        }
        __syncthreads();
    }
}
