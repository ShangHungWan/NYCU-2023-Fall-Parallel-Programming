#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *img, int resX, int resY, int maxIterations)
{
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    float x = lowerX + thisX * stepX;
    float y = lowerY + thisY * stepY;

    int iteration = 0;
    float xtemp;
    while (x * x + y * y < 4 && iteration < maxIterations)
    {
        xtemp = x * x - y * y + x;
        y = 2 * x * y + y;
        x = xtemp;
        iteration++;
    }

    img[thisX + thisY * resX] = iteration;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *DImg;
    int size = resX * resY * sizeof(int);
    size_t pitch;
    cudaHostAlloc(&img, size, cudaHostAllocMapped);
    cudaMallocPitch(&DImg, &pitch, resX * sizeof(int), resY);
    cudaHostGetDevicePointer(&DImg, img, 0);

    dim3 dimGrid(resX / BLOCK_SIZE, resY / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    mandelKernel<<<dimGrid, dimBlock>>>(upperX, upperY, stepX, stepY, DImg, resX, resY, maxIterations);

    cudaFree(DImg);
    cudaFreeHost(img);
}
