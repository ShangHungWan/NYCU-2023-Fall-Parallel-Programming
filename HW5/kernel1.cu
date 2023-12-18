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

    float z_re = x, z_im = y;
    int iteration = 0;
    while (z_re * z_re + z_im * z_im <= 4.f && iteration < maxIterations)
    {
        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2 * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iteration++;
    }

    img[thisX + thisY * resX] = iteration;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *DImg, *HImg;
    int size = resX * resY * sizeof(int);

    cudaMalloc(&DImg, size);
    HImg = (int *)malloc(size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((resX + dimBlock.x - 1) / dimBlock.x, (resY + dimBlock.y - 1) / dimBlock.y);
    mandelKernel<<<dimGrid, dimBlock>>>(lowerX, lowerY, stepX, stepY, DImg, resX, resY, maxIterations);

    cudaMemcpy(HImg, DImg, size, cudaMemcpyDeviceToHost);
    memcpy(img, HImg, size);

    free(HImg);
    cudaFree(DImg);
}
