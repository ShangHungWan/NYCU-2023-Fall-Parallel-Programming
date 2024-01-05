#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSIze = imageHeight * imageWidth * sizeof(float);

    cl_command_queue cmdQueue = clCreateCommandQueue(*context, *device, 0, &status);

    cl_mem bufInput = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageSIze, NULL, &status);
    cl_mem bufFilter = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize, NULL, &status);
    cl_mem bufOutput = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSIze, NULL, &status);

    status = clEnqueueWriteBuffer(cmdQueue, bufInput, CL_TRUE, 0, imageSIze, inputImage, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmdQueue, bufFilter, CL_TRUE, 0, filterSize, filter, 0, NULL, NULL);

    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufInput);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufFilter);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufOutput);
    status = clSetKernelArg(kernel, 3, sizeof(cl_int), &filterWidth);
    status = clSetKernelArg(kernel, 4, sizeof(cl_int), &imageHeight);
    status = clSetKernelArg(kernel, 5, sizeof(cl_int), &imageWidth);

    size_t globalWorkSize[2] = {imageWidth, imageHeight};
    size_t localWorkSize[2] = {8, 8};

    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    status = clEnqueueReadBuffer(cmdQueue, bufOutput, CL_TRUE, 0, imageSIze, outputImage, 0, NULL, NULL);
}