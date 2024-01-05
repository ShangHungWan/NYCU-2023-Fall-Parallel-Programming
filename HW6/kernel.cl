__kernel void convolution(
    const __global float *inputImage,
    const __global float *filter,
    __global float *outputImage,
    const int filterWidth,
    const int imageHeight,
    const int imageWidth)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int diff = filterWidth / 2;

    float sum = 0.0;
    for (int i = y - diff, filter_i = 0; i <= y + diff; i++, filter_i++)
    {
        if (i < 0 || i >= imageHeight)
            continue;

        for (int j = x - diff, filter_j = 0; j <= x + diff; j++, filter_j++)
        {
            if (j < 0 || j >= imageWidth)
                continue;

            sum += inputImage[i * imageWidth + j] * filter[filter_i * filterWidth + filter_j];
        }
    }
    outputImage[y * imageWidth + x] = sum;
}
