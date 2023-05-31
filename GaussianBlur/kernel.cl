int getIndex(int width, int offset, int dimension, int dimOffset) {

    if (dimension == 0) {
        return dimOffset + offset * width;
    }

    return dimOffset * width + offset;
}

__kernel void GaussianBlur(__global const float4* inputBuffer, __global float4* outputBuffer, int width, int height, int sigma, int dimension)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int index = y * width + x;

    // Calculate the blur radius based on the sigma value
    int radius = (3 * sigma);

    int otherDim = dimension == 0 ? 1 : 0;
    int dimOffset = get_global_id(dimension);
    int otherDimOffset = get_global_id(otherDim);

    // Create a local memory buffer
    __local float4 localBuf[512]; //use vector add to detemrine local workgroup size and input as buffersize
 

    int range = dimension == 0 ? height : width;

    //localBuf[x] = inputBuffer[index];

    // Wait for all work-items to finish loading
    barrier(CLK_LOCAL_MEM_FENCE);

    // Apply the Gaussian blur filter to the loaded pixels
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = -radius; i <= radius; i++)
    {

        int offset = otherDimOffset + i;
         
        if (offset >= 0 && offset < range)
        {
            int localIndex = getIndex(width, offset, dimension, dimOffset);
            sum += inputBuffer[localIndex];
        }
        
    }

    // Store the blurred pixel value
    outputBuffer[index] = sum / (2 * radius + 1);


}
