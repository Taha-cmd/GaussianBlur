__kernel void GaussianBlur(__global const float4* inputBuffer, __global float4* outputBuffer, int width, int height, int sigma, int pass)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int index = y * width + x;

    // Calculate the blur radius based on the sigma value
    int radius = (3 * sigma);

    // Create a local memory buffer
    __local float4 localBuf[512];

    // Perform the column-wise pass
    if (pass == 1)
    {
        //printf("I: %d", index);
        //printf("X: %d", x);
        // Load the input pixels into local memory
        //localBuf[x] = inputBuffer[index];

        // Wait for all work-items to finish loading
        barrier(CLK_LOCAL_MEM_FENCE);

        // Apply the Gaussian blur filter to the loaded pixels
        float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = -radius; i <= radius; i++)
        {
            int offset = y + i;
            if (offset >= 0 && offset < height)
            {
                sum += inputBuffer[x + offset * width];
            }
        }

        // Store the blurred pixel value
        outputBuffer[index] = sum / (2 * radius + 1);
    }

    // Perform the row-wise pass
    else if (pass == 2)
    {
        // Load the input pixels into local memory
        localBuf[y] = inputBuffer[index];

        // Wait for all work-items to finish loading
        barrier(CLK_LOCAL_MEM_FENCE);

        // Apply the Gaussian blur filter to the loaded pixels
        float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = -radius; i <= radius; i++)
        {
            int offset = x + i;
            if (offset >= 0 && offset < width)
            {
                sum += inputBuffer[y * width + offset];
            }
        }

        // Store the blurred pixel value
        outputBuffer[index] = sum / (2 * radius + 1);
    }
}