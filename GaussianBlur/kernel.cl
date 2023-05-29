#define KERNEL_SIZE 5

__kernel void gaussian_blur(__global const float4* inputBuffer,
    __global float4* outputBuffer, const int width,
    const int height, const int pass) {

    const int2 coords = (int2)(get_global_id(0), get_global_id(1));

    const int pixelIndex = coords.y * width + coords.x;

    const float gaussian_kernel[5][5] = {
    { 0.023528f, 0.033969f, 0.038393f, 0.033969f, 0.023528f },
    { 0.033969f, 0.049045f, 0.055432f, 0.049045f, 0.033969f },
    { 0.038393f, 0.055432f, 0.062651f, 0.055432f, 0.038393f },
    { 0.033969f, 0.049045f, 0.055432f, 0.049045f, 0.033969f },
    { 0.023528f, 0.033969f, 0.038393f, 0.033969f, 0.023528f }
    };

    const int radius = KERNEL_SIZE / 2;
    __local float4 localBuffer[KERNEL_SIZE];
    float4 blurredPixel = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    // Column-wise pass
    if(pass == 1)
    {

        // Load the input pixels into local memory
        for (int i = -radius; i <= radius; i++) {
            const int neighborY = coords.y + i;
            const int localIndex = i + radius;

            if (neighborY >= 0 && neighborY < height) {
                const int neighborIndex = neighborY * width + coords.x;
                localBuffer[localIndex] = inputBuffer[neighborIndex];
            }
        }

        // Wait for all work-items to finish loading
        barrier(CLK_LOCAL_MEM_FENCE);

        // Apply the Gaussian blur filter to the loaded pixels
        for (int i = -radius; i <= radius; i++) {
            const int localIndex = i + radius;
            const float4 weight = gaussian_kernel[i + radius];

            blurredPixel += localBuffer[localIndex] * weight;
        }

        outputBuffer[pixelIndex] = blurredPixel;
    }

    // Row-wise pass
    else if (pass == 2) {

        // Load the input pixels into local memory
        for (int i = -radius; i <= radius; i++) {
            const int neighborX = coords.x + i;
            const int localIndex = i + radius;

            if (neighborX >= 0 && neighborX < width) {
                const int neighborIndex = coords.y * width + neighborX;
                localBuffer[localIndex] = outputBuffer[neighborIndex];
            }
        }

        // Wait for all work-items to finish loading
        barrier(CLK_LOCAL_MEM_FENCE);

        // Apply the Gaussian blur filter to the loaded pixels
        for (int i = -radius; i <= radius; i++) {
            const int localIndex = i + radius;
            const float4 weight = gaussian_kernel[i + radius];

            blurredPixel += localBuffer[localIndex] * weight;
        }

        outputBuffer[pixelIndex] = blurredPixel;
    }
}