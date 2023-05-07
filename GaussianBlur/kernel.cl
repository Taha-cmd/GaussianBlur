__kernel void gaussian_blur(__global const float4 *inputBuffer,
                            __global float4 *outputBuffer, const int width,
                            const int height) {

  const int2 coords = (int2)(get_global_id(0), get_global_id(1));

  const int pixelIndex = coords.y * width + coords.x;

  const float gaussian_kernel[5][5] = {
  { 0.023528f, 0.033969f, 0.038393f, 0.033969f, 0.023528f },
  { 0.033969f, 0.049045f, 0.055432f, 0.049045f, 0.033969f },
  { 0.038393f, 0.055432f, 0.062651f, 0.055432f, 0.038393f },
  { 0.033969f, 0.049045f, 0.055432f, 0.049045f, 0.033969f },
  { 0.023528f, 0.033969f, 0.038393f, 0.033969f, 0.023528f }
  };

  float4 blurredPixel = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

  for (int j = -2; j <= 2; j++) {
    for (int i = -2; i <= 2; i++) {
      const int neighborX = coords.x + i;
      const int neighborY = coords.y + j;

      if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
        const int neighborIndex = neighborY * width + neighborX;
        blurredPixel += inputBuffer[neighborIndex] * gaussian_kernel[j + 2][i + 2];
      }
    }
  }

  outputBuffer[pixelIndex] = blurredPixel;
  // outputBuffer[pixelIndex] = inputBuffer[pixelIndex];
}
