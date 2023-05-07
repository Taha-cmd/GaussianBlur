__kernel void gaussian_blur(__global const float *input, __global float *output,
                           const int width, const int height) {

  const int2 coords = (int2)(get_global_id(0), get_global_id(1));

  const int pixelIndex = coords.y * width + coords.x;

  const float gaussian_kernel[5] = {0.0625f, 0.125f, 0.25f, 0.125f,
                                    0.0625f}; // Gaussian kernel values

  float blurredPixel = 0.0f;

  for (int i = -2; i <= 2; i++) {
    const int neighborIndex = pixelIndex + i;
    const int neighborX = coords.x + i;

    if (neighborX >= 0 && neighborX < width) {
      blurredPixel += input[neighborIndex] * gaussian_kernel[i + 2];
    }
  }

  output[pixelIndex] = blurredPixel;
}
