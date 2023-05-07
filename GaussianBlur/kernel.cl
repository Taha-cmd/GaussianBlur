__kernel void gaussian_blur(__global const float *input, __global float *output,
                            const int size) {
  //const float sigma = 5.0f;
  //const int radius = (int)ceil(sigma * 3);
  //const int diameter = radius * 2 + 1;
  //const float guassianKernel[3][3] = {{0.0625f, 0.125f, 0.0625f},
  //                                    {0.125f, 0.25f, 0.125f},
  //                                    {0.0625f, 0.125f, 0.0625f}};

  int i = get_global_id(0);
  int j = get_global_id(1);

  //if (i >= size || j >= size) {
  //  return;
  //}

  //float sum = 0.0f;
  //for (int x = -radius; x <= radius; x++) {
  //  for (int y = -radius; y <= radius; y++) {
  //    int xi = i + x;
  //    int yj = j + y;
  //    if (xi >= 0 && xi < size && yj >= 0 && yj < size) {
  //      float value = input[xi * size + yj];
  //      float kernelValue = guassianKernel[x + radius][y + radius];
  //      sum += value * kernelValue;
  //    }
  //  }
  //}

  //printf("%f", sum);
  output[i * size + j] = 5;
}
