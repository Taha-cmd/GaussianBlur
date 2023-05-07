/*
 * a kernel that add the elements of two vectors pairwise
 */
__kernel void vector_add(__global const int *A, __global int *B) {

  __private size_t length = get_global_size(0);
  __private size_t i = get_global_id(0);

  B[i] = i == length - 1 ? A[i] + 0 : A[i] + A[i + 1];
}