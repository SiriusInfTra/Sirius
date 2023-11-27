#include <iostream>

using namespace std;

#define CUDA_CALL(func) do { \
    auto error = func; \
    if (error != cudaSuccess) { \
    std::cout << #func << " " << cudaGetErrorString(error); \
      exit(EXIT_FAILURE); \
    } \
  } while (0);

__global__ void kernel(int n, int *x) {
  for (int i = 0; i < n; i++) {
    x[i] = x[i] + 1;
  }
}

int main() {
  int n = 1 << 20;
  int *x, *y;
  CUDA_CALL(cudaMallocHost(&y, n * sizeof(int)));
  CUDA_CALL(cudaMalloc(&x, 2 * n * sizeof(int)));
  for (int i = 0; i < n; i++) {
    y[i] = i;
  }

  cudaStream_t s1, s2;
  CUDA_CALL(cudaStreamCreate(&s1));
  CUDA_CALL(cudaStreamCreate(&s2));

  for (int i = 0; i < 10; i++) {
    CUDA_CALL(cudaMemcpyAsync(x, y, 2  * n * sizeof(int), cudaMemcpyHostToDevice, s2));
    kernel<<<1, 1, 0, s1>>>(n, x);
  }
  CUDA_CALL(cudaStreamSynchronize(s1));
  CUDA_CALL(cudaStreamSynchronize(s2));
  
  int *z;
  cudaMallocHost(&z, n * sizeof(int));
  CUDA_CALL(cudaMemcpyAsync(z, x, n * sizeof(int), cudaMemcpyDeviceToHost, s1));
  for (int i = 0; i < 20; i++) {
    std::cout << z[i] << " ";
  }
  std::cout << std::endl;
  cout << "Success!" << endl;
  return 0;
}