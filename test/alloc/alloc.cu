#include <iostream>
#include <chrono>
#include <atomic>
#include <pthread.h>
#include <vector>
#include <thread>
#include <algorithm>
#include <numeric>

#define CUDA_CALL(func) do { \
  auto error = func; \
  if (error != cudaSuccess) { \
    std::cout << #func << " " << cudaGetErrorString(error); \
    exit(EXIT_FAILURE); \
  } \
  } while (0)

double test_size(size_t alloc_mb, size_t dummy_mb) {
  int* dummy_ptr;
  int* alloc_ptr;
  
  int n = 10;
  size_t alloc_sz = alloc_mb * 1024 * 1024;
  size_t dummy_sz = dummy_mb * 1024 * 1024;

  double total = 0;
  CUDA_CALL(cudaMalloc(&dummy_ptr, dummy_sz));
  for (int i = 0; i < n; i++) {
    auto t0 = std::chrono::steady_clock::now();
    CUDA_CALL(cudaMalloc(&alloc_ptr, alloc_sz));
    auto t1 = std::chrono::steady_clock::now();
    // std::cout << "Allocated " << alloc_sz << "/" << dummy_sz << " " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;
    total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    CUDA_CALL(cudaFree(alloc_ptr));
  }
  CUDA_CALL(cudaFree(dummy_ptr));
  std::cout << "test_size " << alloc_mb << "/" << dummy_mb << " " << total / n << " ms" << std::endl;
  return total / n;
}

double test_con(size_t concurrency, size_t alloc_mb, size_t dummy_mb) {
  int n = 100;
  // std::atomic<uint64_t> total_us{0};
  std::vector<double> total_ms(concurrency, 0);

  size_t dummy_sz = dummy_mb * 1024 * 1024;
  size_t alloc_sz = alloc_mb * 1024 * 1024;
  int* dummy_ptr;
  CUDA_CALL(cudaMalloc(&dummy_ptr, dummy_sz));

  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, NULL, concurrency + 1);
  auto lamda = [&](int rank) {
    pthread_barrier_wait(&barrier);
    int* alloc_ptr;

    for (int i = 0; i < n; i++) {
      auto t0 = std::chrono::steady_clock::now();
      CUDA_CALL(cudaMalloc(&alloc_ptr, alloc_sz));
      auto t1 = std::chrono::steady_clock::now();
      total_ms[rank] += std::chrono::duration<double, std::milli>(t1 - t0).count();
      CUDA_CALL(cudaFree(alloc_ptr));
    }
  };
  
  std::vector<std::thread> threads;
  for (int i = 0; i < concurrency; i++) {
    threads.emplace_back(lamda, i);
  }
  pthread_barrier_wait(&barrier);

  for (auto& t : threads) {
    t.join();
  }

  CUDA_CALL(cudaFree(dummy_ptr));
  auto avg_ltc_ms = std::accumulate(total_ms.begin(), total_ms.end(), 0.0) / concurrency / n;
  std::cout << "test_con " << concurrency << "-" << alloc_mb << "/" << dummy_mb << " "
            << avg_ltc_ms << " ms" << std::endl;
  return avg_ltc_ms;
}

int main() {
  
  // size_t alloc_sz = 250 * 1024 * 1024;
  // size_t dummy_sz = 2ULL * 1024 * 1024 * 1024;

  // test_size(100, 15 * 1024);
  // test_size(100, 12 * 1024);
  // test_size(100, 10 * 1024);
  // test_size(100, 6 * 1024);
  // std::cout << std::endl;

  // test_size(400, 15 * 1024);
  // test_size(400, 12 * 1024);
  // test_size(400, 10 * 1024);
  // test_size(400, 6 * 1024);
  // std::cout << std::endl;

  test_con(1, 100, 0);
  test_con(4, 100, 0);
  test_con(16,100,  0);
  test_con(32,100,  0);
  test_con(64,100,  0);
  // test_con(128,100,  0);
  std::cout << std::endl;
  
  test_con(1,200, 0);
  test_con(4,200, 0);
  test_con(16,200, 0);
  test_con(32,200, 0);
  test_con(64,200, 0);
  // test_con(128,200, 0);
  std::cout << std::endl;

  // test_con(1, 100, 12 * 1024);
  // test_con(4, 100, 12 * 1024);
  // test_con(16, 100, 12 * 1024);
  // test_con(32, 100, 12 * 1024);
  // // test_con(64, 100, 12 * 1024);
  // std::cout << std::endl;
  
  return 0;
}