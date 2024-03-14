#ifndef COLSERVE_COMMON_UTIL_H
#define COLSERVE_COMMON_UTIL_H

#include <iostream>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define CUDA_CALL(func) do { \
  auto error = func; \
  if (error != cudaSuccess) { \
    LOG(FATAL) << #func << " " << cudaGetErrorString(error); \
    exit(EXIT_FAILURE); \
  } \
  } while (0)

#define CU_CALL(func) \
  do { \
    auto err = func; \
    if (err != CUDA_SUCCESS) { \
      const char* pstr = nullptr; \
      cuGetErrorString(err, &pstr); \
      LOG(FATAL) << #func << ": " << pstr; \
    } \
  } while (0);


namespace colserve {
namespace literals {

constexpr size_t operator ""_KB(unsigned long long n) {
  return static_cast<size_t>(n) * 1024;
}

constexpr size_t operator ""_KB(long double n) {
  return static_cast<size_t>(n) * 1024;
}

constexpr size_t operator ""_MB(unsigned long long n) {
  return static_cast<size_t>(n) * 1024 * 1024;
}

constexpr size_t operator ""_MB(long double n) {
  return static_cast<size_t>(n) * 1024 * 1024;
}

constexpr size_t operator ""_GB(unsigned long long n) {
  return static_cast<size_t>(n) * 1024 * 1024 * 1024;
}

constexpr size_t operator ""_GB(long double n) {
  return static_cast<size_t>(n) * 1024 * 1024 * 1024;
}

}
}

using namespace colserve::literals;

#endif