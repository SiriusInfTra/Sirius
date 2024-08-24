#ifndef COLSERVE_COMMON_UTIL_H
#define COLSERVE_COMMON_UTIL_H

#include <common/log_as_glog_sta.h>
#include <common/device_manager.h>

#include <boost/format.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include <sys/unistd.h>

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <sstream>


namespace colserve {

#define COL_CUDA_CALL(func) do { \
  auto error = func; \
  if (error != cudaSuccess) { \
    LOG(FATAL) << #func << " " << cudaGetErrorString(error); \
    exit(EXIT_FAILURE); \
  } \
  } while (0)

#define COL_CU_CALL(func) \
  do { \
    auto err = func; \
    if (err != CUDA_SUCCESS) { \
      const char* pstr = nullptr; \
      cuGetErrorString(err, &pstr); \
      LOG(FATAL) << #func << ": " << pstr; \
      exit(EXIT_FAILURE); \
    } \
  } while (0);

#define COL_NVML_CALL(func) do{ \
    auto error = func; \
    if (error != NVML_SUCCESS) { \
      LOG(FATAL) << #func << " " << nvmlErrorString(error); \
      exit(EXIT_FAILURE); \
    } \
  } while(0);

#define COL_NCCL_CALL(func) do { \
  auto error = func; \
  if (error != ncclSuccess) { \
    LOG(FATAL) << #func << " " << ncclGetErrorString(error); \
    exit(EXIT_FAILURE); \
  } \
  } while (0)

constexpr int MAX_DEVICE_NUM = 8;

template <typename T, size_t dim_1, size_t dim_2>
using array_2d_t = std::array<std::array<T, dim_2>, dim_1>;

using memory_byte_t = size_t;
using memory_mb_t = double;

namespace memory_literals {

constexpr size_t operator ""_B(unsigned long long n) {
  return static_cast<size_t>(n);
}

constexpr size_t operator ""_B(long double n) {
  return static_cast<size_t>(n);
}

constexpr size_t operator ""_KB(unsigned long long n) {
  return static_cast<size_t>(n) * 1024;
}

constexpr size_t operator ""_KB(long double n) {
  return static_cast<size_t>(n * 1024);
}

constexpr size_t operator ""_MB(unsigned long long n) {
  return static_cast<size_t>(n) * 1024 * 1024;
}

constexpr size_t operator ""_MB(long double n) {
  return static_cast<size_t>(n * 1024 * 1024);
}

constexpr size_t operator ""_GB(unsigned long long n) {
  return static_cast<size_t>(n) * 1024 * 1024 * 1024;
}

constexpr size_t operator ""_GB(long double n) {
  return static_cast<size_t>(n * 1024 * 1024 * 1024);
}
}

namespace sta {
inline double ByteToMB(size_t nbytes) {
  return static_cast<double>(nbytes) / 1024 / 1024;
}

inline std::string PrintByte(size_t nbytes) {
  return str(boost::format("%.2fMB (%dB)") % ByteToMB(nbytes) % nbytes);
}
}

inline std::string GetDefaultShmNamePrefix(int device_id) {
  CHECK_LT(device_id, sta::DeviceManager::GetNumVisibleGpu());
  return (boost::format("colserve_%s_%s") % sta::DeviceManager::GetGpuSystemUuid(device_id) 
                                          % getuid())
                                          .str();
}

inline std::string GetDefaultShmNamePrefix() {
  return (boost::format("colserve_%s") % getuid()).str();
}

} // namespace colserve



using namespace colserve::memory_literals;

#endif