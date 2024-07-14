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
      exit(EXIT_FAILURE); \
    } \
  } while (0);

#define NVML_CALL(func) do{ \
    auto error = func; \
    if (error != NVML_SUCCESS) { \
      LOG(FATAL) << #func << " " << nvmlErrorString(error); \
      exit(EXIT_FAILURE); \
    } \
  } while(0);


constexpr int MAX_DEVICE_NUM = 8;

using memory_nbyte_t = size_t;
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

inline std::string ByteDisplay(size_t nbytes) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2)
     << static_cast<double>(nbytes) / 1024 / 1024 << "MB (" << nbytes << " Bytes)";
  return ss.str();
}
}

inline std::string GetDefaultShmNamePrefix(int device_id) {
  CHECK_LT(device_id, sta::DeviceManager::GetNumVisibleGpu());
  return (boost::format("colserve_%s_%s") % sta::DeviceManager::GetGpuSystemUuid(device_id) 
                                          % getuid())
                                          .str();
}

} // namespace colserve



using namespace colserve::memory_literals;

#endif