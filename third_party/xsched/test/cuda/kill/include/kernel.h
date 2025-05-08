#pragma once

#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ void Wait(uint64_t clock_cnt);

__global__ void WaitFlag(int32_t *flag);
__global__ void Sleep(uint64_t clock_cnt);
