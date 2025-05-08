#include "utils.h"

__device__ __forceinline__ void Wait(uint64_t clock_cnt)
{
    if (clock_cnt == 0) return;
    uint64_t elapsed = 0;
    uint64_t start = clock64();
    while (elapsed < clock_cnt) { elapsed = clock64() - start; }
}

__global__ void WaitFlag(int32_t *flag)
{
    while (*flag == 0) { Wait(1); }
}

__global__ void Sleep(uint64_t clock_cnt)
{
    Wait(clock_cnt);
}
