#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#include "utils/log.h"
#include "utils/common.h"

#define CUDART_ASSERT(cmd) \
    do { \
        cudaError_t result = cmd; \
        if (UNLIKELY(result != cudaSuccess)) { \
            XERRO("cuda runtime error %d", result); \
        } \
    } while (0);
