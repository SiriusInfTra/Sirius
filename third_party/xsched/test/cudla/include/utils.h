#pragma once

#include <cudla.h>
#include <cuda_runtime.h>

#include "utils/log.h"

#define CUDLA_ASSERT(cmd) \
    do { \
        cudlaStatus result = cmd; \
        if (UNLIKELY(result != cudlaSuccess)) { \
            XERRO("cudla error %d", result);    \
        } \
    } while (0);

#define CUDART_ASSERT(cmd) \
    do { \
        cudaError_t result = cmd; \
        if (UNLIKELY(result != cudaSuccess)) {      \
            XERRO("cuda runtime error %d", result); \
        } \
    } while (0);
