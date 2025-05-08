#pragma once

#include <cstdlib>

#include "utils/log.h"
#include "utils/common.h"
#include "hal/cuda/cuda.h"
#include "hal/cuda/driver.h"

#define CUDA_ASSERT(cmd) \
    do { \
        CUresult result = cmd; \
        if (UNLIKELY(result != CUDA_SUCCESS)) { \
            const char *str; \
            xsched::hal::cuda::Driver::GetErrorString(result, &str); \
            XERRO("cuda error %d: %s", result, str); \
        } \
    } while (0);
