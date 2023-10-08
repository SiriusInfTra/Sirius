#ifndef COLSERVE_TENSOR_METHODS_H
#define COLSERVE_TENSOR_METHODS_H

// #include <dlpack/dlpack.h>
#include <ATen/TensorUtils.h>
#include "undef_log.h"

#include <iostream>
#include <vector>

#include "dlpack.h"

namespace colserve {
namespace sta {

uint64_t Empty(at::IntArrayRef shape, DLDataType dtype);

uint64_t AsStrided(uint64_t handle, at::IntArrayRef size, 
                   at::IntArrayRef stride, 
                   c10::optional<int64_t> storage_offset);

}
}

#endif