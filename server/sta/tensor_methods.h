#ifndef COLSERVE_TENSOR_METHODS_H
#define COLSERVE_TENSOR_METHODS_H

// #include <dlpack/dlpack.h>
#include <ATen/TensorUtils.h>
#include <c10/core/MemoryFormat.h>

#include <iostream>
#include <vector>

#include "tensor_pool.h"
#include "dlpack.h"
#include "dtype_helper.h"


namespace colserve {
namespace sta {

uint64_t Null(at::IntArrayRef size, DLDataType dtype);

STensor RawNull(at::IntArrayRef size, DLDataType dtype);

uint64_t Empty(at::IntArrayRef size, at::MemoryFormat memory_format, DLDataType dtype, MemType mtype);

STensor RawEmpty(at::IntArrayRef size, DLDataType dtype, MemType mtype);

uint64_t EmptyStrided(at::IntArrayRef size, at::IntArrayRef stride, 
                      DLDataType dtype, MemType mtype);

uint64_t ViewDtype(uint64_t handle, DLDataType dtype);

uint64_t ViewShapeDtype(uint64_t handle, at::IntArrayRef size, DLDataType dtype);

STensor RawViewShapeDtype(STensor tensor, at::IntArrayRef size, DLDataType dtype);

uint64_t AsStrided(uint64_t handle, at::IntArrayRef size,
                   at::IntArrayRef stride, 
                   c10::optional<int64_t> storage_offset);

void AsStrided_(uint64_t handle, at::IntArrayRef size,
                at::IntArrayRef stride, 
                c10::optional<int64_t> storage_offset);

}
}

#endif