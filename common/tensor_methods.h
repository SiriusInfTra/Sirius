#ifndef COLSERVE_TENSOR_METHODS_H
#define COLSERVE_TENSOR_METHODS_H

// #include <dlpack/dlpack.h>
#include <ATen/TensorUtils.h>
#include <c10/core/MemoryFormat.h>

#include <iostream>
#include <vector>

#include "tensor.h"
#include "dlpack.h"
#include "dtype_helper.h"


namespace colserve {
namespace sta {

STensor Null(at::IntArrayRef size, DLDataType dtype);

STensor Empty(at::IntArrayRef size, at::MemoryFormat memory_format, 
              DLDataType dtype, MemType mtype);

STensor EmptyStrided(at::IntArrayRef size, at::IntArrayRef stride, 
                     DLDataType dtype, MemType mtype);

STensor ViewDtype(uint64_t tensor, DLDataType dtype);

STensor ViewShapeDtype(const STensor tensor, at::IntArrayRef size, DLDataType dtype);

STensor AsStrided(const STensor tensor, at::IntArrayRef size,
                  at::IntArrayRef stride, 
                  c10::optional<int64_t> storage_offset);

void AsStrided_(STensor tensor, at::IntArrayRef size,
                at::IntArrayRef stride, 
                c10::optional<int64_t> storage_offset);

}
}

#endif