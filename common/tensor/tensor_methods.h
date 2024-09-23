#ifndef COLSERVE_TENSOR_METHODS_H
#define COLSERVE_TENSOR_METHODS_H

// #include <dlpack/dlpack.h>
// #include <ATen/TensorUtils.h>
// #include <c10/core/MemoryFormat.h>

#include <common/tensor/tensor.h>
#include <common/tensor/dlpack.h>
#include <common/tensor/dtype_helper.h>

#include <iostream>
#include <vector>
#include <optional>


namespace colserve {
namespace sta {


STensor Null(const dim_vec_t &size, DLDevice device, DLDataType dtype);

STensor Empty(const dim_vec_t &size, MemoryFormat memory_format, 
              DLDevice device, DLDataType dtype, MemType mtype);

STensor Empty(const dim_vec_t &size, DLDevice device, DLDataType dtype,
              MemType mtype);

STensor HostEmpty(const dim_vec_t &size, DLDataType dtype, MemType mtype);

STensor EmptyStrided(const dim_vec_t &size, const dim_vec_t &stride, 
                     DLDevice device, DLDataType dtype, MemType mtype);

STensor ViewDtype(uint64_t tensor, DLDataType dtype);

STensor ViewShapeDtype(const STensor tensor, const dim_vec_t &size, DLDataType dtype);

STensor AsStrided(const STensor tensor, const dim_vec_t &size,
                  const dim_vec_t &stride, 
                  std::optional<int64_t> storage_offset);

void AsStrided_(STensor tensor, const dim_vec_t &size,
                const dim_vec_t &stride, 
                std::optional<int64_t> storage_offset);

}
}

#endif