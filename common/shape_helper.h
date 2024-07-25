#ifndef COLSERVE_SHAPE_HELPER_H
#define COLSERVE_SHAPE_HELPER_H

// #include <dlpack/dlpack.h>
#include <ATen/Tensor.h>
#include "log_as_glog_sta.h"

#include <vector>
#include <cstdint>

#include "dlpack.h"
#include "tensor.h"
#include "dtype_helper.h"

namespace colserve {
namespace sta {

size_t ComputeStorageNbytes(at::IntArrayRef size,
                            DLDataType dtype,
                            size_t storage_offset = 0);

size_t ComputeStorageNbytes(at::IntArrayRef size, 
                            at::IntArrayRef stride, 
                            DLDataType dtype,
                            size_t storage_offset = 0);

inline void CheckMemoryBound(at::IntArrayRef size,
                             at::IntArrayRef stride,
                             DLDataType dtype,
                             size_t storage_offset,
                             const TensorContainer::memory_data_t mdata) {
  size_t nbytes = ComputeStorageNbytes(size, stride, dtype, storage_offset);
  CHECK(mdata != nullptr);
  CHECK_LE(nbytes, mdata->nbytes) << "CheckMemoryBound: Out of memory bound";
}

inline std::vector<int64_t> 
ComputeStridesForViewDtypeDownsize(at::IntArrayRef old_strides, int64_t size_ratio, 
    DLDataType old_dtype, DLDataType new_dtype) {
  const int64_t ndim = old_strides.size();

  CHECK_EQ(old_strides[ndim - 1], 1) << "stride[-1] must be 1 to view "
                                     << old_dtype << " as " << new_dtype;

  std::vector<int64_t> new_strides(ndim);
  for (int64_t dim_idx = 0; dim_idx < ndim - 1; dim_idx++) {
    new_strides[dim_idx] = old_strides[dim_idx] * size_ratio;
  }
  new_strides[ndim - 1] = 1;
  return new_strides;
}

inline std::vector<int64_t> 
ComputeStridesForViewDtypeUpsize(at::IntArrayRef old_strides, int64_t size_ratio, 
    DLDataType old_dtype, DLDataType new_dtype) {
  const int64_t ndim = old_strides.size();
  CHECK_EQ(old_strides[ndim - 1], 1) << "stride[-1] must be 1 to view " 
                                     << old_dtype << " as " << new_dtype;

  std::vector<int64_t> new_strides(ndim);
  for (int64_t dim_idx = 0; dim_idx < ndim - 1; dim_idx++) {
    CHECK_EQ(old_strides[dim_idx] % size_ratio, 0) 
        << "stride[" << dim_idx << "] must be divisible by " << size_ratio
        << " to view " << old_dtype << " as " << new_dtype;

    new_strides[dim_idx] = old_strides[dim_idx] / size_ratio;
  }
  new_strides[ndim - 1] = 1;
  return new_strides;
}

inline std::vector<int64_t> ComputeStrides(at::IntArrayRef size) {
  std::vector<int64_t> stride(size.size());
  if (size.size() > 0) {
    stride.rbegin()[0] = 1;
    for (int64_t i = static_cast<int64_t>(stride.size()) - 2; i >= 0; i--) {
      stride[i] = stride[i + 1] * size[i + 1];
    }
  }
  return stride;
}

}
}

#endif