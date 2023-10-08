#ifndef COLSERVE_SHAPE_HELPER_H
#define COLSERVE_SHAPE_HELPER_H

// #include <dlpack/dlpack.h>
#include <ATen/Tensor.h>
#include "../undef_log.h"
#include <glog/logging.h>

#include <vector>

#include "dlpack.h"
#include "tensor_pool.h"

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
  CHECK_LE(nbytes, mdata->size) << "Out of memory bound";
  
}

}
}

#endif