#ifndef COLSERVE_SHAPE_HELPER_H
#define COLSERVE_SHAPE_HELPER_H

#include <dlpack/dlpack.h>
#include <vector>
#include <glog/logging.h>

#include "tensor_pool.h"

namespace colserve {
namespace sta {

size_t ComputeStorageNbytes(const std::vector<int64_t> &size,
                            DLDataType dtype);

size_t ComputeStorageNbytes(const std::vector<int64_t> &size, 
                            const std::vector<int64_t> &stride, 
                            DLDataType dtype,
                            size_t storage_offset = 0);

inline void CheckMemoryBound(const std::vector<int64_t> &size,
                             const std::vector<int64_t> &stride,
                             DLDataType dtype,
                             size_t storage_offset,
                             const TensorContainer::memory_data_t mdata) {
  size_t nbytes = ComputeStorageNbytes(size, stride, dtype, storage_offset);
  CHECK_LE(nbytes, mdata->size) << "Out of memory bound";
}

}
}

#endif