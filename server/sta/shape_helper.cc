#include "shape_helper.h"


namespace colserve {
namespace sta {

size_t ComputeStorageNbytes(const std::vector<int64_t> &size,
                            DLDataType dtype) {
  size_t result = 1;
  for (auto s : size) {
    if (s == 0) return 0;
    result *= s;
  }
  return result * (dtype.bits / 8);
}

size_t ComputeStorageNbytes(const std::vector<int64_t> &size, 
                            const std::vector<int64_t> &stride, 
                            DLDataType dtype,
                            size_t storage_offset) {
  size_t result = storage_offset;
  for (size_t i = 0; i < size.size(); i++) {
    if (size[i] == 0) return 0;

    size_t stride_size = stride[i] * (size[i] - 1);
    result += stride_size;
  }
  return result * (dtype.bits / 8);
}

}
}