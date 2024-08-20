#include <common/tensor/shape_helper.h>


namespace colserve {
namespace sta {

size_t ComputeStorageNbytes(at::IntArrayRef size,
                            DLDataType dtype,
                            size_t storage_offset) {
  size_t result = 1;
  for (auto s : size) {
    if (s == 0) return 0;
    result *= s;
  }
  // return (result + storage_offset) * (dtype.bits >> 3);
  return (result + storage_offset) * GetDataTypeNbytes(dtype);
}

size_t ComputeStorageNbytes(at::IntArrayRef size, 
                            at::IntArrayRef stride, 
                            DLDataType dtype,
                            size_t storage_offset) {
  size_t result = storage_offset + 1;
  for (size_t i = 0; i < size.size(); i++) {
    if (size[i] == 0) return 0;

    size_t stride_size = stride[i] * (size[i] - 1);
    result += stride_size;
  }
  // return result * (dtype.bits >> 3);
  return result * GetDataTypeNbytes(dtype);
}

}
}