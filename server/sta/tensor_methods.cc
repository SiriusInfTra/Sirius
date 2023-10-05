#include "tensor_methods.h"
#include "tensor_pool.h"
#include "shape_helper.h"

namespace colserve {
namespace sta {

uint64_t Empty(const std::vector<int64_t> &shape, DLDataType dtype) {
  auto entry = CUDAMemPool::Get()->Alloc(ComputeStorageNbytes(shape, dtype));
  if (entry == nullptr) {
    return 0;
  }
  return TensorPool::Get()->Insert(STensor(entry, shape, dtype));
}

uint64_t AsStrided(uint64_t handle, const std::vector<int64_t> &size, 
                   const std::vector<int64_t> &stride, int64_t storage_offset) {
  auto tensor = TensorPool::Get()->Tensor(handle);
  CheckMemoryBound(size, stride, tensor->dtype, storage_offset, tensor.MData());
  return TensorPool::Get()->Insert(STensor(tensor.MData(), size, stride, tensor->dtype, storage_offset));
}

}
}