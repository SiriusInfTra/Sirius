#include "tensor_methods.h"
#include "tensor_pool.h"
#include "shape_helper.h"

namespace colserve {
namespace sta {

uint64_t Empty(at::IntArrayRef shape, DLDataType dtype) {
  LOG(INFO) << shape;
  auto storage_nbytes = ComputeStorageNbytes(shape, dtype);
  auto entry = CUDAMemPool::Get()->Alloc(storage_nbytes);
  if (entry == nullptr && storage_nbytes != 0) {
    LOG(FATAL) << "Tensor Method Empty: Out of memory, required " << storage_nbytes << " bytes";
    return 0;
  } else if (entry == nullptr) {
    LOG(WARNING) << "Tensor Method Empty: tensor without memory";
  }
  return TensorPool::Get()->Insert(STensor(entry, shape.vec(), dtype));
}

uint64_t AsStrided(uint64_t handle, at::IntArrayRef size,
                   at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  std::cout << "Astrided " << size << " " << stride << " " <<  storage_offset.value_or(0) << std::endl;
  auto tensor = TensorPool::Get()->Tensor(handle);
  CheckMemoryBound(size, stride, tensor->dtype, storage_offset.value_or(0), tensor.MData());
  return TensorPool::Get()->Insert(STensor(tensor.MData(), size.vec(), stride.vec(), tensor->dtype, storage_offset.value_or(0)));
}

void STensor::Resize(at::IntArrayRef size, at::OptionalIntArrayRef stride) {
  // std::cout << "resize tensor " << size << std::endl;
  bool same_size = true;
  if (size.size() == get()->tensor_.ndim) {
    for (size_t i = 0; i < size.size(); i++) {
      if (size[i] != get()->tensor_.shape[i]) {
        same_size = false;
        break;
      }
    }
  }
  if (same_size) {
    return;
  }

  size_t storage_nbytes;
  if (stride.has_value()) {
    storage_nbytes = ComputeStorageNbytes(size, stride.value(), get()->tensor_.dtype) 
        + get()->tensor_.byte_offset;
  } else {
    storage_nbytes = ComputeStorageNbytes(size, get()->tensor_.dtype) 
        + get()->tensor_.byte_offset;
  }

  // std::cout << "storage_nbytes: " << storage_nbytes << std::endl;
  TensorContainer::memory_data_t mdata = MData();
  if (mdata == nullptr || mdata->size < storage_nbytes) {
    auto new_mdata = CUDAMemPool::Get()->Resize(mdata, storage_nbytes);
    CUDAMemPool::Get()->CopyFromTo(mdata, new_mdata);
    mdata = new_mdata;
  }
  // if (mdata) {
  //   std::cout << " mdata:" << mdata->addr << " " << mdata->size << std::endl;
  // }

  if (stride.has_value()) {
    get()->SetTensor(mdata, size.vec(), stride.value().vec(), 
        get()->tensor_.dtype, std::nullopt);
  } else {
    get()->SetTensor(mdata, size.vec(), get()->tensor_.dtype, std::nullopt);
  }
}

}
}