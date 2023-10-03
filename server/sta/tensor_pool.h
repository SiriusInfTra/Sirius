#ifndef COLSERVE_TENSOR_POOL_H
#define COLSERVE_TENSOR_POOL_H

#include <memory>
#include <unordered_map>
#include <dlpack/dlpack.h>
#include <vector>

#include "cuda_allocator.h"

namespace colserve {
namespace sta {

class STensor;
class TensorContainer {
 public:
  TensorContainer();
  TensorContainer(const CUDAMemPool::PoolEntry &mdata_, std::vector<int64_t> shape, DLDataType dtype);
  virtual ~TensorContainer();

  friend STensor;
 private:
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
  
  DLTensor tensor_;
  CUDAMemPool::PoolEntry mdata_;
};

class STensor : public std::shared_ptr<TensorContainer> {
 public:
  STensor() : std::shared_ptr<TensorContainer>() {}
  STensor(STensor &tensor) : std::shared_ptr<TensorContainer>(tensor) {}
  STensor(const STensor &tensor) : std::shared_ptr<TensorContainer>(tensor) {}
  STensor(STensor &&tensor) : std::shared_ptr<TensorContainer>(std::move(tensor)) {}
  template<typename... Args>
  STensor(Args&&... args) : 
      std::shared_ptr<TensorContainer>(std::make_shared<TensorContainer>(std::forward<Args>(args)...)) {}
  STensor& operator=(STensor &tensor) {
    std::shared_ptr<TensorContainer>::operator=(tensor);
    return *this;
  }
  STensor& operator=(const STensor &tensor) {
    std::shared_ptr<TensorContainer>::operator=(tensor);
    return *this;
  }
  STensor& operator=(STensor &&tensor) {
    std::shared_ptr<TensorContainer>::operator=(std::move(tensor));
    return *this;
  }
  const DLTensor* operator->() const {
    return &get()->tensor_;
  }
};


class TensorPool {
 public:
  static void Init();
  static TensorPool* Get();

  TensorPool();
  uint64_t Empty(std::vector<int64_t> shape, DLDataType dtype);
  void Free(uint64_t handle);
  STensor Tensor(uint64_t handle);
  const STensor CTensor(uint64_t handle) const;
  
 private:
  static std::unique_ptr<TensorPool> tensor_pool_;

  std::mutex mutex_;
  std::atomic<uint64_t> handle_counter_;
  std::unordered_map<uint64_t, STensor> tensor_by_handle_;
  
};

}
}


#endif