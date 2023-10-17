#ifndef COLSERVE_TENSOR_POOL_H
#define COLSERVE_TENSOR_POOL_H

#include <memory>
#include <unordered_map>
// #include <dlpack/dlpack.h>
#include <vector>
#include <optional>
#include <ATen/Tensor.h>


#include "dlpack.h"
#include "cuda_allocator.h"

namespace colserve {
namespace sta {

using handle_t = uint64_t;

class STensor;
class TensorContainer {
 public:
  using memory_data_t = std::shared_ptr<CUDAMemPool::PoolEntry>;

  TensorContainer();

  // null tensor container, same as normal tensor but without memory and have null flag set
  TensorContainer(std::vector<int64_t> shape, DLDataType dtype);
  TensorContainer(std::vector<int64_t> shape, std::vector<int64_t> stride, 
                  DLDataType dtype, size_t storage_offset);
  
  TensorContainer(memory_data_t mdata_, std::vector<int64_t> shape, DLDataType dtype);
  TensorContainer(memory_data_t mdata_, std::vector<int64_t> shape, std::vector<int64_t> stride, 
                  DLDataType dtype, size_t storage_offset);
  virtual ~TensorContainer();

  void SetTensor(TensorContainer::memory_data_t mdata, std::vector<int64_t> shape, 
                 DLDataType dtype, std::optional<size_t> storage_offset);
  void SetTensor(TensorContainer::memory_data_t mdata, std::vector<int64_t> shape, 
                 std::vector<int64_t> stride, 
                 DLDataType dtype, std::optional<size_t> storage_offset);

  friend STensor;
 private:
  std::vector<int64_t> shape_;
  std::vector<int64_t> stride_;
  bool is_null_{false};
  
  DLTensor tensor_;
  memory_data_t mdata_;
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

  TensorContainer::memory_data_t MData() {
    return get()->mdata_;
  }
  std::vector<int64_t> ShapeVec() const {
    return get()->shape_;
  }
  std::vector<int64_t> StrideVec() const {
    return get()->stride_;
  }
  at::IntArrayRef Shape() const {
    return at::IntArrayRef(get()->shape_);
  }
  at::IntArrayRef Stride() const {
    return at::IntArrayRef(get()->stride_);
  }
  inline int64_t StorageOffset() const {
    return get()->tensor_.byte_offset / (get()->tensor_.dtype.bits >> 3);
  }
  bool IsNull() const {
    return get()->is_null_;
  }

  bool ComputeContiguous() const;
  size_t ComputeNumel() const;

  void Resize(at::IntArrayRef size, at::OptionalIntArrayRef stride);
  void AllocForNull(bool raw_alloc);
  void AssignMDataForNull(TensorContainer::memory_data_t mdata, bool check_memory_bound = false);
  void DeallocToNull();

  DLTensor* MutableDLTensor() {
    return &get()->tensor_;
  }

  STensor& operator=(STensor &tensor) {
    std::shared_ptr<TensorContainer>::operator=(tensor);
    return *this;
  }
  STensor& operator=(const STensor &tensor) {
    std::shared_ptr<TensorContainer>::operator=(tensor);
    return *this;
  }
  STensor& operator=(STensor &&tensor) noexcept {
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
  // uint64_t Empty(std::vector<int64_t> shape, DLDataType dtype);
  uint64_t Insert(STensor tensor);
  void Remove(uint64_t handle);

  STensor Tensor(uint64_t handle) const;
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