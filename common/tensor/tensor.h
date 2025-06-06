#ifndef COLSERVE_TENSOR_POOL_H
#define COLSERVE_TENSOR_POOL_H

// #include <c10/core/MemoryFormat.h>
// #include <ATen/Tensor.h>

#include <common/tensor/dlpack.h>
#include <common/cuda_allocator.h>
#include <common/tensor/dtype_helper.h>

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>

namespace colserve {
namespace sta {

using dim_vec_t = std::vector<int64_t>;

enum class MemoryFormat {
  Contiguous,
};

class STensor;
class TensorContainer {
 public:
  using memory_data_t = std::shared_ptr<CUDAMemPool::PoolEntry>;

  TensorContainer();

  // null tensor container, same as normal tensor but without memory and have null flag set
  TensorContainer(std::vector<int64_t> shape, DLDevice device, DLDataType dtype);
  TensorContainer(std::vector<int64_t> shape, std::vector<int64_t> stride, 
                  DLDevice device, DLDataType dtype, size_t storage_offset);
                  
  TensorContainer(memory_data_t mdata_, std::vector<int64_t> shape, 
                  DLDevice device, DLDataType dtype);
  TensorContainer(memory_data_t mdata_, std::vector<int64_t> shape, 
                  MemoryFormat memory_format, 
                  DLDevice device, DLDataType dtype);
  TensorContainer(memory_data_t mdata_, std::vector<int64_t> shape, 
                  std::vector<int64_t> stride, 
                  DLDevice device, DLDataType dtype, 
                  size_t storage_offset);
  virtual ~TensorContainer();

  void SetTensor(TensorContainer::memory_data_t mdata, 
                 std::vector<int64_t> shape, 
                 DLDevice device, DLDataType dtype, 
                 std::optional<size_t> storage_offset);
  void SetTensor(TensorContainer::memory_data_t mdata, 
                 std::vector<int64_t> shape, 
                 std::vector<int64_t> stride, 
                 DLDevice device, DLDataType dtype, 
                 std::optional<size_t> storage_offset);

  friend STensor;
 private:
  std::vector<int64_t> shape_;
  std::vector<int64_t> stride_;
  bool is_null_{false};
  
  DLTensor tensor_;
  memory_data_t mdata_;

  size_t stensor_version_{0};
};

class STensor : public std::shared_ptr<TensorContainer> {
 public:
  STensor() : std::shared_ptr<TensorContainer>() {}
  STensor(STensor &tensor) : std::shared_ptr<TensorContainer>(tensor) {}
  STensor(const STensor &tensor) : std::shared_ptr<TensorContainer>(tensor) {}
  STensor(STensor &&tensor) : std::shared_ptr<TensorContainer>(std::move(tensor)) {}
  template<typename... Args>
  STensor(Args&&... args) : 
      std::shared_ptr<TensorContainer>(
        std::make_shared<TensorContainer>(std::forward<Args>(args)...)
      ) {}

  TensorContainer::memory_data_t MData() const {
    return get()->mdata_;
  }
  // std::vector<int64_t> ShapeVec() const {
  //   return get()->shape_;
  // }
  // std::vector<int64_t> StrideVec() const {
  //   return get()->stride_;
  // }
  const dim_vec_t & Shape() const {
    return get()->shape_;
  }
  const dim_vec_t & Stride() const {
    return get()->stride_;
  }
  inline int64_t StorageOffset() const {
    // return get()->tensor_.byte_offset / (get()->tensor_.dtype.bits >> 3);
    return get()->tensor_.byte_offset / GetDataTypeNbytes(get()->tensor_.dtype);
  }
 
  inline void SetByteOffset(int64_t byte_offset) {
    auto & tensor = get()->tensor_;
    tensor.byte_offset = byte_offset;
    UpdateVersion();
  }
 
  inline void SetStorageOffset(int64_t storage_offset) {
    auto & tensor = get()->tensor_;
    // tensor.byte_offset = storage_offset * (tensor.dtype.bits >> 3);
    tensor.byte_offset = storage_offset * GetDataTypeNbytes(tensor.dtype);
    UpdateVersion();
  }
 
  bool IsNull() const;

  bool ComputeContiguous() const;
  size_t ComputeNumel() const;
  size_t ComputeNbytes() const;

  // void Resize(at::IntArrayRef size, at::OptionalIntArrayRef stride);
  void AllocForNull(MemType mtype);
  void SetMDataForNull(TensorContainer::memory_data_t mdata, 
                       bool check_memory_bound = false);
  void DeallocToNull();
  void DeallocToDummy();
  void Rearrange();

  inline size_t Version() const {
    return get()->stensor_version_;
  }
  inline void UpdateVersion() {
    get()->stensor_version_++;
  }

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


}
}


#endif