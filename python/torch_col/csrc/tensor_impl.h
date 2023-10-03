#ifndef COLSERVE_TENSOR_H
#define COLSERVE_TENSOR_H

#include <c10/core/TensorImpl.h>

#include <sta/tensor_pool.h>

namespace torch_col {

class ColTensorImpl : public c10::TensorImpl {
 public:
  struct Data {
    uint64_t handle;
    Data(uint64_t handle) : handle(handle) {};
    ~Data() {
      colserve::sta::TensorPool::Get()->Free(handle);
    };
  };

  explicit ColTensorImpl(std::shared_ptr<Data> data);
  
 private:
  std::shared_ptr<Data> data_;

  static caffe2::TypeMeta GetTypeMeta(const std::shared_ptr<Data> &data);

};

}

#endif