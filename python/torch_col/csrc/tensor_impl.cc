#include "tensor_impl.h"
#include "dlpack_convert.h"

namespace torch_col {
  
ColTensorImpl::ColTensorImpl(std::shared_ptr<Data> data)
    : c10::TensorImpl(c10::DispatchKeySet{c10::DispatchKey::PrivateUse1,
                                          c10::DispatchKey::AutogradPrivateUse1},
                      GetTypeMeta(data),
                      c10::DeviceType::CUDA),
      data_(data) {
  
}

caffe2::TypeMeta ColTensorImpl::GetTypeMeta(const std::shared_ptr<Data> &data) {
  auto tensor = colserve::sta::TensorPool::Get()->Tensor(data->handle);
  return getCaffeTypeMeta(tensor->dtype);
}

}