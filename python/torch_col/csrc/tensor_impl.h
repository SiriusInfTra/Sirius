#ifndef COLSERVE_TENSOR_H
#define COLSERVE_TENSOR_H

#include <c10/core/TensorImpl.h>
#include <ATen/TensorOptions.h>

#include <sta/tensor_pool.h>


namespace torch_col {


class ColTensorImpl : public c10::TensorImpl {
 public:
  struct Data {
    uint64_t handle;
    Data(uint64_t handle) : handle(handle) {};
    ~Data() {
      colserve::sta::TensorPool::Get()->Remove(handle);
    };
  };

  explicit ColTensorImpl(std::shared_ptr<Data> data);
  explicit ColTensorImpl(std::shared_ptr<Data> data, const at::Storage &storage);

  uint64_t Handle() const { return data_->handle; }
  colserve::sta::STensor Tensor() const;
  const colserve::sta::STensor CTensor() const;

  at::IntArrayRef sizes_custom() const override;
  at::IntArrayRef strides_custom() const override;
  int64_t dim_custom() const override;
  int64_t numel_custom() const override;
  bool is_contiguous_custom(at::MemoryFormat memory_format) const override;

  bool has_storage() const override;
  const at::Storage& storage() const override;
  int64_t storage_offset() const override;

  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;
  
  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;
  
  // update aten::TensorImpl properties with ColTensor
  void UpdateStorage();
  void UpdateSize();
  void UpdateAll();

 private:
  static caffe2::TypeMeta GetTypeMeta(const std::shared_ptr<Data> &data);

  template <typename VariableVersion>
  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach_core_custom(
      VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const;

  std::shared_ptr<Data> data_;
};


inline at::Tensor MakeColTensor(uint64_t handle) {
  return at::detail::make_tensor_base<ColTensorImpl>(std::make_shared<ColTensorImpl::Data>(handle));
}

inline at::Tensor MakeColTensorAlias(uint64_t handle, const at::Tensor& tensor) {
  return at::detail::make_tensor_base<ColTensorImpl>(
      std::make_shared<ColTensorImpl::Data>(handle), tensor.storage());
}

at::Tensor MakeColTensorEmpty(at::IntArrayRef size, const at::TensorOptions &options);
at::Tensor MakeColTensorEmpty(at::IntArrayRef size, at::ScalarType scalar_type);

}

#endif