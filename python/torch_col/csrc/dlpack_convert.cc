#include <ATen/DLConvertor.h>

#include "dlpack_convert.h"


namespace torch_col {

DLDataType getDLDataType(const at::ScalarType& scalar_type) {
  DLDataType dtype;
  caffe2::TypeMeta caffe2_type = c10::scalarTypeToTypeMeta(scalar_type);
  dtype.lanes = 1;
  dtype.bits = caffe2_type.itemsize() * 8;
  switch (scalar_type) {
    using ScalarType = c10::ScalarType;
    case ScalarType::Byte:
      dtype.code = DLDataTypeCode::kDLUInt;
      break;
    case ScalarType::Char:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case ScalarType::Double:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Float:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case ScalarType::Int:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Long:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Short:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Half:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Bool:
      TORCH_CHECK(false, "Bool type is not supported by dlpack");
      break;
    case ScalarType::ComplexHalf:
      dtype.code = DLDataTypeCode::kDLComplex;
      break;
    case ScalarType::ComplexFloat:
      dtype.code = DLDataTypeCode::kDLComplex;
      break;
    case ScalarType::ComplexDouble:
      dtype.code = DLDataTypeCode::kDLComplex;
      break;
    case ScalarType::BFloat16:
      dtype.code = DLDataTypeCode::kDLBfloat;
      break;
    case ScalarType::QInt8:
    case ScalarType::QUInt8:
    case ScalarType::QInt32:
    case ScalarType::QUInt4x2:
    case ScalarType::QUInt2x4:
      TORCH_CHECK(false, "QUInt/QInt types are not supported by dlpack");
      break;
    case ScalarType::Undefined:
      TORCH_CHECK(false, "Undefined is not a valid ScalarType");
    case ScalarType::NumOptions:
      TORCH_CHECK(false, "NumOptions is not a valid ScalarType");
  }
  return dtype;
}

caffe2::TypeMeta getCaffeTypeMeta(const DLDataType &dtype) {
  auto stype = at::toScalarType(dtype);
  return c10::scalarTypeToTypeMeta(stype);
}

}