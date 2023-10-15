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
      // TORCH_CHECK(false, "Bool type is not supported by dlpack");
      dtype.code = DLDataTypeCode::kDLBool;
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

DLDataType getDLDataType(const caffe2::TypeMeta &meta_type) {
  auto scalar_type = c10::typeMetaToScalarType(meta_type);
  return getDLDataType(scalar_type);
}

c10::ScalarType toScalarType(const DLDataType& dtype) {
  c10::ScalarType stype;
  TORCH_CHECK(dtype.lanes == 1, "ATen does not support lanes != 1");
  switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:
      switch (dtype.bits) {
        case 8:
          stype = c10::ScalarType::Byte;
          break;
        default:
          TORCH_CHECK(
              false, "Unsupported kUInt bits " + c10::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLInt:
      switch (dtype.bits) {
        case 8:
          stype = c10::ScalarType::Char;
          break;
        case 16:
          stype = c10::ScalarType::Short;
          break;
        case 32:
          stype = c10::ScalarType::Int;
          break;
        case 64:
          stype = c10::ScalarType::Long;
          break;
        default:
          TORCH_CHECK(
              false, "Unsupported kInt bits " + c10::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat:
      switch (dtype.bits) {
        case 16:
          stype = c10::ScalarType::Half;
          break;
        case 32:
          stype = c10::ScalarType::Float;
          break;
        case 64:
          stype = c10::ScalarType::Double;
          break;
        default:
          TORCH_CHECK(
              false, "Unsupported kFloat bits " + c10::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLBfloat:
      switch (dtype.bits) {
        case 16:
          stype = c10::ScalarType::BFloat16;
          break;
        default:
          TORCH_CHECK(
              false, "Unsupported kFloat bits " + c10::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLComplex:
      switch (dtype.bits) {
        case 32:
          stype = c10::ScalarType::ComplexHalf;
          break;
        case 64:
          stype = c10::ScalarType::ComplexFloat;
          break;
        case 128:
          stype = c10::ScalarType::ComplexDouble;
          break;
        default:
          TORCH_CHECK(
              false, "Unsupported kFloat bits " + c10::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLBool:
      switch (dtype.bits) {
        case 8:
          stype = c10::ScalarType::Bool;
          break;
        default:
          TORCH_CHECK(
              false, "Unsupported kDLBool bits " + c10::to_string(dtype.bits));
      }
      break;
    default:
      TORCH_CHECK(
          false, "Unsupported code " + c10::to_string(dtype.code));
  }
  return stype;
}

caffe2::TypeMeta getCaffeTypeMeta(const DLDataType &dtype) {
  auto stype = toScalarType(dtype);
  return c10::scalarTypeToTypeMeta(stype);
}

}