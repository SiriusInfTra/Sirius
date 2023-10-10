#include "dtype_helper.h"

namespace colserve {
namespace sta {

std::ostream & operator<<(std::ostream &os, const DLDataType &dtype) {
  os << "[";
  switch (dtype.code)
  {
  case kDLInt:
    os << "kDLInt";
    break;
  case kDLUInt:
    os << "kDLUInt";
    break;
  case kDLFloat:
    os << "kDLFloat";
    break;
  case kDLOpaqueHandle:
    os << "kDLOpaqueHandle";
    break;
  case kDLBfloat:
    os << "kDLBfloat";
    break;
  case kDLComplex:
    os << "kDLComplex";
    break;
  case kDLBool:
    os << "kDLBool";
    break;
  default:
    CHECK(false) << "known DLDataType code " << dtype.code;
  }
  os << ":" << dtype.bits << ":" << dtype.lanes << "]";
  return os;
}

}
}