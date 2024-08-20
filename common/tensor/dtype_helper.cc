#include "dtype_helper.h"

#include <common/log_as_glog_sta.h>

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

bool operator==(const DLDataType &dtype1, const DLDataType &dtype2) {
  return dtype1.code == dtype2.code && dtype1.bits == dtype2.bits && dtype1.lanes == dtype2.lanes;
}

}
}