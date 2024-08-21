#ifndef COLSERVE_DTYPE_HELPER_H
#define COLSERVE_DTYPE_HELPER_H

#include <iostream>

#include "dlpack.h"

namespace colserve {
namespace sta {

inline size_t GetDataTypeNbytes(const DLDataType &dtype) {
  return ((dtype.bits + 7) >> 3);
}

std::ostream & operator<<(std::ostream &os, const DLDataType &dtype);

bool operator==(const DLDataType &dtype1, const DLDataType &dtype2);

inline bool DLDataTypeEqual(const DLDataType &dtype1, const DLDataType &dtype2) {
  return dtype1 == dtype2;
}

constexpr DLDataType DLFloat32 = DLDataType{kDLFloat, 32, 1};
constexpr DLDataType DLFloat64 = DLDataType{kDLFloat, 64, 1};
constexpr DLDataType DLInt8 = DLDataType{kDLInt, 8, 1};
constexpr DLDataType DLInt32 = DLDataType{kDLInt, 32, 1};
constexpr DLDataType DLInt64 = DLDataType{kDLInt, 64, 1};

}
}

#endif