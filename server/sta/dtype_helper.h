#ifndef COLSERVE_DTYPE_HELPER_H
#define COLSERVE_DTYPE_HELPER_H

#include <iostream>
#include <glog/logging.h>

#include "dlpack.h"

namespace colserve {
namespace sta {

inline size_t GetDataTypeNbytes(const DLDataType &dtype) {
  return dtype.bits >> 3;
}

std::ostream & operator<<(std::ostream &os, const DLDataType &dtype);

}
}

#endif