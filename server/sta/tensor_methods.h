#ifndef COLSERVE_TENSOR_METHODS_H
#define COLSERVE_TENSOR_METHODS_H

#include <dlpack/dlpack.h>

#include <iostream>
#include <vector>

namespace colserve {
namespace sta {

uint64_t Empty(const std::vector<int64_t> &shape, DLDataType dtype);

uint64_t AsStrided(uint64_t handle, const std::vector<int64_t> &size, 
                   const std::vector<int64_t> &stride, int64_t storage_offset);



}
}

#endif