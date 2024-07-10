// #include <glog/logging.h>

#include "init.h"

namespace colserve {
namespace sta {

bool allocate_tensor_from_memory_pool = false;

void InitMemoryPool(size_t memory_pool_nbytes, bool cleanup, bool observe, FreeListPolicyType free_list_policy) {
  CUDAMemPool::Init(memory_pool_nbytes, cleanup, observe, free_list_policy);
  allocate_tensor_from_memory_pool = true;
}

}
}