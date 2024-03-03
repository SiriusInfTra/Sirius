// #include <glog/logging.h>

#include "init.h"
#include "sta/mempool.h"

namespace colserve {
namespace sta {

void Init(size_t memory_pool_nbytes, bool cleanup, bool observe, FreeListPolicyType free_list_policy) {
  CUDAMemPool::Init(memory_pool_nbytes, cleanup, observe, free_list_policy);
}

}
}