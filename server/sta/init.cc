// #include <glog/logging.h>

#include "init.h"

namespace colserve {
namespace sta {

void Init(size_t memory_pool_nbytes, bool master) {
  CUDAMemPool::Init(memory_pool_nbytes, master);
  TensorPool::Init();
}

}
}