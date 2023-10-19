// #include <glog/logging.h>

#include "init.h"

namespace colserve {
namespace sta {

void Init(bool master) {
  CUDAMemPool::Init(6ULL << 30, master); // 12 Gb
  TensorPool::Init();
}

}
}