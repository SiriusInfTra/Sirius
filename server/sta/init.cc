// #include <glog/logging.h>

#include "init.h"

namespace colserve {
namespace sta {

void Init() {
  CUDAMemPool::Init(12ULL << 30); // 12 Gb
  TensorPool::Init();
}

}
}