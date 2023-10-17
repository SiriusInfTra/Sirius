#include <glog/logging.h>

namespace torch_col {

struct Initializer {
  Initializer() {
    ::google::InitGoogleLogging("torch_col");
  }
};

static Initializer initializer __attribute__ ((init_priority (101)));

}