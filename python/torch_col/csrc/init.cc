#include <glog/logging.h>
#include <sta/init.h>

namespace torch_col {

struct Initializer {
  Initializer() {
    colserve::sta::Init();
    ::google::InitGoogleLogging("torch_col");
  }
};

static Initializer initializer __attribute__ ((init_priority (101)));

}