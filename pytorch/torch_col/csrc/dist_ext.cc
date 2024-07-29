#include <torch_col/csrc/dist_ext.h>

namespace torch_col {

void Reducer::finalize_dropped_batch() {
  // ::c10d::Reducer::finalize_backward();
  if (expect_autograd_hooks_) {
    ::c10d::Reducer::finalize_backward();
  } else {
    require_finalize_ = false;
  }
  
}

}