#include <c10/core/DispatchKeySet.h>

#include "override_ops.h"

namespace torch_col {

bool _has_compatible_shallow_copy_type(const at::Tensor & self, const at::Tensor & from) {
  if (!(self.key_set().has(c10::DispatchKey::PrivateUse1) 
      || from .key_set().has(c10::DispatchKey::PrivateUse1))
  ) {
    return self.unsafeGetTensorImpl()->has_compatible_shallow_copy_type(
        from.key_set());
  } else if (self.key_set().has(c10::DispatchKey::PrivateUse1) 
      && from.key_set().has(c10::DispatchKey::PrivateUse1)
  ) {
    return true;
  } else {
    return false;
  }
}


}