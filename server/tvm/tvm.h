#ifndef COLSERVE_TVM_H
#define COLSERVE_TVM_H

#ifdef GLOG_LOGGING_H
  static_assert(false, "glog/glog.h should be included after this file");
#endif


// #include <dlpack/dlpack.h>
#include "../sta/dlpack.h"
#include <dmlc/json.h>
#include <dmlc/memory_io.h>

#include "../undef_log.h"

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include "../undef_log.h"

#include <glog/logging.h>

#endif