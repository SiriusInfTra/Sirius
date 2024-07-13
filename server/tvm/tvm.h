#ifndef COLSERVE_TVM_H
#define COLSERVE_TVM_H


// #ifdef TVM_RUNTIME_LOGGING_H_
//   static_assert(false, "tvm/runtime/logging.h should be included after this file");
// #endif

#include <common/dlpack.h>

#include <dmlc/json.h>
#include <dmlc/memory_io.h>

// #include "undef_log.h"

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/logging.h>

// #ifdef GLOG_LOGGING_H 
//   static_assert(false, "glog/glog.h should be included after this file");
// #endif

// #ifdef _LOGGING_H_
//   static_assert(false, "glog/logging.h should be included after this file");
// #endif

// #include "undef_log.h"

// #include <glog/logging.h>
#endif