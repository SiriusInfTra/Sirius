#ifndef COLSERVE_TVM_H
#define COLSERVE_TVM_H

#ifdef GLOG_LOGGING_H
  static_assert(false, "glog/glog.h should be included after this file");
#endif


#include <dlpack/dlpack.h>
#include <dmlc/json.h>
#include <dmlc/memory_io.h>

#undef LOG
#undef LOG_FATAL
#undef LOG_INFO
#undef LOG_ERROR
#undef LOG_WARNING
#undef CHECK
#undef CHECK_LT
#undef CHECK_GT
#undef CHECK_LE
#undef CHECK_GE
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_NOTNULL
#undef LOG_IF
#undef LOG_DFATAL
#undef DFATAL
#undef DLOG
#undef DLOG_IF
#undef VLOG
#undef LOG_IF
#undef CHECK
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LE
#undef CHECK_LT
#undef CHECK_GE
#undef CHECK_GT
#undef CHECK_NOTNULL
#undef LOG_EVERY_N
#undef DLOG
#undef DLOG_IF
#undef DCHECK
#undef DCHECK_EQ
#undef DCHECK_NE
#undef DCHECK_LE
#undef DCHECK_LT
#undef DCHECK_GE
#undef DCHECK_GT
#undef VLOG

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#undef LOG
#undef LOG_FATAL
#undef LOG_INFO
#undef LOG_ERROR
#undef LOG_WARNING
#undef CHECK
#undef CHECK_LT
#undef CHECK_GT
#undef CHECK_LE
#undef CHECK_GE
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_NOTNULL
#undef LOG_IF
#undef LOG_DFATAL
#undef DFATAL
#undef DLOG
#undef DLOG_IF
#undef VLOG
#undef LOG_IF
#undef CHECK
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LE
#undef CHECK_LT
#undef CHECK_GE
#undef CHECK_GT
#undef CHECK_NOTNULL
#undef LOG_EVERY_N
#undef DLOG
#undef DLOG_IF
#undef DCHECK
#undef DCHECK_EQ
#undef DCHECK_NE
#undef DCHECK_LE
#undef DCHECK_LT
#undef DCHECK_GE
#undef DCHECK_GT
#undef VLOG

#include <glog/logging.h>

#endif