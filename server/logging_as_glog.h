#ifndef COLSERVE_LOGGING_AS_GLOG_H
#define COLSERVE_LOGGING_AS_GLOG_H
#ifdef GLOG_LOGGING_H 
  #define GLOG_LOGGING_H
  #error "glog/glog.h should be included after this file"
#endif

#ifdef _LOGGING_H_
  #define _LOGGING_H_
  #error "glog/logging.h should be included after this file"
#endif

#ifndef NO_TVM
  #include <dmlc/logging.h>
  #include <common/undef_log.h>
  #include <tvm/runtime/logging.h>
  #include <common/undef_log.h>
#endif

#ifndef NO_TORCH
  #include <c10/util/Logging.h>
  #include <common/undef_log.h>
#endif
#include <glog/logging.h>
#endif