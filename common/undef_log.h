#ifdef GLOG_LOGGING_H 
  #define GLOG_LOGGING_H
  #error "glog/glog.h should be included after this file"
#endif

#ifdef _LOGGING_H_
  #define _LOGGING_H_
  #error "glog/logging.h should be included after this file"
#endif

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

#undef VLOG_IS_ON
#undef CHECK_OP
#undef DCHECK_NOTNULL
#undef VLOG_IF
