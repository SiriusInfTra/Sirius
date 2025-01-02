#include <server/logging_as_glog.h>
#include <server/llm/llm_util.h>


namespace colserve {

void CallGLOG_INFO(const char* log_str) {
  google::LogMessage("py", 0).stream() 
      << "[LLM Server] " << log_str;
}

void CallGLOG_DINFO(const char* str) {
#ifndef NDEBUG
  CallGLOG_INFO(str);
#endif
}

}