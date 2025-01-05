#include <server/logging_as_glog.h>
#include <server/llm/llm_util.h>


namespace colserve {

void CallGLOG_INFO(const char* log_str) {
  google::LogMessage("py", 0).stream() 
      << "[LLM Server] " << log_str;
}

void CallGLOG_INFO_WITH_FRAME(const char* log_str) {
  try {
    bp::object inspect = bp::import("inspect");
    bp::object frame = inspect.attr("currentframe")();
    bp::object info = inspect.attr("getframeinfo")(frame);

    std::string filename = bp::extract<std::string>(info.attr("filename"));
    int lineno = bp::extract<int>(info.attr("lineno"));

    google::LogMessage(filename.c_str(), lineno).stream()
        << "[LLM Server] " << log_str;
  } catch (bp::error_already_set&) {
    PyErr_Print();
  }
}

void CallGLOG_DINFO(const char* str) {
#ifndef NDEBUG
  CallGLOG_INFO(str);
#endif
}

PyObject* LLMRequestsConvert::convert(
    const std::vector<LLMRequest>& requests) {
  bp::list py_requests;
  for (const auto& request : requests) {
    py_requests.append(request);
  }
  return bp::incref(py_requests.ptr());
}

}
