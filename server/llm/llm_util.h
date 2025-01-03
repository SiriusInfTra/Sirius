#include <server/llm/llm.h>
#include <boost/python.hpp>

namespace colserve {

void CallGLOG_INFO(const char* str);
void CallGLOG_DINFO(const char* str);

struct LLMRequestsConvert {
  static PyObject* convert(const std::vector<LLMRequest>& requests);
};

}