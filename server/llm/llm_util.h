#pragma once

#include <server/llm/llm.h>
#include <boost/python.hpp>

namespace bp = boost::python;

namespace colserve {

void CallGLOG_INFO(const char* str);
void CallGLOG_DINFO(const char* str);
void CallGLOG_INFO_WITH_FRAME(const char* str);

struct LLMRequestsConvert {
  static PyObject* convert(const std::vector<LLMRequest>& requests);
};

struct LLMRequest {
  uint64_t request_id;
  std::string prompt;
  int max_tokens;
};

struct LLMRequestMetric {
  LLMRequestMetric() = default;
  LLMRequestMetric(int num_prompt_token, int num_output_token, 
                   double queue_ms, double prefill_ms, double decode_ms)
    : num_prompt_token(num_prompt_token),
      num_output_token(num_output_token),
      queue_ms(queue_ms),
      prefill_ms(prefill_ms),
      decode_ms(decode_ms) {}

  int num_prompt_token;
  int num_output_token;

  double queue_ms;
  double prefill_ms;
  double decode_ms;
};

}