#include "logging_as_glog.h"
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/logging.h>
#include "cache.h" 

namespace colserve {

std::unique_ptr<colserve::GraphCache> colserve::GraphCache::instance_ = nullptr;

void colserve::GraphCache::Init(size_t nbytes) {
  // instance_ = std::make_unique<GraphCache>(nbytes);
  // LOG(INFO) << "Init GraphCache, nbytes " << nbytes;
  LOG(INFO) << "GraphCache deprecated";
}

colserve::GraphCache* colserve::GraphCache::Get() {
  LOG(FATAL) << "GraphCache deprecated";
  // return instance_.get();
  return nullptr;
}

}