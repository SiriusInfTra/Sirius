#include "cache.h" 

std::unique_ptr<colserve::GraphCache> colserve::GraphCache::instance_ = nullptr;

void colserve::GraphCache::Init(size_t nbytes) {
  instance_ = std::make_unique<GraphCache>(nbytes);
  LOG(INFO) << "Init GraphCache, nbytes" << nbytes;
}

colserve::GraphCache* colserve::GraphCache::Get() {
  return instance_.get();
}
