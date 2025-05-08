#include "shim/cuda/extra.h"
#include "utils/log.h"

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <string>

namespace xsched::shim::cuda::extra {

std::unique_ptr<Nccl> Nccl::nccl_ = nullptr;

void Nccl::Init() {
  if (nccl_ == nullptr) {
    nccl_ = std::make_unique<Nccl>();
  }
}

void Nccl::MaybeAddStream(CUfunction func, CUstream stream) {
  const char* func_name = GetFuncName(func);
  if (func_name == nullptr) return;
  std::string func_name_str(func_name);
  if (func_name_str.find("ncclKernel") != std::string::npos) {
    XDEBG("find nccl stream %p call %s", stream, func_name);
    std::lock_guard<std::mutex> lock(nccl_->mutex_);
    nccl_->streams_.insert(stream);
  }
}

std::vector<CUstream> Nccl::GetStreams() {
  std::lock_guard<std::mutex> lock(nccl_->mutex_);
  return std::vector<CUstream>{nccl_->streams_.begin(), 
                               nccl_->streams_.end()};
}

void Nccl::GuessNcclBegin() {
  std::lock_guard<std::mutex> lock(nccl_->mutex_);
  nccl_->guessing_ = true;
}

void Nccl::GuessNcclEnd() {
  std::lock_guard<std::mutex> lock(nccl_->mutex_);
  nccl_->guessing_ = false;
}

bool Nccl::IsGuessNcclBegined() {
  std::lock_guard<std::mutex> lock(nccl_->mutex_);
  return nccl_->guessing_;
}

bool Nccl::IsGuessingNccl() {
  if (nccl_ == nullptr) return false;

  std::lock_guard<std::mutex> lock(nccl_->mutex_);
  return nccl_->guessing_;
}

}

