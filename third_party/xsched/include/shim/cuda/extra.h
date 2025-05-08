#include "hal/cuda/cuda.h"

#include <vector>
#include <mutex>
#include <memory>
#include <unordered_set>

namespace xsched::shim::cuda::extra {

inline const char* GetFuncName(CUfunction func) {
  return *(const char**)((uintptr_t)func + 8);
}

inline bool IsNcclFunc(CUfunction func) {
  const char* func_name = GetFuncName(func);
  if (func_name == nullptr) return false;
  std::string func_name_str(func_name);
  return func_name_str.find("ncclKernel") != std::string::npos;
}


class Nccl {
 public:
  static void Init();
  static void MaybeAddStream(CUfunction func, CUstream stream);
  static std::vector<CUstream> GetStreams();
  static void GuessNcclBegin();
  static void GuessNcclEnd();
  static bool IsGuessNcclBegined();
  static bool IsGuessingNccl();

 private:
  static std::unique_ptr<Nccl> nccl_;

  bool guessing_{false};
  std::unordered_set<CUstream> streams_;
  std::mutex mutex_;
};

}

