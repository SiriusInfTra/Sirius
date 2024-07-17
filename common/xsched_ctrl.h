#ifndef COLSERVE_XSCHED_CTRL_H
#define COLSERVE_XSCHED_CTRL_H

#include <functional>


namespace colserve {
namespace sta {
namespace xsched {

uint64_t RegisterStream(uint64_t stream);
uint64_t GetRegisteredGlobalHandle();
uint64_t GetRegisteredGlobalStream();
void UnRegisterStream();
size_t AbortStream();
bool SyncStream();
size_t GetXQueueSize();
bool QueryRejectCudaCalls();
void SetRejectCudaCalls(bool reject);
bool RegisterCudaKernelLaunchPreHook(std::function<void*()> fn);

} // namespace xsched
} // namespace sta
} // namespace colserve


#endif