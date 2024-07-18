#ifndef COLSERVE_XSCHED_CTRL_H
#define COLSERVE_XSCHED_CTRL_H

#include <functional>
#include <optional>


namespace colserve {
namespace sta {
namespace xsched {

uint64_t RegisterStream(uint64_t stream);
uint64_t GetRegisteredGlobalHandle();
uint64_t GetRegisteredGlobalStream();
void UnRegisterStream();
size_t AbortStream();
int SyncStream();
size_t GetXQueueSize(std::optional<uint64_t> stream);
int QueryRejectCudaCalls();
void SetRejectCudaCalls(int reject);
int RegisterCudaKernelLaunchPreHook(std::function<void*()> fn);

} // namespace xsched
} // namespace sta
} // namespace colserve


#endif