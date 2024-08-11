#ifndef COLSERVE_XSCHED_CTRL_H
#define COLSERVE_XSCHED_CTRL_H

// #include <common/util.h>

#include <functional>
#include <optional>


namespace colserve {
namespace sta {
namespace xsched {

uint64_t RegisterStream(uint64_t stream);
uint64_t GetRegisteredHandle(uint64_t stream);
std::vector<uint64_t> GetRegisteredStreams();
void UnRegisterStream(uint64_t stream);
void UnRegisterAllStreams();
size_t AbortStream(uint64_t stream);
size_t AbortAllStreams();
int SyncStream(uint64_t stream);
int SyncAllStreams();
size_t GetXQueueSize(uint64_t stream);
size_t GetTotalXQueueSize();
int QueryRejectCudaCalls();
void SetRejectCudaCalls(bool reject);
int RegisterCudaKernelLaunchPreHook(std::function<void*()> fn);

void StreamApply(std::function<void(uint64_t stream)> fn);

void GuessNcclBegin();
void GuessNcclEnd();
std::vector<uint64_t> GetNcclStreams();

} // namespace xsched
} // namespace sta
} // namespace colserve


#endif