#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
extern "C" {
uint64_t RegisterStream(uint64_t stream);
uint64_t GetRegisteredGlobalHandle();
uint64_t GetRegisteredGlobalStream();
void UnRegisterStream();
size_t AbortStream();
void SyncStream();
size_t GetXQueueSize();
void SetBlockCudaCalls(uint64_t handle, bool enable);
bool IsBlockCudaCalls_v2();
void SetBlockCudaCalls_v2(bool block);
void StopSubmit(bool synchronous);
void ResumeSubmit();
bool RegisterCudaKernelLaunchPreHook(std::function<void*()> fn);
}