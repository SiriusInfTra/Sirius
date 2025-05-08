#pragma once

#include <list>
#include <thread>
#include <memory>
#include <cstdint>
#include <condition_variable>

#include "utils/lock.h"
#include "preempt/hal/hal_queue.h"
#include "preempt/xqueue/xtype.h"
#include "preempt/xqueue/wait_queue.h"

namespace xsched::preempt
{

enum WorkerState
{
    kWorkerStateRunning    = 0,
    kWorkerStatePaused     = 1,
    kWorkerStateTerminated = 2,
};

class Worker
{
public:
    Worker(std::shared_ptr<HalQueue> hal_queue,
           std::shared_ptr<WaitQueue> wait_queue,
           XPreemptMode mode,
           int64_t queue_length,
           int64_t sync_interval);
    ~Worker();

    /// @brief Pause the submit worker.
    void Pause();

    /// @brief Resume the submit worker.
    void Resume();

    /// @brief Resume the submit worker and drop all HalCommands.
    /// @param drop_idx Index of the last HalCommand that needs to be dropped.
    void ResumeAndDrop(int64_t drop_idx);

    const CommandLog &GetCommandLog() const { return command_log_; }

    /// @brief Synchronize the Worker, will return until
    /// all HalCommands submitted to Worker are completed.
    void SynchronizeQueue();


    /// @brief Synchronize the HalCommand, will return until the
    /// HalCommand is completed. The HalCommand MUST be HalSynchronizable
    /// and its state MUST have turned to kCommandStateHalSubmmited.
    /// @param hal_command The HalSynchronizable HalCommand to sync.
    void SynchronizeCommand(std::shared_ptr<HalCommand> hal_command);

    void RegisterAfterHalSubmitHook(std::function<void(std::shared_ptr<HalCommand>)> fn);

private:
    const XPreemptMode kPreemptMode;
    const int64_t kSyncInterval;
    const int64_t kQueueLength;

    const std::shared_ptr<HalQueue> hal_queue_;
    const std::shared_ptr<WaitQueue> wait_queue_;

    CommandLog command_log_;
    CommandLog sync_command_log_;
    int64_t last_synchronizable_idx_ = -1;

    std::condition_variable_any cond_var_;
    std::unique_ptr<utils::MutexLock> mutex_ = nullptr;

    int64_t drop_idx_ = -1;
    int64_t pause_count_ = 0;
    WorkerState state_ = kWorkerStateRunning;
    std::unique_ptr<std::thread> worker_thread_ = nullptr;

    std::function<void(std::shared_ptr<HalCommand>)>
        after_hal_submit_hook_ = nullptr;

    void WorkerLoop();

    void SubmitHalCommand(std::shared_ptr<HalCommand> hal_command);

    std::unique_lock<utils::MutexLock>
    SynchronizeQueueInternal(std::unique_lock<utils::MutexLock> lock);

    std::unique_lock<utils::MutexLock>
    SynchronizeCommandInternal(std::shared_ptr<HalCommand> hal_command,
                               std::unique_lock<utils::MutexLock> lock);
};

} // namespace xsched::preempt
