#pragma once

#include "preempt/xqueue/xcommand.h"

namespace xsched::preempt
{

class XQueue;

enum HalCommandType
{
    /// @brief An normal command is a non-cancelable and non-idempotent
    /// HalCommand. We must wait all previous cancelable HalCommands to
    // complete before we submit this HalCommand. If we did not wait,
    /// when preemption happened, all previous HalCommands will be canceled,
    /// and this will not, then this HalCommand will write system memory,
    /// leading to an inconsist memory state.
    kHalCommandTypeNormal       = 0,

    /// @brief A cancelable command should be canceled in
    /// HalQueue->CancelAllCommands(). The HAL layer should guarantee that the
    /// cancel will not cause side effects and the re-execution of the canceled
    /// HalCommand will not cause errors.
    kHalCommandTypeDeactivatable   = 1,

    /// @brief An idepotent command is a NON-CANCELABLE HalCommand but can be
    // re-executed without side effects (will not write device or host memory).
    kHalCommandTypeIdempotent   = 2,
};

class HalCommand : public XCommand
{
public:
    const HalCommandType kHalCommandType;

    HalCommand(HalCommandType hal_type)
        : XCommand(kCommandTypeHAL), kHalCommandType(hal_type) {}
    
    virtual ~HalCommand() = default;


    /// @brief Will be called before submit to HalQueue. Can do
    ///        synchronization in BeforeHalSubmit(). For example,
    ///        cuStreamWaitEvent can wait until the cuEvent is
    ///        synchronized before HalSubmit.
    /// NOTE: This function will be called without holding the
    ///       submit worker lock.
    virtual void BeforeHalSubmit();


    /// @brief Synchronize with the hardware to make sure that
    ///        the HalCommand has been executed (or killed if
    ///        called HalQueue->CancelAllCommands()). Do not
    ///        necessarily mean that the HalCommand has been
    ///        "Completed".
    virtual void HalSynchronize() = 0;


    /// @brief Check whether the HalCommand currently supports
    ///        synchronization with the hardware. For example,
    ///        CudaEventRecordCommand will return true, because
    ///        we can use cuEventWait() to sync. May return
    ///        false before EnableHalSynchronization() and true
    ///        after EnableHalSynchronization().
    /// @return Whether the HalCommand currently supports
    ///         synchronization with the hardware.
    virtual bool HalSynchronizable() = 0;


    /// @brief Enable synchronization between the HalCommand and
    ///        the hardware. For example, a CudaLaunchKernelCommand
    ///        can be enabled, because we can attach a CudaEvent
    ///        to it and synchronize the CudaEvent.
    /// @return Whether the synchronization has been enabled successfully.
    virtual bool EnableHalSynchronization() = 0;


    /// @brief Wait until the HalCommand actually become "Completed".
    virtual void Synchronize() override;


    /// @brief Get the index of the HalCommand submitted to the XQueue.
    ///        The index starts from 1, available AFTER the HalCommand
    ///        is submitted to the XQueue.
    /// @return The index of the HalCommand.
    int64_t GetIdx() const { return idx_; }
    
    void SetIdx(int64_t idx) { idx_ = idx; }

    /// @brief Get the internal XQueue pointer.
    /// @return The pointer to the XQueue.
    std::shared_ptr<XQueue> GetXQueue() const { return xqueue_; }


    /// @brief Set the internal XQueue pointer.
    ///        Should be called when it is submitted to XQueue.
    /// @param xqueue The pointer to the XQueue.
    void SetXQueue(std::shared_ptr<XQueue> xqueue) { xqueue_ = xqueue; }

private:
    int64_t idx_ = -1;
    std::shared_ptr<XQueue> xqueue_ = nullptr;
};

} // namespace xsched::preempt
