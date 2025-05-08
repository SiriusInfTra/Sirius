#pragma once

#include <list>
#include <memory>

#include "preempt/hal/hal_command.h"

namespace xsched::preempt
{

using CommandLog = std::list<std::shared_ptr<HalCommand>>;

class HalQueue
{
public:
    HalQueue() = default;
    virtual ~HalQueue() = default;


    /// @brief XQueue initialize event. Will be called on the
    ///        submitting thread when the thread is created. can
    ///        call platform specific APIs like cuCtxSetCurrent().
    virtual void OnInitialize();


    /// @brief HalCommand submit event. Will be called when a HalCommand
    ///        is submitted to XQueue. Can do some pre-processing here.
    /// @param hal_command The HalCommand submitted to the XQueue.
    virtual void OnSubmit(std::shared_ptr<HalCommand> hal_command);


    /// @brief Synchronize with the hardware to make sure that all
    ///        HalCommands submitted to HalQueue has been executed
    ///        (or killed if called HalQueue->CancelAllCommands()).
    ///        Do not necessarily mean that these HalCommand has been
    ///        "Completed".
    virtual void HalSynchronize() = 0;


    /// @brief Submit HalCommand to the hardware.
    /// @param hal_command The pointer to the HalCommand.
    /// NOTE: This function will be called while holding the submit
    /// worker lock, so it should not do any blocking operations
    /// like synchronizing another queue or command.
    virtual void HalSubmit(std::shared_ptr<HalCommand> hal_command) = 0;


    /// @brief Cancel all cancelable HalCommands submitted to HalQueue.
    ///        The platform layer should guarantee that the cancel
    ///        will not cause side effects and the re-execution
    ///        of the canceled HalCommands will not cause errors.
    virtual void Deactivate();


    /// @brief Resume the execution of the hardware.
    /// @return The idx of the first HalCommand that needs to be
    ///         re-submitted.
    virtual void Reactivate(const CommandLog &log);

    virtual void Interrupt();
};

} // namespace xsched::preempt
