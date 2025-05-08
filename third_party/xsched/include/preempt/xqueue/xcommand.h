#pragma once

#include <list>
#include <mutex>
#include <cstdint>
#include <condition_variable>

#include "preempt/xqueue/xtype.h"

namespace xsched::preempt
{

class XCommand : public std::enable_shared_from_this<XCommand>
{
public:
    const XCommandType kCommandType;

    XCommand(XCommandType type): kCommandType(type) {}
    virtual ~XCommand() = default;

    XCommandState GetState();
    void SetState(XCommandState state);
    void WaitUntil(XCommandState state);

    virtual void Synchronize();
    virtual void BeforeEnqueued();
    virtual void AfterCompleted();

private:
    std::mutex state_mutex_;
    std::condition_variable state_cond_var_;
    XCommandState state_ = kCommandStateCreated;
};

class XQueueSynchronizeCommand : public XCommand
{
public:
    XQueueSynchronizeCommand():
        XCommand(kCommandTypeXQueueSynchronize) {}
    virtual ~XQueueSynchronizeCommand() = default;
};

class IntervalSynchronizeCommand : public XCommand
{
public:
    IntervalSynchronizeCommand():
        XCommand(kCommandTypeIntervalSynchronize) {}
    virtual ~IntervalSynchronizeCommand() = default;
};

class XQueueDestroyCommand : public XCommand
{
public:
    XQueueDestroyCommand():
        XCommand(kCommandTypeXQueueDestroy) {}
    virtual ~XQueueDestroyCommand() = default;
};

} // namespace xsched::preempt
