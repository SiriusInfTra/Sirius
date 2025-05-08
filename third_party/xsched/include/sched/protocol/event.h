#pragma once

#include <memory>
#include <cstddef>

#include "utils/common.h"
#include "preempt/xqueue/xtype.h"
#include "sched/protocol/hint.h"
#include "sched/protocol/status.h"

namespace xsched::sched
{

enum EventType
{
    kEventHint           = 0,
    kEventTerminate      = 1,
    kEventProcessCreate  = 2,
    kEventProcessDestroy = 3,
    kEventXQueueCreate   = 4,
    kEventXQueueDestroy  = 5,
    kEventXQueueReady    = 6,
    kEventXQueueIdle     = 7,
    kEventStatusQuery    = 8,
};

struct EventMeta
{
    EventType type;
    PID       pid;
};

class Event
{
public:
    Event() = default;
    virtual ~Event() = default;

    /// @brief Get the data of the event. MUST start with EventMeta.
    virtual const void *Data() const = 0;
    virtual size_t      Size() const = 0;
    virtual EventType   Type() const = 0;
    virtual PID         Pid()  const = 0;

    static std::unique_ptr<const Event> CopyConstructor(const void *data);
};

class HintEvent : public Event
{
public:
    HintEvent(const void *data);
    HintEvent(std::unique_ptr<const Hint> hint);
    virtual ~HintEvent();

    virtual const void *Data() const override { return data_; }
    virtual size_t      Size() const override { return data_->size; }
    virtual EventType   Type() const override { return kEventHint; }
    virtual PID         Pid()  const override { return data_->meta.pid; }
    std::unique_ptr<const Hint> GetHint() const
    { return Hint::CopyConstructor(data_->hint_data); }

private:
    struct EventData
    {
        EventMeta meta;
        /// @brief The size of the whole EventData.
        size_t    size;
        /// @brief 64 is only a placeholder.
        /// The actual size of the hint data can be calculated by
        /// size - offsetof(EventData, hint_data).
        char      hint_data[64];
    };

    const EventData *data_ = nullptr;
};

/// @brief Terminate the worker thread of the scheduler.
class TerminateEvent : public Event
{
public:
    TerminateEvent()
        : meta_{ .type = kEventTerminate, .pid = GetProcessId() } {}
    TerminateEvent(const void *data)
        : meta_(*(const EventMeta *)data) {}
    
    virtual ~TerminateEvent() = default;

    virtual const void *Data() const override { return (void *)&meta_; }
    virtual size_t      Size() const override { return sizeof(meta_); }
    virtual EventType   Type() const override { return kEventTerminate; }
    virtual PID         Pid()  const override { return meta_.pid; }

private:
    EventMeta meta_;
};

class ProcessCreateEvent : public Event
{
public:
    ProcessCreateEvent()
        : meta_{ .type = kEventProcessCreate, .pid = GetProcessId() } {}
    ProcessCreateEvent(const void *data)
        : meta_(*(const EventMeta *)data) {}
    
    virtual ~ProcessCreateEvent() = default;

    virtual const void *Data() const override { return (void *)&meta_; }
    virtual size_t      Size() const override { return sizeof(meta_); }
    virtual EventType   Type() const override { return kEventProcessCreate; }
    virtual PID         Pid()  const override { return meta_.pid; }

private:
    EventMeta meta_;
};

class ProcessDestroyEvent : public Event
{
public:
    ProcessDestroyEvent()
        : meta_{ .type = kEventProcessDestroy, .pid = GetProcessId() } {}
    ProcessDestroyEvent(PID pid)
        : meta_{ .type = kEventProcessDestroy, .pid = pid } {}
    ProcessDestroyEvent(const void *data)
        : meta_(*(const EventMeta *)data) {}
    
    virtual ~ProcessDestroyEvent() = default;

    virtual const void *Data() const override { return (void *)&meta_; }
    virtual size_t      Size() const override { return sizeof(meta_); }
    virtual EventType   Type() const override { return kEventProcessDestroy; }
    virtual PID         Pid()  const override { return meta_.pid; }

private:
    EventMeta meta_;
};

class XQueueCreateEvent : public Event
{
public:
    XQueueCreateEvent(preempt::XQueueHandle handle, preempt::XDevice device)
        : data_{
            .meta = {
                .type = kEventXQueueCreate,
                .pid = GetProcessId()
            },
            .handle = handle,
            .device = device
        } {}
    XQueueCreateEvent(const void *data)
        : data_(*(const EventData *)data) {}
    
    virtual ~XQueueCreateEvent() = default;

    virtual const void *Data() const override { return (void *)&data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual EventType   Type() const override { return kEventXQueueCreate; }
    virtual PID         Pid()  const override { return data_.meta.pid; }

    preempt::XQueueHandle Handle() const { return data_.handle; }
    preempt::XDevice      Device() const { return data_.device; }

private:
    struct EventData
    {
        EventMeta meta;
        preempt::XQueueHandle handle;
        preempt::XDevice      device;
    };

    EventData data_;
};

class XQueueDestroyEvent : public Event
{
public:
    XQueueDestroyEvent(preempt::XQueueHandle handle)
        : data_{
            .meta = {
                .type = kEventXQueueDestroy,
                .pid = GetProcessId()
            },
            .handle = handle
        } {}
    XQueueDestroyEvent(const void *data)
        : data_(*(const EventData *)data) {}
    
    virtual ~XQueueDestroyEvent() = default;

    virtual const void *Data() const override { return (void *)&data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual EventType   Type() const override { return kEventXQueueDestroy; }
    virtual PID         Pid()  const override { return data_.meta.pid; }

    preempt::XQueueHandle Handle() const { return data_.handle; }

private:
    struct EventData
    {
        EventMeta meta;
        preempt::XQueueHandle handle;
    };

    EventData data_;
};

class XQueueReadyEvent : public Event
{
public:
    XQueueReadyEvent(preempt::XQueueHandle handle)
        : data_{
            .meta = {
                .type = kEventXQueueReady,
                .pid = GetProcessId()
            },
            .handle = handle
        } {}
    XQueueReadyEvent(const void *data)
        : data_(*(const EventData *)data) {}
    
    virtual ~XQueueReadyEvent() = default;

    virtual const void *Data() const override { return (void *)&data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual EventType   Type() const override { return kEventXQueueReady; }
    virtual PID         Pid()  const override { return data_.meta.pid; }

    preempt::XQueueHandle Handle() const { return data_.handle; }

private:
    struct EventData
    {
        EventMeta meta;
        preempt::XQueueHandle handle;
    };

    EventData data_;
};

class XQueueIdleEvent : public Event
{
public:
    XQueueIdleEvent(preempt::XQueueHandle handle)
        : data_{
            .meta = {
                .type = kEventXQueueIdle,
                .pid = GetProcessId()
            },
            .handle = handle
        } {}
    XQueueIdleEvent(const void *data)
        : data_(*(const EventData *)data) {}
    
    virtual ~XQueueIdleEvent() = default;

    virtual const void *Data() const override { return (void *)&data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual EventType   Type() const override { return kEventXQueueIdle; }
    virtual PID         Pid()  const override { return data_.meta.pid; }

    preempt::XQueueHandle Handle() const { return data_.handle; }

private:
    struct EventData
    {
        EventMeta meta;
        preempt::XQueueHandle handle;
    };

    EventData data_;
};

class StatusQueryEvent : public Event
{
public:
    StatusQueryEvent(StatusQuery *query_data)
        : data_{
            .meta = {
                .type = kEventStatusQuery,
                .pid = GetProcessId()
            },
            .query_data = query_data
        } {}
    StatusQueryEvent(const void *data)
        : data_(*(const EventData *)data) {}
    
    virtual ~StatusQueryEvent() = default;

    virtual const void *Data() const override { return (void *)&data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual EventType   Type() const override { return kEventStatusQuery; }
    virtual PID         Pid()  const override { return data_.meta.pid; }

    StatusQuery *QueryData() const;

private:
    struct EventData
    {
        EventMeta meta;
        StatusQuery *query_data;
    };

    EventData data_;
};

} // namespace xsched::sched
