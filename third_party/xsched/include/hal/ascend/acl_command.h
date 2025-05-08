#pragma once

#include <queue>
#include <mutex>

#include "hal/ascend/api.h"
#include "preempt/hal/def.h"
#include "preempt/hal/hal_command.h"

namespace xsched::hal::ascend
{

class AclCommand : public preempt::HalCommand
{
public:
    AclCommand(preempt::HalCommandType hal_type): HalCommand(hal_type) {}
    virtual ~AclCommand();

    aclError EnqueueWrapper(aclrtStream stream);

    virtual void HalSynchronize() override;
    virtual bool HalSynchronizable() override;
    virtual bool EnableHalSynchronization() override;

private:
    aclrtEvent following_event_ = nullptr;

    virtual aclError Enqueue(aclrtStream stream) = 0;
};

class AclModelExecuteCommand : public AclCommand
{
public:
    AclModelExecuteCommand(uint32_t model_id,
                           const aclmdlDataset *input,
                           aclmdlDataset *output);
    virtual ~AclModelExecuteCommand() = default;

private:
    const uint32_t model_id_;
    const aclmdlDataset *input_;
    aclmdlDataset *output_;

    virtual aclError Enqueue(aclrtStream stream) override;
};

class AclMemoryCommand : public AclCommand
{
public:
    AclMemoryCommand(): AclCommand(preempt::kHalCommandTypeNormal) {}
    virtual ~AclMemoryCommand() = default;
};

DEFINE_HAL_COMMAND5(AclMemcpyCommand, AclMemoryCommand,
                    aclError, aclrtStream, false,
                    Api::MemcpyAsync,
                    void *         , dst     , false,
                    size_t         , dest_max, false,
                    const void *   , src     , false,
                    size_t         , count   , false,
                    aclrtMemcpyKind, kind    , false);

DEFINE_HAL_COMMAND4(AclMemsetCommand, AclMemoryCommand,
                    aclError, aclrtStream, false,
                    Api::MemsetAsync,
                    void * , dev_ptr  , false,
                    size_t , max_count, false,
                    int32_t, value    , false,
                    size_t , count    , false);

class AclEventRecordCommand : public AclCommand
{
public:
    AclEventRecordCommand(aclrtEvent event);
    virtual ~AclEventRecordCommand() = default;

    virtual void Synchronize() override;
    virtual void HalSynchronize() override;
    virtual bool HalSynchronizable() override { return true; }
    virtual bool EnableHalSynchronization() override { return true; }

private:
    aclrtEvent event_;

    virtual aclError Enqueue(aclrtStream stream) override;
};

} // namespace xsched::hal::ascend
