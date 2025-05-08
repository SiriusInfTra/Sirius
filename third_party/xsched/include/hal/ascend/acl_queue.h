#pragma once

#include "hal/ascend/acl.h"
#include "preempt/xqueue/xtype.h"
#include "preempt/hal/hal_queue.h"

namespace xsched::hal::ascend
{

class AclQueue : public preempt::HalQueue
{
public:
    AclQueue(preempt::XPreemptMode mode, aclrtStream stream);
    virtual ~AclQueue() = default;

    aclrtStream GetAclStream() const { return stream_; }
    
    virtual void OnInitialize() override;
    virtual void HalSynchronize() override;
    virtual void HalSubmit(
        std::shared_ptr<preempt::HalCommand> hal_command) override;

private:
    const preempt::XPreemptMode mode_;
    const aclrtStream stream_;
    aclrtContext context_ = nullptr;
};

} // namespace xsched::hal::ascend
