#include "utils/xassert.h"
#include "hal/ascend/event_pool.h"
#include "hal/ascend/acl_assert.h"
#include "hal/ascend/acl_command.h"
#include "preempt/xqueue/xqueue.h"

using namespace xsched::hal::ascend;

AclCommand::~AclCommand()
{
    if (following_event_ == nullptr) return;
    g_event_pool.Push(following_event_);
}

aclError AclCommand::EnqueueWrapper(aclrtStream stream)
{
    aclError ret = Enqueue(stream);
    if (UNLIKELY(ret != ACL_SUCCESS)) return ret;
    if (following_event_ != nullptr) {
        ret = Api::RecordEvent(following_event_, stream);
    }
    return ret;
}

void AclCommand::HalSynchronize()
{
    XASSERT(following_event_ != nullptr, "following_event_ is nullptr");
    ACL_ASSERT(Api::SynchronizeEvent(following_event_));
}

bool AclCommand::HalSynchronizable()
{
    return following_event_ != nullptr;
}

bool AclCommand::EnableHalSynchronization()
{
    following_event_ = (aclrtEvent)g_event_pool.Pop();
    return following_event_ != nullptr;
}

AclModelExecuteCommand::AclModelExecuteCommand(uint32_t model_id,
                                               const aclmdlDataset *input,
                                               aclmdlDataset *output)
    : AclCommand(preempt::kHalCommandTypeNormal)
    , model_id_(model_id), input_(input), output_(output)
{
    
}

aclError AclModelExecuteCommand::Enqueue(aclrtStream stream)
{
    return Api::ModelExecuteAsync(model_id_, input_, output_, stream);
}

AclEventRecordCommand::AclEventRecordCommand(aclrtEvent event)
    : AclCommand(preempt::kHalCommandTypeIdempotent), event_(event)
{

}

void AclEventRecordCommand::Synchronize()
{
    auto xqueue = GetXQueue();
    
    if (xqueue == nullptr) {
        HalSynchronize();
        return;
    }

    xqueue->Synchronize(
        std::static_pointer_cast<AclCommand>(shared_from_this()));
}

void AclEventRecordCommand::HalSynchronize()
{
    ACL_ASSERT(Api::SynchronizeEvent(event_));
}

aclError AclEventRecordCommand::Enqueue(aclrtStream stream)
{
    return Api::RecordEvent(event_, stream);
}
