#include "hal/ascend/api.h"
#include "hal/ascend/acl_queue.h"
#include "hal/ascend/acl_assert.h"
#include "hal/ascend/acl_command.h"

thread_local static bool acl_initialized = false;

using namespace xsched::hal::ascend;

AclQueue::AclQueue(preempt::XPreemptMode mode, aclrtStream stream)
    : mode_(mode)
    , stream_(stream)
{
    XASSERT(mode_ <= preempt::kPreemptModeStopSubmission,
            "AclQueue only support StopSubmission mode");
    ACL_ASSERT(Api::GetCurrentContext(&context_));
}

void AclQueue::OnInitialize()
{
    if (acl_initialized) return;
    acl_initialized = true;
    ACL_ASSERT(Api::SetCurrentContext(context_));
}

void AclQueue::HalSynchronize()
{
    OnInitialize();
    ACL_ASSERT(Api::SynchronizeStream(stream_));
}

void AclQueue::HalSubmit(std::shared_ptr<preempt::HalCommand> hal_command)
{
    auto acl_command = std::dynamic_pointer_cast<AclCommand>(hal_command);
    XASSERT(acl_command != nullptr, "hal_command is not an AclCommand");
    ACL_ASSERT(acl_command->EnqueueWrapper(stream_));
}
