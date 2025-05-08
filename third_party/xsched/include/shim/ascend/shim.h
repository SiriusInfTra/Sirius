#pragma once

#include "shim/common/def.h"
#include "shim/common/xmanager.h"
#include "shim/ascend/handle.h"
#include "hal/ascend/acl.h"
#include "hal/ascend/acl_command.h"

namespace xsched::shim::ascend
{

static inline aclError SubmitAclCommand(
    std::shared_ptr<hal::ascend::AclCommand> cmd, aclrtStream stream)
{
    preempt::XQueueHandle handle = GetXHandle(stream);
    if (XManager::Submit(cmd, handle)) return ACL_SUCCESS;
    return cmd->EnqueueWrapper(stream);
}

DEFINE_SHIM_FUNC3(SubmitAclCommand, aclrtStream,
                  hal::ascend::AclModelExecuteCommand,
                  aclError             , XModelExecuteAsync,
                  uint32_t             , model_id,
                  const aclmdlDataset *, input,
                  aclmdlDataset *      , output);

DEFINE_SHIM_FUNC5(SubmitAclCommand, aclrtStream,
                  hal::ascend::AclMemcpyCommand,
                  aclError       , XMemcpyAsync,
                  void *         , dst,
                  size_t         , dest_max,
                  const void *   , src,
                  size_t         , count,
                  aclrtMemcpyKind, kind);

DEFINE_SHIM_FUNC4(SubmitAclCommand, aclrtStream,
                  hal::ascend::AclMemsetCommand,
                  aclError, XMemsetAsync,
                  void *  , dev_ptr,
                  size_t  , max_count,
                  int32_t , value,
                  size_t  , count);

aclError XEventRecord(aclrtEvent event, aclrtStream stream);
aclError XEventSynchronize(aclrtEvent event);
aclError XStreamSynchronize(aclrtStream stream);
aclError XCreateStream(aclrtStream *stream);

} // namespace xsched::shim::ascend
