#include "utils/symbol.h"
#include "utils/intercept.h"
#include "hal/ascend/acl.h"
#include "hal/ascend/api.h"
#include "shim/ascend/shim.h"

using namespace xsched::hal::ascend;
using namespace xsched::shim::ascend;

DEFINE_INTERCEPT_FUNC4(XModelExecuteAsync, false,
                       aclError, aclmdlExecuteAsync,
                       uint32_t             , model_id,
                       const aclmdlDataset *, input,
                       aclmdlDataset *      , output,
                       aclrtStream          , stream);

DEFINE_INTERCEPT_FUNC6(XMemcpyAsync, false,
                       aclError, aclrtMemcpyAsync,
                       void *         , dst,
                       size_t         , dest_max,
                       const void *   , src,
                       size_t         , count,
                       aclrtMemcpyKind, kind,
                       aclrtStream    , stream);

DEFINE_INTERCEPT_FUNC5(XMemsetAsync, false,
                       aclError, aclrtMemsetAsync,
                       void *     , dev_ptr,
                       size_t     , max_count,
                       int32_t    , value,
                       size_t     , count,
                       aclrtStream, stream);

DEFINE_INTERCEPT_FUNC2(XEventRecord, false,
                       aclError, aclrtRecordEvent,
                       aclrtEvent , event,
                       aclrtStream, stream);

DEFINE_INTERCEPT_FUNC1(XEventSynchronize, false,
                       aclError, aclrtSynchronizeEvent,
                       aclrtEvent, event);

DEFINE_INTERCEPT_FUNC1(XStreamSynchronize, false,
                       aclError, aclrtSynchronizeStream,
                       aclrtStream, stream);

DEFINE_INTERCEPT_FUNC1(XCreateStream, false,
                       aclError, aclrtCreateStream,
                       aclrtStream *, stream);
