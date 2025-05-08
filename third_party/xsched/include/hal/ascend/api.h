#pragma once

#include "utils/common.h"
#include "utils/symbol.h"
#include "hal/ascend/acl.h"

namespace xsched::hal::ascend
{

class Api
{
public:
    STATIC_CLASS(Api);

    DEFINE_STATIC_SYMBOL1("aclrtCreateStream", GetSymbol,
                          aclError     , CreateStream,
                          aclrtStream *, stream);
    
    DEFINE_STATIC_SYMBOL1("aclrtDestroyStream", GetSymbol,
                          aclError   , DestroyStream,
                          aclrtStream, stream);
    
    DEFINE_STATIC_SYMBOL1("aclrtSynchronizeStream", GetSymbol,
                          aclError   , SynchronizeStream,
                          aclrtStream, stream);
    
    DEFINE_STATIC_SYMBOL1("aclrtCreateEvent", GetSymbol,
                          aclError    , CreateEvent,
                          aclrtEvent *, event);

    DEFINE_STATIC_SYMBOL1("aclrtDestroyEvent", GetSymbol,
                          aclError  , DestroyEvent,
                          aclrtEvent, event);
    
    DEFINE_STATIC_SYMBOL2("aclrtRecordEvent", GetSymbol,
                          aclError   , RecordEvent,
                          aclrtEvent , event,
                          aclrtStream, stream);

    DEFINE_STATIC_SYMBOL1("aclrtSynchronizeEvent", GetSymbol,
                          aclError  , SynchronizeEvent,
                          aclrtEvent, event);
    
    DEFINE_STATIC_SYMBOL6("aclrtMemcpyAsync", GetSymbol,
                          aclError       , MemcpyAsync,
                          void *         , dst,
                          size_t         , dest_max,
                          const void *   , src,
                          size_t         , count,
                          aclrtMemcpyKind, kind,
                          aclrtStream    , stream);
    
    DEFINE_STATIC_SYMBOL5("aclrtMemsetAsync", GetSymbol,
                          aclError   , MemsetAsync,
                          void *     , dev_ptr,
                          size_t     , max_count,
                          int32_t    , value,
                          size_t     , count,
                          aclrtStream, stream);

    DEFINE_STATIC_SYMBOL4("aclmdlExecuteAsync" , GetSymbol,
                          aclError             , ModelExecuteAsync,
                          uint32_t             , model_id,
                          const aclmdlDataset *, input,
                          aclmdlDataset *      , output,
                          aclrtStream          , stream);
    
    DEFINE_STATIC_SYMBOL1("aclrtGetCurrentContext", GetSymbol,
                          aclError      , GetCurrentContext,
                          aclrtContext *, context);
    
    DEFINE_STATIC_SYMBOL1("aclrtSetCurrentContext", GetSymbol,
                          aclError    , SetCurrentContext,
                          aclrtContext, context);

public:
    DEFINE_GET_SYMBOL_FUNC(GetSymbol, "libascendcl.so", ENV_ASCENDCL_DLL_PATH,
        {"/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64"});
};

} // namespace xsched::hal::ascend
