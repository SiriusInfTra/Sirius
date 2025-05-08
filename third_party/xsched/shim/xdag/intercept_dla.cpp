#include "utils/symbol.h"
#include "utils/intercept.h"
#include "shim/xdag/shim.h"
#include "hal/cudla/cudla.h"
#include "hal/cudla/driver.h"

using namespace xsched::shim::xdag;
using namespace xsched::hal::cudla;

DEFINE_INTERCEPT_FUNC1(DlaDriver::GetVersion, false,
                       cudlaStatus, cudlaGetVersion,
                       uint64_t * , version);

DEFINE_INTERCEPT_FUNC1(DlaDriver::DeviceGetCount, false,
                       cudlaStatus, cudlaDeviceGetCount,
                       uint64_t * , p_num_devices);

DEFINE_INTERCEPT_FUNC3(XCreateDevice, false,
                       cudlaStatus, cudlaCreateDevice,
                       uint64_t const        , device,
                       cudlaDevHandle * const, dev_handle,
                       uint32_t const        , flags);

DEFINE_INTERCEPT_FUNC5(DlaDriver::MemRegister, false,
                       cudlaStatus, cudlaMemRegister,
                       cudlaDevHandle const  , dev_handle,
                       const uint64_t * const, ptr,
                       size_t const          , size,
                       uint64_t ** const     , dev_ptr,
                       uint32_t const        , flags);

DEFINE_INTERCEPT_FUNC5(DlaDriver::ModuleLoadFromMemory, false,
                       cudlaStatus, cudlaModuleLoadFromMemory,
                       cudlaDevHandle const , dev_handle,
                       const uint8_t * const, p_module,
                       size_t const         , module_size,
                       cudlaModule * const  , h_module,
                       uint32_t const       , flags);

DEFINE_INTERCEPT_FUNC3(DlaDriver::ModuleGetAttributes, false,
                       cudlaStatus, cudlaModuleGetAttributes,
                       cudlaModule const             , h_module,
                       cudlaModuleAttributeType const, attr_type,
                       cudlaModuleAttribute * const  , attribute);

DEFINE_INTERCEPT_FUNC5(DlaDriver::SubmitTask, false,
                       cudlaStatus, cudlaSubmitTask,
                       cudlaDevHandle const   , dev_handle,
                       const cudlaTask * const, ptr_to_tasks,
                       uint32_t const         , num_tasks,
                       void * const           , stream,
                       uint32_t const         , flags);

DEFINE_INTERCEPT_FUNC2(DlaDriver::ModuleUnload, false,
                       cudlaStatus, cudlaModuleUnload,
                       cudlaModule const, h_module,
                       uint32_t const   , flags);

DEFINE_INTERCEPT_FUNC3(DlaDriver::DeviceGetAttribute, false,
                       cudlaStatus, cudlaDeviceGetAttribute,
                       cudlaDevHandle const       , dev_handle,
                       cudlaDevAttributeType const, attrib,
                       cudlaDevAttribute * const  , pAttribute);

DEFINE_INTERCEPT_FUNC2(DlaDriver::MemUnregister, false,
                       cudlaStatus, cudlaMemUnregister,
                       cudlaDevHandle const  , dev_handle,
                       const uint64_t * const, dev_ptr);

DEFINE_INTERCEPT_FUNC1(DlaDriver::GetLastError, false,
                       cudlaStatus, cudlaGetLastError,
                       cudlaDevHandle const, dev_handle);

DEFINE_INTERCEPT_FUNC1(DlaDriver::DestroyDevice, false,
                       cudlaStatus, cudlaDestroyDevice,
                       cudlaDevHandle const, dev_handle);

DEFINE_INTERCEPT_FUNC4(DlaDriver::ImportExternalMemory, false,
                       cudlaStatus, cudlaImportExternalMemory,
                       cudlaDevHandle const, dev_handle,
                       const cudlaExternalMemoryHandleDesc * const, desc,
                       uint64_t ** const   , dev_ptr,
                       uint32_t const      , flags);

DEFINE_INTERCEPT_FUNC2(DlaDriver::GetNvSciSyncAttributes, false,
                       cudlaStatus, cudlaGetNvSciSyncAttributes,
                       uint64_t * const, attr_list,
                       uint32_t const  , flags);

DEFINE_INTERCEPT_FUNC4(DlaDriver::ImportExternalSemaphore, false,
                       cudlaStatus, cudlaImportExternalSemaphore,
                       cudlaDevHandle const, dev_handle,
                       const cudlaExternalSemaphoreHandleDesc* const, desc,
                       uint64_t ** const   , dev_ptr,
                       uint32_t const      , flags);

DEFINE_INTERCEPT_FUNC2(DlaDriver::SetTaskTimeoutInMs, false,
                       cudlaStatus, cudlaSetTaskTimeoutInMs,
                       cudlaDevHandle const, dev_handle,
                       uint32_t const      , timeout);
