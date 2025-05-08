#include <fstream>
#include <sstream>

#include "hal/cuda/trap.h"
#include "hal/cuda/instrument.h"
#include "hal/cuda/cuda_assert.h"

using namespace xsched::hal::cuda;

TrapManager::TrapManager(CUdevice device,
                         CUcontext context,
                         CUstream operation_stream)
    : device_(device)
    , context_(context)
    , operation_stream_(operation_stream)
{
    cuXtraGetTrapHandlerInfo(context_, &trap_handler_dev_, &trap_handler_size_);
}

void TrapManager::SetTrapHandler()
{

// #define DUMP_TRAP_HANDLER

#ifdef DUMP_TRAP_HANDLER
    DumpTrapHandler();
#endif

    static const size_t kExtraInstrsSize = 1024;

    char *trap_handler_host = (char *)malloc(trap_handler_size_);
    char *extra_instrs_host = (char *)malloc(kExtraInstrsSize);

    CUdeviceptr extra_instrs_device;
    CUDA_ASSERT(Driver::MemAllocAsync(&extra_instrs_device,
                                          kExtraInstrsSize,
                                          operation_stream_));
    CUDA_ASSERT(Driver::StreamSynchronize(operation_stream_));

    // copy trap handler instructions to host
    cuXtraMemcpyDtoH(trap_handler_host, trap_handler_dev_,
                     trap_handler_size_, operation_stream_);

    InstrumentManager::InstrumentTrapHandler(trap_handler_host,
                                             trap_handler_dev_,
                                             trap_handler_size_,
                                             extra_instrs_host,
                                             extra_instrs_device,
                                             kExtraInstrsSize);

    CUDA_ASSERT(Driver::MemcpyHtoDAsyncV2(extra_instrs_device,
                                              extra_instrs_host,
                                              kExtraInstrsSize,
                                              operation_stream_));
    // copy trap handler instructions back to device
    cuXtraMemcpyHtoD(trap_handler_dev_, trap_handler_host,
                     trap_handler_size_, operation_stream_);

    CUDA_ASSERT(Driver::StreamSynchronize(operation_stream_));
    free(trap_handler_host);
    free(extra_instrs_host);
}

void TrapManager::InterruptContext()
{
    cuXtraTriggerTrap(context_);
}

void TrapManager::DumpTrapHandler()
{
    printf("dumping trap handler...\n");

    char *trap_handler_host = (char *)malloc(trap_handler_size_);
    // copy trap handler instructions to host
    cuXtraMemcpyDtoH(trap_handler_host, trap_handler_dev_,
                     trap_handler_size_, operation_stream_);

    std::stringstream filename;
    filename << "trap_handler_0x" << std::hex << trap_handler_dev_ << ".bin";
    std::ofstream out_file(filename.str(), std::ios::binary);
    out_file.write(trap_handler_host, trap_handler_size_);
    out_file.close();

    printf("dumped trap handler in %s\n", filename.str().c_str());
}
