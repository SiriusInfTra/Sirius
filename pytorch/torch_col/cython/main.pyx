# cython: c_string_type=unicode, c_string_encoding=utf8
# distutils: language = c++
include "./ctrl_stub.pxi"

from libcpp.string cimport string
from libcpp.optional cimport optional, nullopt_t, make_optional
from libcpp.functional cimport function
from libcpp cimport bool

from posix.unistd cimport pid_t
from libc.stdint cimport uint64_t, uint32_t
from enum import Enum
import os

############################
#  MARK: Torch Col Init    #
############################

cdef extern from "<torch_col/csrc/config.h>" namespace "torch_col":
    cdef cppclass TorchColConfig:
        @staticmethod
        void InitConfig()
        @staticmethod
        bint HasColocatedInferServer()
        @staticmethod
        bint IsEnableSharedTensor()
        @staticmethod
        bint IsEnableDynamicSmPartition()
        @staticmethod
        bint IsEnableXsched()
        @staticmethod
        string GetHookMode()
        @staticmethod
        bint IsReleaseIntermMemoryByGradFn()
        @staticmethod
        void SetReleaseIntermMemoryByGradFn(bint)
        @staticmethod
        bint IsReleaseIntermMemoryByTagging()
        @staticmethod
        void SetReleaseIntermMemoryByTagging(bint)
        @staticmethod
        bint IsEnableFbwardHook()
        @staticmethod
        int GetTrainRank()
        @staticmethod
        void SetTrainRank(int)
        @staticmethod
        int GetTrainWorldSize()
        @staticmethod
        void SetTrainWorldSize(int)


cdef extern from "<common/device_manager.h>" namespace "colserve::sta":
    cdef cppclass DeviceManager:
        @staticmethod
        void Init()


cdef extern from "<torch_col/csrc/init.h>" namespace "torch_col":
    cpdef void TorchColInit(int, int)
    cpdef void SMPartitionInit(uint64_t stream)
    cpdef void TorchExtInit()


class HookMode(Enum):
    NONE = 'none'
    SYNC = 'sync'
    # XSCHED_ASYNC_SIGNAL = 'xsched-async-signal'  
    XSCHED_SYNC = 'xsched-sync'
    XSCHED_SYNC2 = 'xsched-sync2'

    def use_xsched(self):
        return self in {HookMode.XSCHED_SYNC, HookMode.XSCHED_SYNC2}


def is_enable_shared_tensor():
    return TorchColConfig.IsEnableSharedTensor()


def is_enable_dynamic_sm_partition():
    return TorchColConfig.IsEnableDynamicSmPartition()


def is_enable_xsched():
    return TorchColConfig.IsEnableXsched()


def get_hook_mode():
    # return TorchColConfig.GetHookMode()
    cdef hook_mode_cstr = TorchColConfig.GetHookMode()
    for hook_mode in HookMode:
        if hook_mode.value == hook_mode_cstr:
            return hook_mode
    raise Exception(f"Invalid hook mode: {hook_mode_cstr}")


def is_release_interm_memory_v1():
    return TorchColConfig.IsReleaseIntermMemoryByGradFn()


def is_release_interm_memory_v2():
    return TorchColConfig.IsReleaseIntermMemoryByTagging()


def disable_release_interm_memory():
    TorchColConfig.SetReleaseIntermMemoryByGradFn(False)
    TorchColConfig.SetReleaseIntermMemoryByTagging(False)


def disable_fbward_hook():
    TorchColConfig.SetReleaseIntermMemoryByGradFn(False)


def is_enable_fbward_hook():
    return TorchColConfig.IsEnableFbwardHook()


def torch_col_init(train_rank = 0, train_world_size = 1):
    assert train_rank >= 0 and train_world_size > 0
    assert train_rank < train_world_size
    TorchColInit(train_rank, train_world_size)


def get_train_rank():
    return TorchColConfig.GetTrainRank()


def set_train_rank(rank):
    TorchColConfig.SetTrainRank(rank)


def get_train_world_size():
    return TorchColConfig.GetTrainWorldSize()


def set_train_world_size(world_size):
    TorchColConfig.SetTrainWorldSize(world_size)


def set_train_rank_world_size(rank, world_size):
    set_train_rank(rank)
    set_train_world_size(world_size)


def has_colocated_infer_server():
    return TorchColConfig.HasColocatedInferServer()

#############################
#  MARK: Memory Management  #
#############################

cdef extern from "<torch_col/csrc/torch_allocator_plugin.h>" namespace "torch::cuda::CUDAColAllocator":
    cdef cppclass CUDAColAllocator:
        @staticmethod
        CUDAColAllocator* Get()
        @staticmethod
        void Init()
        @staticmethod
        void SetCurrentAllocator()
        void init(int)


cdef extern from "<common/cuda_allocator.h>" namespace "colserve::sta":
    cdef cppclass CUDAMemPool:
        @staticmethod
        CUDAMemPool* Get(int)
        @staticmethod
        size_t InferMemUsage()
        @staticmethod
        size_t TrainMemUsage()
        @staticmethod
        size_t TrainAllMemUsage()
        @staticmethod
        void FreeTrainLocals()


# release activations by traversing grad_fn 
cdef extern from "<torch_col/csrc/util.h>" namespace "torch_col":
  cdef void ReleaseGradFnSavedTensor(PyObject* function)
  cdef void ReleaseUnderlyingStorage(PyObject* tensor)


# release activations by tagging activations,
# interactive with memory pool to release memory dirrectly 
cdef extern from "<torch_col/csrc/mem_tagging.h>" namespace "torch_col":
    cdef void TagModelParameterStart()
    cdef void TagModelParameterEnd()
    cdef void TagIntermMemory(PyObject* obj)
    cdef void ReleaseIntermMemory()
    cdef void UntagIntermMemory()
    cdef void RearrangeMemory()


cdef extern from "<torch_col/csrc/util.h>" namespace "torch_col":
    cpdef void DumpMempoolFreeList(string filename)
    cpdef void DumpMempoolBlockList(string filename)


def cuda_memory_pool_infer_usage(device_id):
    return CUDAMemPool.Get(device_id).InferMemUsage()


def cuda_memory_pool_train_usage(device_id):
    return CUDAMemPool.Get(device_id).TrainMemUsage()


def cuda_memory_pool_train_all_usage(device_id):
    return CUDAMemPool.Get(device_id).TrainAllMemUsage()


def cuda_memory_pool_free_train_local(device_id):
    CUDAMemPool.Get(device_id).FreeTrainLocals()


def release_grad_fn_saved_tensor(grad_fn):
    cdef PyObject* obj = <PyObject*> grad_fn
    ReleaseGradFnSavedTensor(obj)


def release_underlying_storage(tensor):
    cdef PyObject* obj = <PyObject*> tensor
    ReleaseUnderlyingStorage(obj)


def tag_model_start():
    TagModelParameterStart()


def tag_model_end():
    TagModelParameterEnd()


def tag_interm_memory(tensor):
    cdef PyObject* obj = <PyObject*> tensor
    TagIntermMemory(obj)


def release_interm_memory():
    ReleaseIntermMemory()


def untag_interm_memory():
    UntagIntermMemory()


def rearrange_memory():
    RearrangeMemory()



#############################
#  MARK: GPU SM Management  #
#############################

cdef extern from "<common/xsched_ctrl.h>" namespace "colserve::sta::xsched":
    cpdef uint64_t RegisterStream(uint64_t stream)
    cpdef void UnRegisterStream(uint64_t stream)
    cpdef void UnRegisterAllStreams()
    cpdef size_t GetXQueueSize(uint64_t stream)
    cpdef size_t GetTotalXQueueSize()
    cpdef uint64_t AbortStream(uint64_t stream)
    cpdef uint64_t AbortAllStreams()
    cpdef int SyncStream(uint64_t stream)
    cpdef int SyncAllStreams()
    cpdef void GuessNcclBegin()
    cpdef void GuessNcclEnd()
    cpdef vector[uint64_t] GetNcclStreams()


cdef extern from "<common/sm_partition.h>" namespace "colserve":
    cdef cppclass SMPartitioner:
        @staticmethod
        void Init(int device_id)
        @staticmethod
        SMPartitioner* Get(int device_id)

        int GetInferRequiredTpcNum()
        int GetTrainAvailTpcNum()
        uint64_t GetTrainAvailTpcMask()


# def GetXQueueSize_(stream):
#     cdef optional[uint64_t] stream_opt
#     if stream is not None:
#         stream_opt = make_optional[uint64_t](<uint64_t> stream)
#     return GetXQueueSize(stream_opt)


def monitor_sm_partition(interval: float):
    import sys, time

    if not TorchColConfig.HasColocatedInferServer():
        print("There not exist colocated infer server")
        return

    SMPartitioner.Init(0)

    fmt = "Infer TPC Num: {}, Train TPC Num: {}, Train Avail TPC Mask: {}"
    while True:
        print(fmt.format(
            SMPartitioner.Get(0).GetInferRequiredTpcNum(),
            SMPartitioner.Get(0).GetTrainAvailTpcNum(),
            hex(SMPartitioner.Get(0).GetTrainAvailTpcMask())
        ))
        time.sleep(interval)


########################
#  MARK: Inf-Tra-Comm  #
########################

cdef extern from "<common/inf_tra_comm/communicator.h>" namespace "colserve::ctrl":
    cpdef enum class CtrlEvent(int):
        # status event
        kTrainStart,
        kTrainEnd,
        kInterruptTrainDone,
        kResumeTrainDone,
        kColocateAdjustL1Done,
        kColocateAdjustL2Done,
        
        kReportBatchSize,

        # cmd event: switch mode
        kInterruptTrain,
        kResumeTrain,
        # cmd event: colocate mode
        kColocateAdjustL1,
        kColocateAdjustL2,
        kInferExit, # train adjust back

        kInferenceWorkloadDone,

        kNumEvent,


    cdef struct CtrlMsgEntry:
        uint64_t id
        int event
        int value


    cdef cppclass InfTraMessageQueue:
        CtrlMsgEntry BlockGet(Direction direction, int id)
        bool TimedGet(uint32_t timeout_ms, Direction direction, int id,
                      CtrlMsgEntry &msg)
        void Put(const CtrlMsgEntry &entry, Direction direction, int id)
        void PutAll(const CtrlMsgEntry &entry, Direction direction)
    

    cdef cppclass InfTraInfoBoard:
        void SetTrainInfo(int id, function fn)
        void SetTrainInfo(int id, optional[pid_t] pid,
                          optional[int] rank, optional[int] world_size,
                          optional[int] init_batch_size, 
                          optional[int] current_batch_size)

    cdef cppclass InfTraCommunicator:
        @staticmethod
        void Init(bool is_server, bool cleanup, int train_world_size)
        @staticmethod
        bool IsInitialized()
        @staticmethod
        InfTraMessageQueue* GetMQ()
        @staticmethod
        InfTraInfoBoard* GetIB()


cdef extern from "<common/inf_tra_comm/communicator.h>" namespace "colserve::ctrl::InfTraMessageQueue":
    cdef enum class Direction:
        kInf2Tra,
        kTra2Inf,
        kNumDirection


cdef class PyCtrlMsgEntry:
    cdef CtrlMsgEntry _cppclass
    
    def __cinit__(self, uint64_t id, CtrlEvent cmd, int value):
        self._cppclass = CtrlMsgEntry(id, int(cmd), value)

    @property
    def event(self):
        return CtrlEvent(self._cppclass.event)

    @property
    def id(self):
        return self._cppclass.id

    @property
    def value(self):
        return self._cppclass.value

    def __repr__(self):
        return "PyCtrlMsgEntry(id={}, event={}, value={})".format(self.id, str(self.event), self.value)


class PyInfTraCommunicator:
    def __init__(self, is_server = None, cleanup = None, train_world_size = None):
        if InfTraCommunicator.IsInitialized():
            return
        if (
            is_server is None 
            or cleanup is None
            or train_world_size is None
        ):
            raise Exception("Invalid InfTraCommunicator init args")
        InfTraCommunicator.Init(<bool> is_server, <bool> cleanup, 
                                <int> train_world_size)

    def put_inf2tra(self, PyCtrlMsgEntry entry, int id):
        InfTraCommunicator.GetMQ().Put(
            entry._cppclass, InfTraMessageQueue.Direction.kInf2Tra, id)

    def put_all_inf2tra(self, PyCtrlMsgEntry entry):
        InfTraCommunicator.GetMQ().PutAll(
            entry._cppclass, InfTraMessageQueue.Direction.kInf2Tra)

    def block_get_inf2tra(self, int id):
        return InfTraCommunicator.GetMQ().BlockGet(
            InfTraMessageQueue.Direction.kInf2Tra, id)

    def block_get_tra2inf(self, int id):
        return InfTraCommunicator.GetMQ().BlockGet(
            InfTraMessageQueue.Direction.kTra2Inf, id)

    def timed_get_inf2tra(self, uint32_t timeout_ms, int id):
        cdef CtrlMsgEntry msg
        if InfTraCommunicator.GetMQ().TimedGet(
            timeout_ms, InfTraMessageQueue.Direction.kInf2Tra, 
            id, msg
        ):
            return PyCtrlMsgEntry(msg.id, CtrlEvent(msg.event), msg.value)
        return None

    def timed_get_tra2inf(self, uint32_t timeout_ms, int id):
        cdef CtrlMsgEntry msg
        if InfTraCommunicator.GetMQ().TimedGet(
            timeout_ms, InfTraMessageQueue.Direction.kTra2Inf, 
            id, msg
        ):
            return PyCtrlMsgEntry(msg.id, CtrlEvent(msg.event), msg.value)
        return None 


def init_train_info(init_batch_size, 
                    current_batch_size,
                    pid = None):
    if not TorchColConfig.HasColocatedInferServer():
        print("There not exist colocated infer server, skip init train info")
        return

    cdef optional[pid_t] pid_opt
    if pid is not None:
        pid_opt = make_optional[pid_t](<pid_t> pid)
    else:
        pid_opt = make_optional[pid_t](<pid_t> os.getpid())

    InfTraCommunicator.GetIB().SetTrainInfo(
        TorchColConfig.GetTrainRank(), pid_opt,
        make_optional[int](TorchColConfig.GetTrainRank()), 
        make_optional[int](TorchColConfig.GetTrainWorldSize()), 
        make_optional[int](<int> init_batch_size), 
        make_optional[int](<int> current_batch_size)
    )


def update_current_batch_size(current_batch_size):
    if not TorchColConfig.HasColocatedInferServer():
        return

    InfTraCommunicator.GetIB().SetTrainInfo(
        TorchColConfig.GetTrainRank(), 
        optional[pid_t](), optional[int](), 
        optional[int](), optional[int](), 
        make_optional[int](<int> current_batch_size)
    )

####################
#  MARK: Utililty  #
####################

cdef extern from "<torch_col/csrc/util.h>" namespace "torch_col":
    cpdef long get_unix_timestamp()
    cpdef long get_unix_timestamp_us()


cdef extern from "<torch_col/csrc/util.h>" namespace "torch_col":
    cdef cppclass TensorWeakRef:
        TensorWeakRef(PyObject *tensor) except +
        size_t Nbytes()
        size_t StorageNbytes()
        void* DataPtr()


cdef class PyTensorWeakRef:
    cdef TensorWeakRef* _cppclass
    
    def __cinit__(self, tensor):
        cdef PyObject* obj = <PyObject*> tensor
        self._cppclass = new TensorWeakRef(obj)

    def nbytes(self):
        return self._cppclass.Nbytes()

    def storage_nbytes(self):
        return self._cppclass.StorageNbytes()

    def data_ptr(self):
        return <size_t>self._cppclass.DataPtr()

    def __dealloc__(self):
        del self._cppclass


