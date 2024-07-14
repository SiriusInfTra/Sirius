# cython: c_string_type=unicode, c_string_encoding=utf8
# distutils: language = c++
include "./ctrl_stub.pxi"

from libcpp.string cimport string
from enum import Enum

############################
#  MARK: Torch Col Config  #
############################

cdef extern from "<csrc/config.h>" namespace "torch_col":
    cdef cppclass TorchColConfig:
        @staticmethod
        void InitConfig()
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


cdef extern from "<csrc/xsched.h>" namespace "torch_col":
    cpdef void InitSMPartition()


cdef extern from "<csrc/init.h>" namespace "torch_col":
    cpdef void TorchColInit(int)


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


def torch_col_init(device_count):
    TorchColInit(device_count)


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


#############################
#  MARK: Memory Management  #
#############################

cdef extern from "<csrc/cuda_allocator_plugin.h>" namespace "torch::cuda::CUDAColAllocator":
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
        size_t InferMemUsage(int)
        @staticmethod
        size_t TrainMemUsage(int)
        @staticmethod
        size_t TrainAllMemUsage(int)
        @staticmethod
        void FreeTrainLocals(int)


# release activations by traversing grad_fn 
cdef extern from "<csrc/util.h>" namespace "torch_col":
  cdef void ReleaseGradFnSavedTensor(PyObject* function)
  cdef void ReleaseUnderlyingStorage(PyObject* tensor)


# release activations by tagging activations,
# interactive with memory pool to release memory dirrectly 
cdef extern from "<csrc/mem_tagging.h>" namespace "torch_col":
    cdef void TagModelParameterStart()
    cdef void TagModelParameterEnd()
    cdef void TagIntermMemory(PyObject* obj)
    cdef void ReleaseIntermMemory()
    cdef void UntagIntermMemory()
    cdef void RearrangeMemory()


cdef extern from "<csrc/util.h>" namespace "torch_col":
    cpdef void DumpMempoolFreeList(string filename)
    cpdef void DumpMempoolBlockList(string filename)


def cuda_memory_pool_infer_usage(device_id):
    return CUDAMemPool.InferMemUsage(device_id)


def cuda_memory_pool_train_usage(device_id):
    return CUDAMemPool.TrainMemUsage(device_id)


def cuda_memory_pool_train_all_usage(device_id):
    return CUDAMemPool.TrainAllMemUsage(device_id)


def cuda_memory_pool_free_train_local(device_id):
    CUDAMemPool.FreeTrainLocals(device_id)


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

cdef extern from "<common/sm_partition.h>" namespace "colserve":
    cdef cppclass SMPartitioner:
        @staticmethod
        void Init(int device, int cleanup, int observe)
        @staticmethod
        int GetInferRequiredTpcNum()
        @staticmethod
        int GetTrainAvailTpcNum()
        @staticmethod
        unsigned long long GetTrainAvailTpcMask()


def monitor_sm_partition(interval: float):
    import sys, time
    SMPartitioner.Init(0, 0, 1)

    fmt = "Infer TPC Num: {}, Train TPC Num: {}, Train Avail TPC Mask: {}"
    while True:
        print(fmt.format(
            SMPartitioner.GetInferRequiredTpcNum(),
            SMPartitioner.GetTrainAvailTpcNum(),
            hex(SMPartitioner.GetTrainAvailTpcMask())
        ))
        time.sleep(interval)



####################
#  MARK: Utililty  #
####################

cdef extern from "<csrc/util.h>" namespace "torch_col":
    cdef cppclass TensorWeakRef:
        TensorWeakRef(PyObject *tensor) except +
        size_t Nbytes()
        size_t StorageNbytes()
        void* DataPtr()


cdef class PyTensorWeakRef:
    cdef TensorWeakRef* _tensor_weak_ref
    
    def __cinit__(self, tensor):
        cdef PyObject* obj = <PyObject*> tensor
        self._tensor_weak_ref = new TensorWeakRef(obj)

    def nbytes(self):
        return self._tensor_weak_ref.Nbytes()

    def storage_nbytes(self):
        return self._tensor_weak_ref.StorageNbytes()

    def data_ptr(self):
        return <size_t>self._tensor_weak_ref.DataPtr()

    def __dealloc__(self):
        del self._tensor_weak_ref





