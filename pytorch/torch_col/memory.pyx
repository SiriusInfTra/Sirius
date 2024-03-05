# cython: c_string_type=unicode, c_string_encoding=utf8
from libcpp.string cimport string


cdef extern from "<csrc/cuda_allocator_plugin.h>" namespace "torch::cuda::CUDAColAllocator":
  cdef cppclass CUDAColAllocator:
    @staticmethod
    void Init()

    @staticmethod
    void SetCurrentAllocator()


cdef extern from "<common/cuda_allocator.h>" namespace "colserve::sta":
    cdef cppclass CUDAMemPool:
        @staticmethod
        size_t InferMemUsage()

        @staticmethod
        size_t TrainMemUsage()

        @staticmethod
        size_t TrainAllMemUsage()

        @staticmethod
        void FreeTrainLocals()

        @staticmethod
        size_t TrainAllocMs()

        @staticmethod
        void ResetTrainAllocMs()


cdef extern from "<csrc/util.h>" namespace "torch_col":
    cpdef void DumpMempoolFreeList(string filename)
    cpdef void DumpMempoolBlockList(string filename)


def init_col_allocator():
  CUDAColAllocator.Init()
  CUDAColAllocator.SetCurrentAllocator()
  print("CUDAColAllocator initialized")


def cuda_memory_pool_infer_usage():
    return CUDAMemPool.InferMemUsage()


def cuda_memory_pool_train_usage():
    return CUDAMemPool.TrainMemUsage()


def cuda_memory_pool_train_all_usage():
    return CUDAMemPool.TrainAllMemUsage()


def cuda_memory_pool_free_train_local():
    CUDAMemPool.FreeTrainLocals()


