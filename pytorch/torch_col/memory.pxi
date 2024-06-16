# cython: c_string_type=unicode, c_string_encoding=utf8
# distutils: language = c++
from libcpp.string cimport string
from cpython.ref cimport PyObject


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

def cuda_memory_pool_infer_usage():
    return CUDAMemPool.InferMemUsage()

def cuda_memory_pool_train_usage():
    return CUDAMemPool.TrainMemUsage()

def cuda_memory_pool_train_all_usage():
    return CUDAMemPool.TrainAllMemUsage()

def cuda_memory_pool_free_train_local():
    CUDAMemPool.FreeTrainLocals()


# release activations by traversing grad_fn 
cdef extern from "<csrc/util.h>" namespace "torch_col":
  cdef void ReleaseGradFnSavedTensor(PyObject* function)
  cdef void ReleaseUnderlyingStorage(PyObject* tensor)

def release_grad_fn_saved_tensor(grad_fn):
    cdef PyObject* obj = <PyObject*> grad_fn
    ReleaseGradFnSavedTensor(obj)

def release_underlying_storage(tensor):
    cdef PyObject* obj = <PyObject*> tensor
    ReleaseUnderlyingStorage(obj)


# release activations by tagging activations,
# interactive with memory pool to release memory dirrectly 
cdef extern from "<csrc/mem_tagging.h>" namespace "torch_col":
    cdef void TagModelParameterStart()
    cdef void TagModelParameterEnd()
    cdef void TagIntermMemory(PyObject* obj)
    cdef void ReleaseIntermMemory()
    cdef void UntagIntermMemory()
    cdef void RearrangeMemory()

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

cdef extern from "<csrc/util.h>" namespace "torch_col":
    cpdef void DumpMempoolFreeList(string filename)
    cpdef void DumpMempoolBlockList(string filename)


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
    