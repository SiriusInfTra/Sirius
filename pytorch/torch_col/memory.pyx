# cython: c_string_type=unicode, c_string_encoding=utf8
cdef extern from "csrc/cuda_allocator_plugin.h" namespace "torch::cuda::CUDAColAllocator":
  cdef cppclass CUDAColAllocator:
    @staticmethod
    void Init()

    @staticmethod
    void SetCurrentAllocator()

  
# def init_col_allocator():
#   CUDAColAllocator.Init()
#   CUDAColAllocator.SetCurrentAllocator()
#   print("CUDAColAllocator initialized")

def _init_col_allocator():
  CUDAColAllocator.Init()
  CUDAColAllocator.SetCurrentAllocator()
  print("CUDAColAllocator initialized")


