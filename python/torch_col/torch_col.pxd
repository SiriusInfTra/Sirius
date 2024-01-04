# distutils: language = c++

from libcpp.string cimport string
from cpython.ref cimport PyObject

    
cdef extern from "<sta/cuda_allocator.h>" namespace "colserve::sta":
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


cdef extern from "<csrc/torch_helper.h>" namespace "torch_col":
    cdef void ReleaseGradFnSavedTensor(PyObject* function)


cdef extern from "<csrc/control_stub.h>" namespace "colserve":
    cdef cppclass MemoryQueue[T]:
        MemoryQueue(string, bint) except + 
        void Put(T)
        T BlockGet()
        bint TimedGet(T&, size_t)


cdef extern from "<csrc/control_stub.h>" namespace "colserve":
    cdef struct CtrlMsgEntry:
        unsigned long long id
        int event
        int value


cdef extern from "<csrc/control_stub.h>" namespace "torch_col":
    cdef cppclass SwitchStub:
        SwitchStub() except +
        int Cmd()
        void Cmd(int)
        void Stop()
        void TrainStart()
        void TrainEnd()
        bint TryInterruptTrainDone()
        void ReportBatchSize(int)
        void StepsNoInteruptBegin()
        void StepsNoInteruptEnd()

    cdef cppclass ColocateStub:
        ColocateStub(int) except +
        void Stop()
        int Cmd()
        int TargetBatchSize()
        void ColocateAdjustL1Done()
        void ColocateAdjustL2Done()
        void TrainStart()
        void TrainEnd()
        void ReportBatchSize(int)
        void StepsNoInteruptBegin()
        void StepsNoInteruptEnd()