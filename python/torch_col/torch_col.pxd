# distutils: language = c++

from libcpp.string cimport string

cdef extern from "<sta/cuda_allocator.h>" namespace "colserve::sta":
    cdef cppclass CUDAMemPool:
        @staticmethod
        size_t InferMemUsage()

        @staticmethod
        size_t TrainMemUsage()


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
        void ReportBatchSize(int)

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