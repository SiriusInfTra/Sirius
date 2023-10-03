# distutils: language = c++

from libcpp.string cimport string

# cdef extern from "./csrc/control_stub.cc" namespace "torch_col":
#     pass

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


cdef extern from "<csrc/control_stub.h>" namespace "torch_col":
    cpdef enum class Event(int):
        # status 
        kTrainStart,
        kTrainEnd,
        kInterruptTrainDone,
        kResumeTrainDone,
        kColocateAdjustL1Done,
        kColocateAdjustL2Done,

        # cmd 
        kInterruptTrain,
        kResumeTrain,
        kColocateAdjustL1,
        kColocateAdjustL2,

    cdef cppclass SwitchStub:
        SwitchStub() except +
        int Cmd()
        void Cmd(int)
        void Stop()
        void TrainStart()
        void TrainEnd()

    cdef cppclass ColocateStub:
        ColocateStub(int) except +
        void Stop()
        int Cmd()
        int TargetBatchSize()
        void ColocateAdjustL1Done()
        void ColocateAdjustL2Done()
        void TrainStart()
        void TrainEnd()





