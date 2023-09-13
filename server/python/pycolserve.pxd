# distutils: language = c++

from libcpp.string cimport string

cdef extern from "control_stub.cc" namespace "pycolserve":
    pass

cdef extern from "../block_queue.h" namespace "colserve":
    cdef cppclass MemoryQueue[T]:
        MemoryQueue(string, bint) except + 
        void Put(T)
        T BlockGet()
        bint TimedGet(T&, size_t)

cdef extern from "control_stub.h" namespace "pycolserve":
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
        ColocateStub() except +
        void Stop()
        int Cmd()
        void Cmd(int)
        void ColocateAdjustL1Done()
        void ColocateAdjustL2Done()
        void TrainStart()
        void TrainEnd()




