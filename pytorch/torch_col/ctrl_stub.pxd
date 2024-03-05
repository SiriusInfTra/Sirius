# distutils: language = c++
from libcpp.string cimport string
from libcpp.vector cimport vector
from cpython.ref cimport PyObject


cdef extern from "<common/block_queue.h>" namespace "colserve":
    cdef cppclass MemoryQueue[T]:
        MemoryQueue(string, bint) except + 
        void Put(T)
        T BlockGet()
        bint TimedGet(T&, size_t)


cdef extern from "<common/controlling.h>" namespace "colserve::ctrl":
    cdef struct CtrlMsgEntry:
        unsigned long long id
        int event
        int value


cdef extern from "<csrc/config.h>" namespace "torch_col":
    cdef int kill_batch_on_recv


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

    cdef cppclass StubProfiler:
        @staticmethod
        vector[long] GetAdjustRequestTimeStamp()
        @staticmethod
        vector[long] GetAdjustDoneTimeStamp()


cdef extern from "<csrc/util.h>" namespace "torch_col":
  cdef void ReleaseGradFnSavedTensor(PyObject* function)
