# cython: c_string_type=unicode, c_string_encoding=utf8
# from .ctrl_stub cimport *
from libcpp.string cimport string
from libcpp.vector cimport vector
from cpython cimport bool
from cpython.ref cimport PyObject


cdef extern from "<torch_col/csrc/control_stub.h>" namespace "torch_col":
    cdef cppclass DummyStub:
        DummyStub() except +
        void Stop()
        void TrainStart()
        void TrainEnd()
        bint CanExitAfterInferWorkloadDone()

    cdef cppclass SwitchStub:
        SwitchStub() except +
        int GetCmd()
        void SetCmd(int)
        void Stop()
        void TrainStart()
        void TrainEnd()
        bint TryInterruptTrainDone()
        void ReportBatchSize(int)
        void StepsNoInteruptBegin()
        void StepsNoInteruptEnd()
        void EnableTorchColEngine()
        bint CanExitAfterInferWorkloadDone()

    cdef cppclass ColocateStub:
        ColocateStub(int) except +
        void Stop()
        int GetCmd()
        void SetCmd(int)
        int GetTargetBatchSize()
        void ColocateAdjustL1Done()
        void ColocateAdjustL2Done()
        void TrainStart()
        void TrainEnd()
        void ReportBatchSize(int)
        void StepsNoInteruptBegin()
        void StepsNoInteruptEnd()
        void EnableTorchColEngine()
        bint CanExitAfterInferWorkloadDone()

    cdef cppclass StubProfiler:
        @staticmethod
        vector[long] GetAdjustRequestTimeStamp()
        @staticmethod
        vector[long] GetAdjustDoneTimeStamp()



cdef class PyDummyStub:
    cdef DummyStub* _stub

    def __cinit__(self):
        self._stub = new DummyStub()

    def train_start(self):
        self._stub.TrainStart()

    def train_end(self):
        self._stub.TrainEnd()

    def stop(self):
        self._stub.Stop()

    def can_exit_after_infer_worklaod_done(self):
        return self._stub.CanExitAfterInferWorkloadDone()
    
    def __dealloc__(self):
        del self._stub


cdef class PySwitchStub:
    cdef SwitchStub* _stub

    def __cinit__(self):
        self._stub = new SwitchStub()

    def train_start(self):
        self._stub.TrainStart()

    def train_end(self):
        self._stub.TrainEnd()

    def stop(self):
        self._stub.Stop()

    def try_interrupt_train_done(self):
        return self._stub.TryInterruptTrainDone()

    def report_batch_size(self, batch_size):
        self._stub.ReportBatchSize(batch_size)
    
    def StepsNoInteruptBegin(self):
        self._stub.StepsNoInteruptBegin()

    def StepsNoInteruptEnd(self):
        self._stub.StepsNoInteruptEnd()

    @property
    def cmd(self):
        return self._stub.GetCmd()
    @cmd.setter
    def cmd(self, cmd):
        if cmd is None:
            self._stub.SetCmd(-1)
        else:
            self._stub.SetCmd(cmd)
    
    def EnableTorchColEngine(self):
        self._stub.EnableTorchColEngine()

    def can_exit_after_infer_worklaod_done(self):
        return self._stub.CanExitAfterInferWorkloadDone()

    def __dealloc__(self):
        del self._stub


cdef class PyColocateStub:
    cdef ColocateStub* _stub

    def __cinit__(self, batch_size):
        self._stub = new ColocateStub(batch_size)

    def stop(self):
        self._stub.Stop()

    @property
    def cmd(self):
        return self._stub.GetCmd()

    @property
    def target_batch_size(self):
        return self._stub.GetTargetBatchSize()

    def adjust_l1_done(self):
        self._stub.ColocateAdjustL1Done()

    def adjust_l2_done(self):
        self._stub.ColocateAdjustL2Done()

    def train_start(self):
        self._stub.TrainStart()

    def train_end(self):
        self._stub.TrainEnd()

    def report_batch_size(self, batch_size):
        self._stub.ReportBatchSize(batch_size)

    def StepsNoInteruptBegin(self):
        self._stub.StepsNoInteruptBegin()

    def StepsNoInteruptEnd(self):
        self._stub.StepsNoInteruptEnd()
    
    def can_exit_after_infer_worklaod_done(self):
        return self._stub.CanExitAfterInferWorkloadDone()

    def EnableTorchColEngine(self):
        self._stub.EnableTorchColEngine()

    def __dealloc__(self):
        del self._stub



def get_adjust_request_time_stamp():
    return StubProfiler.GetAdjustRequestTimeStamp()


def get_adjust_done_time_stamp():
    return StubProfiler.GetAdjustDoneTimeStamp()
