# cython: c_string_type=unicode, c_string_encoding=utf8
# from .ctrl_stub cimport *
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

from cpython.ref cimport PyObject


cdef extern from "<torch_col/csrc/control_stub.h>" namespace "torch_col":
    cdef cppclass DummyStub:
        DummyStub() except +
        void Stop()
        void TrainStart()
        void TrainEnd()
        bint CanExitAfterInferWorkloadDone()
        void SetTrainFirstEpochDone()

    cdef cppclass SwitchStub:
        SwitchStub() except +
        int GetCmd()
        void SetCmd(int)
        void Stop()
        void TrainStart()
        void TrainEnd()
        void SetTrainFirstEpochDone()
        bint TryInterruptTrainDone(bool)
        void ReportBatchSize(int)
        void StepsNoInteruptBegin()
        void StepsNoInteruptEnd()
        void EnableTorchColEngine()
        bint CanExitAfterInferWorkloadDone()
        void SetKilledBatchRecover()
        void SetGlobalInterruptFlag(bool)
        bool PrepareResume()
        bool GetGlobalInterruptFlag()
        void SetGlobalHasBatchKilled(bool)
        bool GetGlobalHasBatchKilled()
        void TrainResumeDone()


    cdef cppclass ColocateStub:
        ColocateStub(int) except +
        void Stop()
        int GetCmd()
        void SetCmd(int)
        int GetTargetBatchSize()
        int GetUnpubTargetBatchSize()
        void ColocateAdjustL1Done()
        void ColocateAdjustL2Done()
        void TrainStart()
        void TrainEnd()
        void SetTrainFirstEpochDone()
        void ReportBatchSize(int)
        void StepsNoInteruptBegin()
        void StepsNoInteruptEnd()
        void EnableTorchColEngine()
        bint CanExitAfterInferWorkloadDone()
        void SetKilledBatchRecover()
        void SetKilledBatchReconfiged()

    cdef cppclass StubProfiler:
        @staticmethod
        vector[long] GetAdjustRequestTimeStamp()
        @staticmethod
        vector[long] GetAdjustDoneTimeStamp()


cdef class PyDummyStub:
    cdef DummyStub* _cppclass

    def __cinit__(self):
        self._cppclass = new DummyStub()

    def train_start(self):
        self._cppclass.TrainStart()

    def train_end(self):
        self._cppclass.TrainEnd()

    def stop(self):
        self._cppclass.Stop()

    def can_exit_after_infer_worklaod_done(self):
        return self._cppclass.CanExitAfterInferWorkloadDone()

    def set_train_first_epoch_done(self):
        self._cppclass.SetTrainFirstEpochDone()
    
    def __dealloc__(self):
        del self._cppclass


cdef class PySwitchStub:
    cdef SwitchStub* _cppclass

    def __cinit__(self):
        self._cppclass = new SwitchStub()

    def train_start(self):
        self._cppclass.TrainStart()

    def train_end(self):
        self._cppclass.TrainEnd()

    def stop(self):
        self._cppclass.Stop()

    def set_train_first_epoch_done(self):
        self._cppclass.SetTrainFirstEpochDone()

    def try_interrupt_train_done(self, barrier: bool):
        return self._cppclass.TryInterruptTrainDone(barrier)

    def report_batch_size(self, batch_size):
        self._cppclass.ReportBatchSize(batch_size)
    
    def StepsNoInteruptBegin(self):
        self._cppclass.StepsNoInteruptBegin()

    def StepsNoInteruptEnd(self):
        self._cppclass.StepsNoInteruptEnd()

    @property
    def cmd(self):
        return self._cppclass.GetCmd()
    @cmd.setter
    def cmd(self, cmd):
        if cmd is None:
            self._cppclass.SetCmd(-1)
        else:
            self._cppclass.SetCmd(cmd)
    
    def EnableTorchColEngine(self):
        self._cppclass.EnableTorchColEngine()

    def can_exit_after_infer_worklaod_done(self):
        return self._cppclass.CanExitAfterInferWorkloadDone()

    def set_killed_batch_recover(self):
        self._cppclass.SetKilledBatchRecover()

    def set_global_interrupt_flag(self, flag):
        self._cppclass.SetGlobalInterruptFlag(flag)

    def prepare_resume(self):
        return self._cppclass.PrepareResume()

    def get_global_interrupt_flag(self):
        return self._cppclass.GetGlobalInterruptFlag()
        
    def set_global_has_batch_killed(self, flag):
        self._cppclass.SetGlobalHasBatchKilled(flag)

    def get_global_has_batch_killed(self):
        return self._cppclass.GetGlobalHasBatchKilled()

    def train_resume_done(self):
        self._cppclass.TrainResumeDone()

    def __dealloc__(self):
        del self._cppclass


cdef class PyColocateStub:
    cdef ColocateStub* _cppclass

    def __cinit__(self, batch_size):
        self._cppclass = new ColocateStub(batch_size)

    def stop(self):
        self._cppclass.Stop()

    @property
    def cmd(self):
        return self._cppclass.GetCmd()

    @property
    def target_batch_size(self):
        return self._cppclass.GetTargetBatchSize()

    @property
    def unpub_target_batch_size(self):
        return self._cppclass.GetUnpubTargetBatchSize()

    def adjust_l1_done(self):
        self._cppclass.ColocateAdjustL1Done()

    def adjust_l2_done(self):
        self._cppclass.ColocateAdjustL2Done()

    def train_start(self):
        self._cppclass.TrainStart()

    def train_end(self):
        self._cppclass.TrainEnd()

    def set_train_first_epoch_done(self):
        self._cppclass.SetTrainFirstEpochDone()

    def report_batch_size(self, batch_size):
        self._cppclass.ReportBatchSize(batch_size)

    def StepsNoInteruptBegin(self):
        self._cppclass.StepsNoInteruptBegin()

    def StepsNoInteruptEnd(self):
        self._cppclass.StepsNoInteruptEnd()
    
    def can_exit_after_infer_worklaod_done(self):
        return self._cppclass.CanExitAfterInferWorkloadDone()

    def EnableTorchColEngine(self):
        self._cppclass.EnableTorchColEngine()

    def set_killed_batch_recover(self):
        self._cppclass.SetKilledBatchRecover()

    def set_killed_batch_reconfiged(self):
        self._cppclass.SetKilledBatchReconfiged()

    def __dealloc__(self):
        del self._cppclass


def get_adjust_request_time_stamp():
    return StubProfiler.GetAdjustRequestTimeStamp()


def get_adjust_done_time_stamp():
    return StubProfiler.GetAdjustDoneTimeStamp()
