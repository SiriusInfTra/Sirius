# cython: c_string_type=unicode, c_string_encoding=utf8
from .ctrl_stub cimport *
from libcpp.string cimport string
from cpython.ref cimport PyObject


cdef extern from "<csrc/util.h>" namespace "torch_col":
    cpdef long get_unix_timestamp()
    cpdef long get_unix_timestamp_us()


cdef class PyMemoryQueue:
    cdef MemoryQueue[CtrlMsgEntry]* _queue

    def __cinit__(self, str name, bint is_server = False):
        self._queue = new MemoryQueue[CtrlMsgEntry](name.encode(), is_server)
      
    def put(self, PyCtrlMsgEntry entry):
        self._queue.Put(entry._entry)

    def timed_get(self, size_t timeout_ms):
        x = PyCtrlMsgEntry(0, CtrlEvent.kNumEvent, -1)
        if self._queue.TimedGet(x._entry, timeout_ms):
            return x
        else:
            return None

    def __dealloc__(self):
        del self._queue


cdef extern from "<common/controlling.h>" namespace "colserve::ctrl":
    cpdef enum class CtrlEvent(int):
        # status event
        kTrainStart,
        kTrainEnd,
        kInterruptTrainDone,
        kResumeTrainDone,
        kColocateAdjustL1Done,
        kColocateAdjustL2Done,
        
        kReportBatchSize,

        # cmd event: switch mode
        kInterruptTrain,
        kResumeTrain,
        # cmd event: colocate mode
        kColocateAdjustL1,
        kColocateAdjustL2,
        kInferExit, # train adjust back

        kInferenceWorkloadDone,

        kNumEvent,


cdef class PyCtrlMsgEntry:
    cdef CtrlMsgEntry _entry
    
    def __cinit__(self, unsigned long long id, CtrlEvent cmd, int value):
        self._entry = CtrlMsgEntry(id, int(cmd), value)

    @property
    def event(self):
        return CtrlEvent(self._entry.event)

    @property
    def id(self):
        return self._entry.id

    @property
    def value(self):
        return self._entry.value

    def __repr__(self):
        return "PyCtrlMsgEntry(id={}, event={}, value={})".format(self.id, str(self.event), self.value)


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
        return self._stub.Cmd()
    @cmd.setter
    def cmd(self, cmd):
        if cmd is None:
            self._stub.Cmd(-1)
        else:
            self._stub.Cmd(cmd)
    
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
        return self._stub.Cmd()

    @property
    def target_batch_size(self):
        return self._stub.TargetBatchSize()

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


def is_kill_batch_on_recv():
    global kill_batch_on_recv
    return kill_batch_on_recv



def get_adjust_request_time_stamp():
    return StubProfiler.GetAdjustRequestTimeStamp()


def get_adjust_done_time_stamp():
    return StubProfiler.GetAdjustDoneTimeStamp()
