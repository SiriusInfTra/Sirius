# cython: c_string_type=unicode, c_string_encoding=utf8
from .torch_col cimport *
from libcpp.string cimport string
from cpython.ref cimport PyObject


def cuda_memory_pool_infer_usage():
    return CUDAMemPool.InferMemUsage()


def cuda_memory_pool_train_usage():
    return CUDAMemPool.TrainMemUsage()


def cuda_memory_pool_train_all_usage():
    return CUDAMemPool.TrainAllMemUsage()


def cuda_memory_pool_free_train_local():
    CUDAMemPool.FreeTrainLocals()


def cuda_memory_pool_train_alloc_ms(reset = False):
    ms = CUDAMemPool.TrainAllocMs()
    if reset:
        CUDAMemPool.ResetTrainAllocMs()
    return ms


def cuda_memory_pool_reset_train_alloc_ms():
    CUDAMemPool.ResetTrainAllocMs()


cdef extern from "<csrc/unix_time.h>" namespace "torch_col":
    cpdef long get_unix_timestamp()


def release_grad_fn_saved_tensor(grad_fn):
    cdef PyObject* obj = <PyObject*> grad_fn
    ReleaseGradFnSavedTensor(obj)


cdef extern from "<csrc/control_stub.h>" namespace "torch_col":
    cpdef void DumpMempoolFreeList(string filename)
    cpdef void DumpMempoolBlockList(string filename)


cdef extern from "<csrc/control_stub.h>":
    cdef int KillBatchOnRecv

def kill_batch_on_recv():
    global KillBatchOnRecv
    return KillBatchOnRecv


cdef extern from "<csrc/control_stub.h>" namespace "torch_col":
    cpdef enum class Event(int):
        # status 
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

        kNumEvent,

cdef class PyCtrlMsgEntry:
    cdef CtrlMsgEntry _entry
    
    def __cinit__(self, unsigned long long id, Event cmd, int value):
        self._entry = CtrlMsgEntry(id, int(cmd), value)

    @property
    def event(self):
        return Event(self._entry.event)

    @property
    def id(self):
        return self._entry.id

    @property
    def value(self):
        return self._entry.value

    def __repr__(self):
        return "PyCtrlMsgEntry(id={}, event={}, value={})".format(self.id, str(self.event), self.value)


cdef class PyMemoryQueue:
    cdef MemoryQueue[CtrlMsgEntry]* _queue

    def __cinit__(self, str name, bint is_server = False):
        self._queue = new MemoryQueue[CtrlMsgEntry](name.encode(), is_server)
      
    def put(self, PyCtrlMsgEntry entry):
        self._queue.Put(entry._entry)

    def timed_get(self, size_t timeout_ms):
        x = PyCtrlMsgEntry(0, Event.kNumEvent, -1)
        if self._queue.TimedGet(x._entry, timeout_ms):
            return x
        else:
            return None

    def __dealloc__(self):
        del self._queue


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
        self._stub.TryInterruptTrainDone()

    def report_batch_size(self, batch_size):
        self._stub.ReportBatchSize(batch_size)

    @property
    def cmd(self):
        return self._stub.Cmd()
    @cmd.setter
    def cmd(self, cmd):
        if cmd is None:
            self._stub.Cmd(-1)
        else:
            self._stub.Cmd(cmd)

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

    def __dealloc__(self):
        del self._stub