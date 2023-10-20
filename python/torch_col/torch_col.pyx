from .torch_col cimport MemoryQueue, Event, SwitchStub, ColocateStub, CtrlMsgEntry

cdef extern from "<csrc/control_stub.h>" namespace "torch_col":
    cpdef void ReleaseMempool()

cdef class PyCtrlMsgEntry:
    cdef CtrlMsgEntry _entry
    
    def __cinit__(self, unsigned long long id, int cmd):
        self._entry = CtrlMsgEntry(id, cmd)


cdef class PyMemoryQueue:
    cdef MemoryQueue[CtrlMsgEntry]* _queue

    def __cinit__(self, str name, bint is_server = False):
        self._queue = new MemoryQueue[CtrlMsgEntry](name.encode(), is_server)
      
    def put(self, PyCtrlMsgEntry entry):
        self._queue.Put(entry._entry)

    def timed_get(self, size_t timeout_ms):
        x = PyCtrlMsgEntry(0, -1)
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


    def __dealloc__(self):
        del self._stub