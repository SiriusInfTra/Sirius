from pycolserve cimport MemoryQueue, Event, SwitchStub

cdef class PyMemoryQueue:
    cdef MemoryQueue[int]* _queue

    def __cinit__(self, str name, bint is_server = False):
        self._queue = new MemoryQueue[int](name.encode(), is_server)
      
    def put(self, event):
        self._queue.Put(event)

    def timed_get(self, size_t timeout_ms):
        cdef int x = -1
        if self._queue.TimedGet(x, timeout_ms):
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

    def __cinit__(self):
        self._stub = new ColocateStub()

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