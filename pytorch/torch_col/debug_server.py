import torch_col
import threading


class DebugServer:
    def __init__(self):
        self._cmd_ids = {}
        self._cmd_mq = torch_col.PyMemoryQueue('cmd-ctrl', True)
        self._status_mq = torch_col.PyMemoryQueue('status-ctrl', True)

        self._running = True
        self._thread = threading.Thread(target=self._recv_status)
        self._thread.start()

    def colocate_l1(self, batch_size: int):
        if torch_col.CtrlEvent.kColocateAdjustL1 not in self._cmd_ids:
            self._cmd_ids[torch_col.CtrlEvent.kColocateAdjustL1] = 1
        else:
            self._cmd_ids[torch_col.CtrlEvent.kColocateAdjustL1] += 1
        cmd_id = self._cmd_ids[torch_col.CtrlEvent.kColocateAdjustL1]
        self._cmd_mq.put(torch_col.PyCtrlMsgEntry(cmd_id, torch_col.CtrlEvent.kColocateAdjustL1, batch_size))

    def colocate_l2(self, batch_size: int):
        if torch_col.CtrlEvent.kColocateAdjustL2 not in self._cmd_ids:
            self._cmd_ids[torch_col.CtrlEvent.kColocateAdjustL2] = 1
        else:
            self._cmd_ids[torch_col.CtrlEvent.kColocateAdjustL2] += 1
        cmd_id = self._cmd_ids[torch_col.CtrlEvent.kColocateAdjustL2]
        self._cmd_mq.put(torch_col.PyCtrlMsgEntry(cmd_id, torch_col.CtrlEvent.kColocateAdjustL2, batch_size))

    def infer_exit(self, batch_size: int):
        if torch_col.CtrlEvent.kInferExit not in self._cmd_ids:
            self._cmd_ids[torch_col.CtrlEvent.kInferExit] = 1
        else:
            self._cmd_ids[torch_col.CtrlEvent.kInferExit] += 1
        cmd_id = self._cmd_ids[torch_col.CtrlEvent.kInferExit]
        self._cmd_mq.put(torch_col.PyCtrlMsgEntry(cmd_id, torch_col.CtrlEvent.kInferExit, batch_size))

    def _recv_status(self):
        while self._running:
            msg = self._status_mq.timed_get(10)
            if msg is None:
                continue
            if msg.event != torch_col.CtrlEvent.kReportBatchSize:
                print(f'timestamp {torch_col.get_unix_timestamp()}: {msg}')

    def stop(self):
        self._running = False