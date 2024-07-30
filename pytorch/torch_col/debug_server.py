import torch_col
import threading


class DebugServer:
    def __init__(self, train_world_size):
        self._cmd_ids = {}
        self._inf_tra_comm = torch_col.PyInfTraCommunicator(
            True, True, train_world_size)
        
        self._running = True
        self._thread = threading.Thread(target=self._recv_from_train)
        self._thread.start()

    def colocate_l1(self, batch_size: int):
        if torch_col.CtrlEvent.kColocateAdjustL1 not in self._cmd_ids:
            self._cmd_ids[torch_col.CtrlEvent.kColocateAdjustL1] = 1
        else:
            self._cmd_ids[torch_col.CtrlEvent.kColocateAdjustL1] += 1
        cmd_id = self._cmd_ids[torch_col.CtrlEvent.kColocateAdjustL1]

        msg = torch_col.PyCtrlMsgEntry(
            cmd_id, torch_col.CtrlEvent.kColocateAdjustL1, batch_size)
        
        self._inf_tra_comm.put_all_inf2tra(msg)
        # self._inf_tra_comm.put_inf2tra(msg, 0)

    def colocate_l2(self, batch_size: int):
        if torch_col.CtrlEvent.kColocateAdjustL2 not in self._cmd_ids:
            self._cmd_ids[torch_col.CtrlEvent.kColocateAdjustL2] = 1
        else:
            self._cmd_ids[torch_col.CtrlEvent.kColocateAdjustL2] += 1
        cmd_id = self._cmd_ids[torch_col.CtrlEvent.kColocateAdjustL2]

        msg = torch_col.PyCtrlMsgEntry(
            cmd_id, torch_col.CtrlEvent.kColocateAdjustL2, batch_size)
        self._cmd_mq.put_inf2tra(msg, 0)

    def infer_exit(self, batch_size: int):
        if torch_col.CtrlEvent.kInferExit not in self._cmd_ids:
            self._cmd_ids[torch_col.CtrlEvent.kInferExit] = 1
        else:
            self._cmd_ids[torch_col.CtrlEvent.kInferExit] += 1
        cmd_id = self._cmd_ids[torch_col.CtrlEvent.kInferExit]

        msg = torch_col.PyCtrlMsgEntry(
            cmd_id, torch_col.CtrlEvent.kInferExit, batch_size)
        # self._cmd_mq.put_inf2_tra(msg, 0)
        self._cmd_mq.put_all_inf2tra(msg)

    def _recv_from_train(self):
        while self._running:
            # msg = self._status_mq.timed_get_inf_tra(1, 0)
            msg = self._inf_tra_comm.timed_get_tra2inf(10, 0)
            if msg is None:
                continue
            if msg.event != torch_col.CtrlEvent.kReportBatchSize:
                print(f'timestamp {torch_col.get_unix_timestamp()}: {msg}')

    def stop(self):
        self._running = False