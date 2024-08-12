from enum import Enum
import abc
import contextlib
import time
import os

import torch
import torch_col
from torch_col._C import HookMode
from .util import TrainMode, EventManager, MemoryPool
from . import xsched
from typing import List

__dummy_adjust = False

def register_saved_tensor_hook():
    def pack_hook(x):
        torch_col.tag_interm_memory(x)
        return x
    def unpack_hook(x):
        return x
    
    global __dummy_adjust
    if not __dummy_adjust:
        torch._C._autograd._push_saved_tensors_default_hooks(pack_hook, unpack_hook)


# class HookMode(Enum):
#     NONE = 'none'
#     SYNC = 'sync'
#     # XSCHED_ASYNC_SIGNAL = 'xsched-async-signal'  
#     XSCHED_SYNC = 'xsched-sync'
#     XSCHED_SYNC2 = 'xsched-sync2'
    
#     def use_xsched(self):
        # return self in {HookMode.XSCHED_SYNC, HookMode.XSCHED_SYNC2}


class HookABC(abc.ABC):
    def __init__(self, train_mode: TrainMode, hook_mode: HookMode, num_epoch: int, batch_size: int):
        self.train_mode = train_mode
        self.hook_mode = hook_mode
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self._stub = None
        self._grad_fn = None

    @abc.abstractmethod
    @contextlib.contextmanager
    def steps_no_interrupt(self):
        yield

    '''
        register forwards and backwards hook
    '''
    @abc.abstractmethod
    def register_pytorch_hook(self, module_list: List[torch.nn.Module]):
        pass
        
    def release_and_reply(self):
        if self.train_mode.is_colocate():
            return self.adjust()
        if self.train_mode.is_taskswitch():
            return self.switch()
        pass
    
    @abc.abstractmethod
    def report_batch_size(self, batch_size):
        pass
    
    @property
    @abc.abstractmethod
    def target_batch_size(self):
        pass
    
    @abc.abstractmethod
    def train_start(self):
        pass

    @abc.abstractmethod
    def train_end(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass
    
    @abc.abstractmethod
    def can_exit_after_infer_worklaod_done(self):
        pass

    @classmethod
    def register_fbward_hook(cls, module: torch.nn.Module, fwd_hood, bwd_hook):
        if not torch_col.is_enable_fbward_hook():
            return
        if len(list(module.children())) == 0:
            module.register_forward_hook(fwd_hood)
            module.register_backward_hook(bwd_hook)
        else:
            for child in module.children():
                cls.register_fbward_hook(child, fwd_hood, bwd_hook)



class SwitchL1Exception(Exception):
    pass

class SwitchHook(HookABC):
    def __init__(self, train_mode:TrainMode, hook_mode: HookMode, num_epoch: int, batch_size: int) -> None:
        assert train_mode in {TrainMode.TASKSWITCH_L1, TrainMode.TASKSWITCH_L0}
        super().__init__(train_mode, hook_mode, num_epoch, batch_size)
        self._stub = torch_col.PySwitchStub()
        self._grad_fn = []

    @contextlib.contextmanager
    def steps_no_interrupt(self):
        if self.train_mode == TrainMode.TASKSWITCH_L0:
            yield
            return
        # TrainMode.TASKSWITCH_L1
        if self.hook_mode.use_xsched():
            while xsched.get_xqueue_size() != 0:
                if self._stub.cmd == torch_col.CtrlEvent.kInterruptTrain:
                    xsched.kill_batch()
                    raise SwitchL1Exception('before_critical_section')
        torch.cuda.current_stream().synchronize()
        if self._stub.cmd == torch_col.CtrlEvent.kInterruptTrain:
            raise SwitchL1Exception('before_critical_section')
        # make sure we will not interrupt step
        self._stub.StepsNoInteruptBegin()
        yield
        torch.cuda.current_stream().synchronize()
        self._stub.StepsNoInteruptEnd()

        # during critical section executing, kill batch request may be sent
        if self._stub.cmd == torch_col.CtrlEvent.kInterruptTrain:
            # since we sync, no kernel is executing
            self.switch()

    def register_pytorch_hook(self, module_list: List[torch.nn.Module]):
        if self.train_mode == TrainMode.TASKSWITCH_L0:
            return
        if self.train_mode == TrainMode.TASKSWITCH_L1 and self.hook_mode == HookMode.XSCHED_SYNC2:
            print("SetUpTorchColEngine")
            self._stub.EnableTorchColEngine()
        else:
            for module in module_list:
                HookABC.register_fbward_hook(module, self.get_fwd_hook(), self.get_bwd_hook())
            if torch_col.is_release_interm_memory_v2():
                torch_col.register_saved_tensor_hook()

    def get_fwd_hook(self):
        # def hook(module, input, output):
        #     torch.cuda.synchronize()
        #     self._grad_fn.append(output._grad_fn)
        #     if self._stub.cmd == torch_col.CtrlEvent.kInterruptTrain:
        #         raise SwitchL1Exception("[Task Switch]")
        #     elif self._stub.cmd == torch_col.CtrlEvent.kResumeTrain:
        #         self._stub.cmd = None
        # match self.hook_mode:
        if self.hook_mode == HookMode.SYNC:
            if self.train_mode == TrainMode.TASKSWITCH_L1:
                def hook(module, input, output):
                    torch.cuda.synchronize()
                    if torch_col.is_release_interm_memory_v1():
                        self._grad_fn.append(output.grad_fn)
                    if self._stub.cmd == torch_col.CtrlEvent.kInterruptTrain:
                        raise SwitchL1Exception("[Task Switch SYNC FWD]")
                    elif self._stub.cmd == torch_col.CtrlEvent.kResumeTrain:
                        # self._stub.cmd = None
                        pass
            elif self.train_mode == TrainMode.TASKSWITCH_L0:
                raise Exception('task switch l0 in cpp workld')
        elif self.hook_mode == HookMode.XSCHED_SYNC:
            if self.train_mode == TrainMode.TASKSWITCH_L1:
                def hook(module, input, output):
                    if torch_col.is_release_interm_memory_v1():
                        self._grad_fn.append(output.grad_fn)
                    if self._stub.cmd == torch_col.CtrlEvent.kInterruptTrain:
                        xsched.kill_batch()
                        raise ColocateAdjustL1Exception("[Task Switch XSCHED_SYNC FWD]")
                    elif self._stub.cmd == torch_col.CtrlEvent.kResumeTrain:
                        # self._stub.cmd = None
                        pass
            elif self.train_mode == TrainMode.TASKSWITCH_L0:
                raise Exception('task switch l0 in cpp workld')
        else:
            raise RuntimeError(f"Unsupported hook_mode: {self.hook_mode.name}")
        return hook
    
    def get_bwd_hook(self):
        # def hook(module, grad_input, grad_output):
        #     torch.cuda.synchronize()
        #     if self._stub.cmd == torch_col.CtrlEvent.kInterruptTrain:
        #         raise SwitchL1Exception("[Task Switch]")
        #     elif self._stub.cmd == torch_col.CtrlEvent.kResumeTrain:
        #         self._stub.cmd = None
        # return hook
        # match self.hook_mode:
        if self.hook_mode == HookMode.SYNC:
            if self.train_mode == TrainMode.TASKSWITCH_L1:
                def hook(module, grad_input, grad_output):
                    torch.cuda.synchronize()
                    if self._stub.cmd == torch_col.CtrlEvent.kInterruptTrain:
                        raise SwitchL1Exception("[Task Switch BWD]")
                    elif self._stub.cmd == torch_col.CtrlEvent.kResumeTrain:
                        # self._stub.cmd = None
                        pass
            elif self.train_mode == TrainMode.TASKSWITCH_L0:
                raise Exception('task switch l0 in cpp workld')
        elif self.hook_mode == HookMode.XSCHED_SYNC: # use this for dummy adjust test
            if self.train_mode == TrainMode.TASKSWITCH_L1:
                def hook(module, grad_input, grad_output):
                    # print(f'{event_name}: {self._stub.cmd}')
                    if self._stub.cmd == torch_col.CtrlEvent.kInterruptTrain:
                        xsched.kill_batch()
                        raise SwitchL1Exception("[Task Switch BWD]")
                    elif self._stub.cmd == torch_col.CtrlEvent.kResumeTrain:
                        # self._stub.cmd = None
                        pass
            elif self.train_mode == TrainMode.TASKSWITCH_L0:
                raise Exception('task switch l0 in cpp workld')
        else:
            raise RuntimeError(f"Unsupported hook_mode: {self.hook_mode.name}")
        return hook

    def switch(self):
        # match self.train_mode:
        if self.train_mode == TrainMode.TASKSWITCH_L1:
            return self.switch_l1()
        else:
            raise RuntimeError(f"Unsupported train_mode: {self.train_mode.name}")

    def switch_l1(self):
        # decouple kill batch and reclaim memory
        t0 = time.time()
        if torch_col.is_release_interm_memory_v1():
            for fn in self._grad_fn:
                torch_col.release_grad_fn_saved_tensor(fn)
            self._grad_fn = []
        else:
            torch_col.release_interm_memory()
        old_gpu_mem = MemoryPool.get_memory_usage()
        MemoryPool.empty_cache()
        cur_gpu_mem = MemoryPool.get_memory_usage()
        succ = self.try_reply_interrupt()
        t1 = time.time()
        if succ:
            print(f'[Switch L1 {(t1-t0)*1e3:.1f} ms] target batch_size: {self.target_batch_size}, memory usage: {old_gpu_mem:.2f}GB -> {cur_gpu_mem:.2f}GB.', flush=True)

    def try_reply_interrupt(self):
        return self._stub.try_interrupt_train_done()

    @property
    def target_batch_size(self):
        return self.batch_size

    def report_batch_size(self, batch_size):
        self._stub.report_batch_size(batch_size)
    
    def train_start(self):
        self._stub.train_start()

    def train_end(self):
        if torch_col.is_enable_shared_tensor():
            torch_col.MemoryPool.empty_cache()
        return self._stub.train_end()

    def stop(self):
        self._stub.stop()

    def can_exit_after_infer_worklaod_done(self):
        return self._stub.can_exit_after_infer_worklaod_done()



class ColocateAdjustL1Exception(Exception):
    pass

class EngineColocateAdjustL1Exception(Exception):
    pass

class ColocateHook(HookABC):
    def __init__(self, train_mode: TrainMode, hook_mode: HookMode, num_epoch: int, batch_size: int):
        assert train_mode == TrainMode.COLOCATE_L1 or train_mode == TrainMode.COLOCATE_L2
        super().__init__(train_mode, hook_mode, batch_size, num_epoch)
        self._stub = torch_col.PyColocateStub(batch_size)
        self._grad_fn = []
        self._torch_stream = torch.cuda.current_stream()

    @contextlib.contextmanager
    def steps_no_interrupt(self):
        if self.train_mode == TrainMode.COLOCATE_L1:
            # before critical section, we need to make queued kernel as less as possible
            if self.hook_mode.use_xsched():
                while xsched.get_xqueue_size() != 0:
                    if self._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL1:
                        xsched.kill_batch()
                        raise ColocateAdjustL1Exception('before_critical_section')
            torch.cuda.current_stream().synchronize()
            if self._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL1:
                raise ColocateAdjustL1Exception('before_critical_section')
            # make sure we will not interrupt step
            self._stub.StepsNoInteruptBegin()
            yield
            torch.cuda.current_stream().synchronize()
            self._stub.StepsNoInteruptEnd()

            # during critical section executing, kill batch request may be sent
            if self._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL1:
                # since we sync, no kernel is executing
                with EventManager.record_duration_event('adjust_after_step'):
                    self.adjust()
        else:
            self._stub.StepsNoInteruptBegin()
            yield
            self._stub.StepsNoInteruptEnd()

    def register_pytorch_hook(self, module_list: List[torch.nn.Module]):
        if self.train_mode == TrainMode.COLOCATE_L1:
            # match self.hook_mode:
            if self.hook_mode == HookMode.SYNC:
                def fwd_hook(module, input, output):
                    with EventManager.record_duration_event('sync_fwd'):
                        torch.cuda.current_stream().synchronize()
                        if torch_col.is_release_interm_memory_v1():
                            self._grad_fn.append(output.grad_fn)
                        if self._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL1:
                            raise ColocateAdjustL1Exception('[Adjust SYNC FWD]')
                def bwd_hook(module, grad_input, grad_output):
                    with EventManager.record_duration_event('sync_bwd'):
                        torch.cuda.current_stream().synchronize()
                        if self._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL1:
                            raise ColocateAdjustL1Exception('[Adjust SYNC BWD]')
            elif self.hook_mode == HookMode.XSCHED_SYNC:
                def fwd_hook(module, input, output):
                    with EventManager.record_duration_event('xsched_sync_fwd'):
                        if torch_col.is_release_interm_memory_v1():
                            self._grad_fn.append(output.grad_fn)
                        if self._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL1:
                            xsched.kill_batch()
                            raise ColocateAdjustL1Exception('[Adjust XSCHED_SYNC BWD]')
                        # torch_col.cuda_memory_pool_reset_train_alloc_ms()
                def bwd_hook(module, grad_input, grad_output):
                    with EventManager.record_duration_event('xsched_sync_bwd'):
                        if self._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL1:
                            xsched.kill_batch()
                            raise ColocateAdjustL1Exception('[Adjust XSCHED_SYNC BWD]')
                        # torch_col.cuda_memory_pool_reset_train_alloc_ms()
            elif self.hook_mode == HookMode.XSCHED_SYNC2:
                print("SetUpTorchColEngine")
                self._stub.EnableTorchColEngine()
                return
            else:
                raise RuntimeError(f"Unsupported hook_mode: {self.hook_mode.name}")
            for module in module_list:
                HookABC.register_fbward_hook(module, fwd_hook, bwd_hook)
            if torch_col.is_release_interm_memory_v2():
                torch_col.register_saved_tensor_hook()
        else:
            pass
    
    # def async_adjust_mem(self):
    #     with self._block_signal_section():
    #         assert self.train_mode == TrainMode.COLOCATE_L1
    #         if self.hook_mode == HookMode.XSCHED_ASYNC_SIGNAL:
    #             if not self._async_killed_batch:
    #                 # if already killed, no need to do again
    #                 xsched.kill_batch(self._torch_stream)
    #                 xsched.disable_cuda_calls(self._torch_stream)
    #                 torch_col.cuda_memory_pool_disble_train_alloc()
    #                 self._async_killed_batch = True
    #         elif self.hook_mode == HookMode.XSCHED_SYNC:
    #             xsched.kill_batch()
            
    #         old_gpu_mem = MemoryPool.get_memory_usage()
    #         for fn in self._grad_fn:
    #             torch_col.release_grad_fn_saved_tensor(fn)
    #         self._grad_fn = []
    #         MemoryPool.empty_cache()
    #         self._stub.adjust_l1_done()
    #         curr_gpu_mem = MemoryPool.get_memory_usage()
    #         print(f'[Adjust] target batch_size: {self.target_batch_size}, memory usage: {old_gpu_mem:.2f}GB -> {curr_gpu_mem:.2f}GB.')

    def adjust(self):
        # match self.train_mode:
        if self.train_mode == TrainMode.COLOCATE_L1:
            return self.adjust_l1()
        elif self.train_mode == TrainMode.COLOCATE_L2:
            return self.adjust_l2()
        else:
            raise RuntimeError(f"Unsupported train_mode: {self.train_mode.name}")

    def adjust_l1(self):
        # decouple kill batch and reclaim memory
        # print(f'[Adjust L1] train alloc cost {torch_col.cuda_memory_pool_train_alloc_ms()}ms', flush=True)
        t0 = time.time()
        with EventManager.record_duration_event('adjust_l1'):
            old_cached_gpu_mem, old_allocate_gpu_mem = MemoryPool.get_memory_usage(), MemoryPool.get_allocated_memory()
            if torch_col.is_release_interm_memory_v1():
                for fn in self._grad_fn:
                    torch_col.release_grad_fn_saved_tensor(fn)
                self._grad_fn = []
            else:
                torch_col.release_interm_memory()
            t1 = time.time()
            MemoryPool.empty_cache()
            t2 = time.time()
            self._stub.adjust_l1_done()
            t3 = time.time()
            cur_cached_gpu_mem, cur_allocate_gpu_mem = MemoryPool.get_memory_usage(), MemoryPool.get_allocated_memory()
        t4 = time.time()
        print(f'[Rank {torch_col.get_train_rank()} | PID {os.getpid()} | {torch_col.get_unix_timestamp_us()/1000}]'
                + f' [Adjust L1 {(t1-t0)*1e3:.1f} | {(t2-t1)*1e3:.1f} | {(t3-t2)*1e3:.1f} | {(t4-t3)*1e3:.1f} ms]'
                + f' target batch_size: {self.target_batch_size},'
                + f' memory cached: {old_cached_gpu_mem:.2f}GB -> {cur_cached_gpu_mem:.2f}GB,'
                + f' memory allocated: {old_allocate_gpu_mem:.2f}GB -> {cur_allocate_gpu_mem:.2f}GB.', flush=True)

    def adjust_l2(self):
        t0 = time.time()
        with EventManager.record_duration_event('adjust_l2'):
            old_cached_gpu_mem, old_allocate_gpu_mem = MemoryPool.get_memory_usage(), MemoryPool.get_allocated_memory()
            old_pytorch_cached = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
            MemoryPool.empty_cache()
            self._stub.adjust_l2_done()
            cur_cached_gpu_mem, cur_allocate_gpu_mem = MemoryPool.get_memory_usage(), MemoryPool.get_allocated_memory()
            cur_pytorch_cached = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
        t1 = time.time()
        print(f'[{torch_col.get_unix_timestamp_us()/1000}] [Adjust L2 {(t1-t0)*1e3:.1f} ms] target batch_size: {self.target_batch_size},'
                + f' memory cached: {old_cached_gpu_mem:.2f}GB -> {cur_cached_gpu_mem:.2f}GB,'
                + f' memory allocated: {old_allocate_gpu_mem:.2f}GB -> {cur_allocate_gpu_mem:.2f}GB.'
                + f' pytorch allocated: {old_pytorch_cached:.2f}GB -> {cur_pytorch_cached:.2f}GB', flush=True)

    def report_batch_size(self, batch_size):
        self._stub.report_batch_size(batch_size)

    @property
    def target_batch_size(self):
        return self._stub.target_batch_size
    
    def train_start(self):
        self._stub.train_start()

    def train_end(self):
        if torch_col.is_enable_shared_tensor():
            torch_col.MemoryPool.empty_cache()
        self._stub.train_end()

    def stop(self):
        self._stub.stop()

    def can_exit_after_infer_worklaod_done(self):
        return self._stub.can_exit_after_infer_worklaod_done()


class DummyHook(HookABC):
    def __init__(self, train_mode: TrainMode, hook_mode: HookMode, num_epoch: int, batch_size: int):
        super().__init__(train_mode, hook_mode, batch_size, num_epoch)
        self._stub = torch_col.PyDummyStub()
        assert hook_mode == HookMode.NONE
    
    @contextlib.contextmanager
    def steps_no_interrupt(self):
        yield
    
    def check_async_killed_batch(self):
        pass
    
    def register_pytorch_hook(self, module_list: List[torch.nn.Module]):
        pass
    
    def report_batch_size(self, batch_size):
        pass
    
    @property
    def target_batch_size(self):
        return self.batch_size
    
    def train_start(self):
        self._stub.train_start()

    def train_end(self):
        self._stub.train_end()
    
    def stop(self):
        self._stub.stop()

    def can_exit_after_infer_worklaod_done(self):
        return self._stub.can_exit_after_infer_worklaod_done()

def get_hook(train_mode: TrainMode, hook_mode: HookMode, num_epoch: int, batch_size: int):
    if train_mode == TrainMode.NORMAL:
        return DummyHook(train_mode, hook_mode, num_epoch, batch_size)
    elif train_mode.is_colocate():
        return ColocateHook(train_mode, hook_mode, num_epoch, batch_size)
    elif train_mode.is_taskswitch():
        return SwitchHook(train_mode, hook_mode, num_epoch, batch_size)
