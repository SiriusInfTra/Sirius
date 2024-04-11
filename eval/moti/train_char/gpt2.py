import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForPreTraining, BertConfig, BertForTokenClassification
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForSequenceClassification
from transformers import GPT2LMHeadModel, OPTForCausalLM
from torch.autograd.profiler import record_function

import numpy as np
import time
import argparse


g_enable_cuda_sync = True
def cuda_sync():
    global g_enable_cuda_sync
    if not g_enable_cuda_sync:
        pass
    torch.cuda.synchronize()

class IntermMemoryStat:
    _model_param_ptr_set = set()
    _interm_memory_ptr_set = set()
    _nbytes_acc = 0
    
    @staticmethod
    def reset():
        IntermMemoryStat._interm_memory_ptr_set = set()
        IntermMemoryStat._nbytes_acc = 0

    @staticmethod
    def get():
        return IntermMemoryStat._nbytes_acc / 1024 / 1024 / 1024
    
    @staticmethod
    def add(x: torch.Tensor):
        if x.data_ptr() in IntermMemoryStat._interm_memory_ptr_set:
            return
        if x.data_ptr() in IntermMemoryStat._model_param_ptr_set:
            return
        IntermMemoryStat._interm_memory_ptr_set.add(x.data_ptr())
        IntermMemoryStat._nbytes_acc += x.storage().nbytes()

    @staticmethod
    def model_param_add(x: torch.Tensor):
        if x.data_ptr() in IntermMemoryStat._model_param_ptr_set:
            return
        IntermMemoryStat._model_param_ptr_set.add(x.data_ptr())


def register_saved_tensor_hook():
    def pack_hook(x):
        IntermMemoryStat.add(x)
        return x
    def unpack_hook(x):
        return x
    torch._C._autograd._push_saved_tensors_default_hooks(pack_hook, unpack_hook)


def gpu_memory_usage(device_id):
    free, total = torch.cuda.mem_get_info(device_id)
    return (total - free) / 1024 / 1024 / 1024


def allocated_memory(device_id):
    # return (total - free) / 1024 / 1024 / 1024
    # free, total = torch.cuda.mem_get_info(device_id)
    return torch.cuda.memory_reserved(device_id) / 1024 / 1024 / 1024

def train(prof=None):
    seq_len = 256
    dataset_size = 128 * 10
    dummy_data = {
        "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
        # "labels": np.random.randint(0, 1, (dataset_size)),
    }
    dummy_data['labels'] = dummy_data['input_ids']
    dataset = Dataset.from_dict(dummy_data)
    dataset.set_format("pt")
    
    # batch_size = 128
    # batch_size = 64
    # batch_size = 32
    batch_size = 16
    # batch_size = 2
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    # model = OPTForCausalLM.from_pretrained("facebook/opt-350m")

    # model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)
    # model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
    model = model.cuda(0)
    for param in model.parameters():
        IntermMemoryStat.model_param_add(param)

    optimizer = torch.optim.SGD(model.parameters(), 0.1, 
                                momentum=0.9, weight_decay=1e-4)
    
    # torch.cuda.memory._record_memory_history(enabled=True)

    # eval_batch_size = [128, 64, 32, 16, 8, 4]
    # eval_batch_size = [64, 32, 16, 8, 4]
    # eval_batch_size = [32, 16, 8, 4]
    # eval_batch_size = [24, 16, 8, 4]
    # eval_batch_size = [16, 8, 4, 2, 1]
    eval_batch_size = [batch_size, ]
    epoch_micro_batch_size = [batch_size, ] # warmup
    for bs in eval_batch_size:
        epoch_micro_batch_size.extend([bs] * 3)

    print(batch_size)
    print(epoch_micro_batch_size)

    scaler = torch.cuda.amp.GradScaler()

    model.train()

    for epoch, mbs in enumerate(epoch_micro_batch_size):
        torch.cuda.reset_max_memory_allocated(0)
        interm_memorys = []
        compute_grad_times = []
        update_param_times = []
        batch_times = []

        epoch_begin_time = time.time()
        for b, batch in enumerate(dataloader):
            batch_begin = time.time()

            # prof.step()

            inputs = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=False)
            for mb in range(batch_size // mbs):
                IntermMemoryStat.reset()
                with torch.cuda.amp.autocast(cache_enabled=False):
                    outputs = model(**{k: v[mb * mbs: (mb + 1) * mbs] 
                                    for k, v in inputs.items()})
                    loss = outputs.loss
                scaler.scale(loss).backward()
                interm_memorys.append(IntermMemoryStat.get())
            cuda_sync()
            compute_grad_end = time.time()  

            scaler.step(optimizer)
            scaler.update()
            cuda_sync()
            update_param_end = time.time()

            batch_end = time.time()

            batch_times.append(batch_end - batch_begin)
            compute_grad_times.append(compute_grad_end - batch_begin)
            update_param_times.append(update_param_end - compute_grad_end)
        epoch_end_time = time.time()

        epoch_time = epoch_end_time - epoch_begin_time
        thpt = len(dataset) / epoch_time

        model_memory = 0
        optimizer_memory = 0
        # calculate model_memory and optimizer_memory here
        for param in model.parameters():
            model_memory += param.storage().nbytes()
        for buffer in model.buffers():
            buffer: torch.Tensor
            model_memory += buffer.storage().nbytes()
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    optimizer_memory += v.numel() * v.element_size()

        persist_memory = (model_memory + optimizer_memory) / 1024 / 1024 / 1024

        # print(torch.cuda.memory._snapshot())
            
        if epoch > 0:
            actual_peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            interm_memory = np.mean(interm_memorys)
            print(f'''Epoch {epoch} batch size {mbs} time: {epoch_time:.2f} thpt {thpt:.2f}
    GPU memory                  {gpu_memory_usage(0):.2f} GiB
    torch cached memory         {allocated_memory(0):.2f} GiB 
    actual peak memory          {actual_peak_memory:.2f} GiB
    interm memory               {interm_memory:.2f} GiB
    persist memory              {persist_memory:.2f} GiB
    current allocated memory    {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024:.2f} GiB
    other memory                {actual_peak_memory - interm_memory - persist_memory:.2f} GiB

    batch time                  {np.mean(batch_times) * 1000:.2f} ms
    compute grad time           {np.mean(compute_grad_times) * 1000:.2f} ms
    update param time           {np.mean(update_param_times) * 1000:.2f} ms
''')
        else:
            print(f'warmup epoch {epoch} time: {epoch_time:.2f} thpt {thpt:.2f}')



    # print(f'persist memory {persist_memory}')

# def trace_handler(prof: torch.profiler.profile):
#    # Prefix for file names.
# #    host_name = socket.gethostname()
# #    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
# #    file_prefix = f"{host_name}_{timestamp}"

#    # Construct the trace file.
#    prof.export_chrome_trace(f"gpt2-profile-{16}.json.gz")

#    # Construct the memory timeline file.
# #    prof.export_memory_timeline(f"gpt2-profile.html", device="cuda:0")
#     # prof.exp

def main():
    parser = argparse.ArgumentParser('Train Resnet152')    
    args = parser.parse_args()
    register_saved_tensor_hook()


    # with torch.profiler.profile(
    #     activities=[
    #        torch.profiler.ProfilerActivity.CPU,
    #        torch.profiler.ProfilerActivity.CUDA,
    #    ],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    #     on_trace_ready=trace_handler,

    # ) as prof:
        # train(prof)
    train(None)


if __name__ == '__main__':
    main()