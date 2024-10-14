# raw tvm
import os
import subprocess
import tvm 
from tvm import relay
from tvm.relay.backend import Executor
import pathlib
import onnx
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import tempfile
from tvm.driver import tvmc
import tarfile


# start global
enc = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

#NOTE max token len is 512!
batch_size = 1
token_len = 64
pesudo_text = np.random.randint(101, enc.vocab_size - 1000, token_len) # in case out of bound
pesudo_text[0] = 101 # [CLS]
pesudo_text[-1] = 102 # [SEP]
pesudo_text[token_len // 2 - 1] = 102 # [SEP]

indexed_tokens = pesudo_text
segments_ids = [0 for _ in range(token_len // 2)] + [1 for _ in range(token_len // 2)]

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = tuple([tokens_tensor, segments_tensors])
# end global
model_name = 'distilbert_base'

print(f"### {enc.vocab_size}")


model = DistilBertModel.from_pretrained("distilbert-base-uncased")
print(dummy_input)
torch.onnx.export(model, dummy_input, f"{model_name}.onnx", verbose=True,
                    input_names=["input_ids", "attention_mask"], output_names=["output"], export_params=True,)


if 'COLSYS_TENSORRT_HOME' in os.environ:
    tensorrt_home = os.environ['COLSYS_TENSORRT_HOME']
    amd64_home = f'{tensorrt_home}/targets/x86_64-linux-gnu'
    envs = os.environ.copy()
    if 'LD_LIBRARY_PATH' in envs:
        envs['LD_LIBRARY_PATH'] = f'{amd64_home}/lib:{envs["LD_LIBRARY_PATH"]}'
    else:
        envs['LD_LIBRARY_PATH'] = f'{amd64_home}/lib'
    subprocess.run([f'{amd64_home}/bin/trtexec', f'--onnx={model_name}.onnx', f'--saveEngine={model_name}.plan'], 
        env=envs)