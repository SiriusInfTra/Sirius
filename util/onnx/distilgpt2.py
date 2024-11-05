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
from transformers import DistilBertTokenizer, DistilBertModel, GPT2Model, GPT2Tokenizer
import tempfile
from tvm.driver import tvmc
import tarfile

platform = 'v100'

model_store = "server/models"
tmp_dir = tempfile.gettempdir()

# start global
enc = GPT2Tokenizer.from_pretrained('distilgpt2')

#NOTE max token len is 512!
batch_size = 1
token_len = 64
pesudo_text = np.random.randint(101, enc.vocab_size - 1000, token_len) # in case out of bound
pesudo_text[0] = 101 # [CLS]
pesudo_text[-1] = 102 # [SEP]
pesudo_text[token_len // 2 - 1] = 102 # [SEP]

indexed_tokens = pesudo_text
# segments_ids = [0 for _ in range(token_len // 2)] + [1 for _ in range(token_len // 2)]
segments_ids = [1 for _ in range(token_len)]

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = tuple([tokens_tensor, segments_tensors])
# end global

model_name = "distilgpt2"
print(f"### {enc.vocab_size}")

#NOTE Generate new trace
# model = DistilBertModel.from_pretrained("distilbert-base-uncased")
class GPT2NoKVCache(GPT2Model):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs, use_cache=False)
        # Only return the last hidden state
        return outputs.last_hidden_state
model = GPT2Model.from_pretrained("distilgpt2")

print(dummy_input)
torch.onnx.export(model, tokens_tensor, f"{model_name}.onnx", verbose=True,
                    input_names=["input_ids",], output_names=["output"], export_params=True,)



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