# raw tvm
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer
import tempfile

import trt_converter
from trt_converter import TrtModelConfig

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

class GPT2NoKVCache(GPT2Model):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs, use_cache=False)
        # Only return the last hidden state
        return outputs.last_hidden_state
model = GPT2Model.from_pretrained("distilgpt2")

torch.onnx.export(
    model, tokens_tensor, f"{model_name}.onnx", verbose=True,
    input_names=["input_ids",], output_names=["output"], export_params=True,
    dynamic_axes={
    "input_ids": {0: "batch_size"},
    "output": {0: "batch_size"}}
)


trt_converter.convert_to_tensorrt("distilgpt2", 4, TrtModelConfig.DistilGPT2)