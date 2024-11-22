# raw tvm
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel

import trt_converter
from trt_converter import TrtModelConfig

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
torch.onnx.export(
    model, dummy_input, f"{model_name}.onnx", verbose=True,
    input_names=["input_ids", "attention_mask"], output_names=["output"], export_params=True,
    dynamic_axes={
    "input_ids": {0: "batch_size"},
    "attention_mask": {0: "batch_size"},
    "output": {0: "batch_size"}}
)


trt_converter.convert_to_tensorrt(model_name, 4, TrtModelConfig.DistilBert)