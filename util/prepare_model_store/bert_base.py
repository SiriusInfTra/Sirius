# raw tvm
import tvm 
from tvm import relay
from tvm.relay.backend import Executor
import pathlib
import onnx
import numpy as np
import torch
from transformers import BertModel, BertTokenizer, BertConfig
import tempfile

model_store = "server/models"
tmp_dir = tempfile.gettempdir()

# start global
enc = BertTokenizer.from_pretrained("bert-base-uncased")

#NOTE max token len is 512!
batch_size = 1
token_len = 128
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

def get_onnx():
    #NOTE Generate new trace
    model = BertModel.from_pretrained("bert-base-uncased")
    print(dummy_input)
    torch.onnx.export(model, dummy_input, f"{tmp_dir}/bert-base.onnx", verbose=True,
                      input_names=["input_ids", "attention_mask"], output_names=["output"], export_params=True)

def tvm_compile():
    onnx_model = onnx.load('{}/bert-base.onnx'.format(tmp_dir))
    shape_dict = {'input_ids' : [1, token_len], 'attention_mask':[1, token_len]}
    mod_bert, params_bert = relay.frontend.from_onnx(onnx_model, shape_dict)
    
    # compile module
    with tvm.transform.PassContext(opt_level=3):
        executor_factory = relay.build(mod_bert, target='cuda', executor=Executor("graph"), params=params_bert)
    
    model_store_path = f'{model_store}/bert-base-b{batch_size}'
    lib_name = "mod.so"
    graph_module_name = "mod.json"
    params_name = "mod.params"

    executor_factory.get_lib().export_library(f'{lib_name}')
    pathlib.Path(model_store_path).mkdir(parents=True, exist_ok=True)
    with open(f'{model_store_path}/{graph_module_name}', "w") as graph_file:
        graph_file.write(executor_factory.get_graph_json())
    with open(f'{model_store_path}/{params_name}', "wb") as params_file:
        params_file.write(relay.save_param_dict(executor_factory.get_params()))


if __name__ == '__main__':
    get_onnx()
    tvm_compile()