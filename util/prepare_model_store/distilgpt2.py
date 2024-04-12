# raw tvm
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


print(f"### {enc.vocab_size}")

def get_onnx():
    #NOTE Generate new trace
    # model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model = GPT2Model.from_pretrained("distilgpt2")
    print(dummy_input)
    torch.onnx.export(model, tokens_tensor, f"{tmp_dir}/distilgpt2.onnx", verbose=True,
                      input_names=["input_ids",], output_names=["output"], export_params=True,)

def tvm_compile():
    shape_dict = {'input_ids' : [1, token_len],}
    tvmc_model = tvmc.load(f"{tmp_dir}/distilgpt2.onnx", 
                            shape_dict=shape_dict)

    # tune_records = "./distilgpt2-tune.json"
    # tvmc.tune(tvmc_model=tvmc_model, target='cuda', 
    #         tuning_records=tune_records, 
    #         enable_autoscheduler=False,
    #         port=9993)
    
    tune_records = f'util/prepare_model_store/distilgpt2-tune-{platform}.json'
    # tune_records = None
    tvmc.compile(tvmc_model=tvmc_model, target='cuda', package_path=f"{tmp_dir}/distilgpt2-tvm.tar", tuning_records=tune_records)

    model_store_path = f'{model_store}/distilgpt2-b{batch_size}' 
    pathlib.Path(model_store_path).mkdir(parents=True, exist_ok=True)
    tarfile.open(f"{tmp_dir}/distilgpt2-tvm.tar").extractall(model_store_path)


    # onnx_model = onnx.load('{}/distilgpt2.onnx'.format(tmp_dir))
    # mod_bert, params_bert = relay.frontend.from_onnx(onnx_model, shape_dict)
    
    # # compile module
    # with tvm.transform.PassContext(opt_level=3):
    #     executor_factory = relay.build(mod_bert, target='cuda', executor=Executor("graph"), params=params_bert)
    
    # model_store_path = f'{model_store}/distilgpt2-b{batch_size}'
    # lib_name = "mod.so"
    # graph_module_name = "mod.json"
    # params_name = "mod.params"

    # executor_factory.get_lib().export_library(f'{model_store_path}/{lib_name}')
    # pathlib.Path(model_store_path).mkdir(parents=True, exist_ok=True)
    # with open(f'{model_store_path}/{graph_module_name}', "w") as graph_file:
    #     graph_file.write(executor_factory.get_graph_json())
    # with open(f'{model_store_path}/{params_name}', "wb") as params_file:
    #     params_file.write(relay.save_param_dict(executor_factory.get_params()))


if __name__ == '__main__':
    get_onnx()
    tvm_compile()