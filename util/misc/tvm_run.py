import tvm
from tvm.driver import tvmc
import numpy as np
from tvm.contrib import graph_executor
import time

# model_name = "vit_s_16-b1"
# model_name = "resnet152"
# model_name = "densenet161-b1"
# model_name = "swin_t-b1-no-tune"
# model_name = "swin_t-b1"
model_name = "efficientnet_v2_s-b1"
# model_name = "inception_v3-b1"
# model_name = "distilbert_base-b1"
# model_name = "efficientvit_b2-b1"
# model_name = "distilgpt2-b1"
# model_name = "distilgpt-b1-no-tune"

input_data = np.fromfile('client/data/resnet/input-0.bin', dtype=np.float32)
input_data = input_data.reshape(1, 3, 224, 224)

# input_data = np.fromfile('client/data/inception/input-0.bin', dtype=np.float32)
# input_data = input_data.reshape(1, 3, 299, 299)

# input_data = np.fromfile(f"client/data/bert/input-0.bin", dtype=np.float32)
# input_data = input_data[:64].reshape(1, 64)
# mask_data = np.fromfile(f"client/data/bert/mask-0.bin", dtype=np.float32)
# mask_data = input_data[:64].reshape(1, 64)

lib = tvm.runtime.load_module(f"server/models/{model_name}/mod.so")
with open(f"server/models/{model_name}/mod.json") as f:
    module = graph_executor.create(f.read(), lib, tvm.cuda(0))
with open(f"server/models/{model_name}/mod.params", "rb") as f:
    module.load_params(f.read())

module.set_input("input", input_data)
# module.set_input("input_ids", input_data)
# module.set_input("attention_mask", mask_data)
# module.run()

output = module.get_output(0).numpy()
output = output.reshape(-1)

print(output[:10])

for i in range(10):
    t0 = time.time()
    module.run()
    t1 = time.time()
    print(t1 - t0)