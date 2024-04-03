import tvm
from tvm.driver import tvmc
import numpy as np
from tvm.contrib import graph_executor
import time

model_name = "vit_s_16-b1"

input_data = np.fromfile('client/data/resnet/input-0.bin', dtype=np.float32)
input_data = input_data.reshape(1, 3, 224, 224)

lib = tvm.runtime.load_module(f"server/models/{model_name}/mod.so")
with open(f"server/models/{model_name}/mod.json") as f:
    module = graph_executor.create(f.read(), lib, tvm.cuda(0))
with open(f"server/models/{model_name}/mod.params", "rb") as f:
    module.load_params(f.read())

module.set_input("input", input_data)
module.run()

output = module.get_output(0).numpy()
output = output.reshape(-1)

print(output[:10])

for i in range(10):
    t0 = time.time()
    module.run()
    t1 = time.time()
    print(t1 - t0)