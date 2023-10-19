## Build system  

1. Prepare the a new conda environment or use docker container, cuda version is `11.6`, cudnn version is `8.4`. Install `boost`, `cmake>=3.24` and `ninja`, `gcc` should have `c++17` support.

 
2. Clone and build [tvm](https://ipads.se.sjtu.edu.cn:1312/infer-train/tvm) for inference, and [pytorch](https://ipads.se.sjtu.edu.cn:1312/infer-train/pytorch) for training. Build [torchvision](https://github.com/pytorch/vision/tree/v0.13.1) to avoid symbol issues. Note CUDA backend should be enabled. Pay attention to pytorch `GLIBCXX_USE_CXX11_ABI` flag, which may cause ABI issues. 

3. Install python dependencies `cython`, `numpy`, `onnxruntime`, `torchvision` and etc.

4. Set `TVM_HOME` environment and configure cmake. For conda env, execute `export CMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib/python3.xx/site-packages/torch/share/cmake` to find pytorch.

```
cmake -DCMAKE_BUILD_TYPE=Release/Debug \
      -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCONDA_PREFIX=${CONDA_PREFIX} \
      -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib/python3.xx/site-packages/torch/share/cmake \
      -B build -G Ninja
cmake --build build --config Release/Debug
```

## Run system

### setup model store

models are stored at `./models`, as following. The model have a directory of the tvm compiled model (`mod.josn`, `mod.params` and `mod.so`)

```
├── config
├── mnist
│   ├── mod.json
│   ├── mod.params
│   └── mod.so
├── resnet152
│   ├── mod.json
│   ├── mod.params
│   └── mod.so
...
```

`config` is used to configure model workers, `path` is directory name of the model, `device` should be cuda, `batch-size` should be consistent with tvm compilation. `num-worker` is the default value for the number of model workers. To simulate `n` models, add `[n]` after model name, such as `resnet152[5]`.

```
resnet152
  path        resnet152
  device      cuda
  batch-size  4
  num-worker  1
```

### launch server

```
GLOG_logtostderr=1 ./build/colserve
```

Options
```
-m,--mode TEXT:{normal,task-switch-l1,task-switch-l2,task-switch-l3,colocate-l2}, server mode, see detail in server/config.h, default is normal
--use-sta, use shared tensor allocator, default is 1           
-p,--port, gRPC server port, default is 8080
```


### launch client/benchmark

```
GLOG_logtostderr=1 ./build/hybrid_workload \
  -p 8080 \ # gRPC port
  -d 30   \ # request duration in seconds
  -c 16   \ # the number of concurrency
  --infer --infer-model resnet152 \
  --train --train-model resnet --num-epoch 5 --batch-size 16 \
  --show-result 10 \ # show first 10 elem in result tensor
  -v 1      # verbose
```

See help for details.