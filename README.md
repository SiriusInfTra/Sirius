## Artifact for paper "<u>#734	Colocating ML Inference and Training with Fast GPU Memory Handover</u>"

<!-- Intro -->

[TOC]

## Project Structure

```bash
$ tree --dirsfirst  -L 2 .
├── client                      
├── cmake                       # cmake helper files
├── common                      # common libraries for infer/train  
├── environment                 # Docker and conda environment files
├── eval
│   ├── runner                  # automic evaluation runner
│   └── ...                     # evaluation scripts of test cases 
├── log                         # running logs       
├── proto                       # grpc proto
├── pytorch                     # pytorch plugin
├── scripts                    
├── server                      # inference server 
│   ├── models                  # contains inference models
│   └── ... 
├── train                       # pytorch training scripts
├── third_party/mpool...        # gpu memory pool
└── ...
```

## Hardware Requirements

- 4 * NVIDIA V100 (16GB)
- 1 * NVIDIA A100 (80GB)

## Build and Install 

### Using Docker Image

**Option 1: Pull from Docker Hub**



**Option 2: Build from Dockerfile**



### Compile From Source (using conda)

**Software Requirements**: `cmake>=3.24`, `gcc>=9.4`, `nvcc>=11.6`, `ninja`

**Create Environment and Build System**:

1. Prepare a new conda environment and install python packages

```bash
conda create -n colserve python=3.12
conda install -y conda-forge::python-devtools nvitop conda-forge::c-ares
pip install -r environment/requirements.txt
```

2. Install `Boost>=1.80` from compiling source (boost installed from apt/conda may require higher version of gcc).  

```bash
export BOOST_HOME=/path/to/install/boost
./scripts/install_boost.sh $BOOST_HOME
```

2. Clone and build [tvm](https://ipads.se.sjtu.edu.cn:1312/infer-train/tvm) for inference; [pytorch](https://ipads.se.sjtu.edu.cn:1312/infer-train/pytorch) and [torchvision](https://github.com/pytorch/vision/tree/v0.13.1) for training. Note CUDA backend should be enabled. Pay attention to pytorch `GLIBCXX_USE_CXX11_ABI` flag, which may cause ABI issues. To accelerate building, set `TORCH_CUDA_ARCH_LIST` flag to gpu computing capability, e.g., `TORCH_CUDA_ARCH_LIST=7.0`.

3. Set `TVM_HOME` environment, run `echo $TVM_HOME` and `echo $CONDA_REFIX` to check. Then configure cmake.

```
export COLSYS_HOME=$(pwd)
export TVM_HOME=/path/to/tvm
export TORCH_HOME=/path/to/pytorch
export BOOST_HOME=/path/to/boost
./scripts/build_colsys.sh $COLSYS_HOME $TVM_HOME $TORCH_HOME $BOOST_HOME
```

4. [Only required for Triton UM+MPS] Clone and build [Triton TensorRT UM Backend]().  

## Run and Evaluate

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
GLOG_logtostderr=1 ./build/server/colserve
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

## Run Triton Benchmark

Triton benchmark need additional steps to setup.

### setup inference models

models are stored at `server/triton_models`, as following. The model have a directory of the trtion compiled model (`model.plan`and `config.pbtxt`)

```
├── mnist
│   └── 1
│   │   └── model.plan
│   └── model.pbtxt
├── resnet152
│   └── 1
│   │   └── model.plan
│   └── model.pbtxt
...
└── config.conf
```

`config.conf` is used to configure memory usage of model.

```ini
resnet152         = 260 # 242MB
distilgpt2        = 345 # 326MB
efficientvit_b2   = 125 # 108MB
efficientnet_v2_s = 115 # 96MB
densenet161       = 90  # 68MB
distilbert_base   = 280 # 260MB
```

### setup triton tensorrt backend with unified memory support

build and install [TensorRT Backend UM](https://ipads.se.sjtu.edu.cn:1312/infer-train/triton_tensorrt_um).
