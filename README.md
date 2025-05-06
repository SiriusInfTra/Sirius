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

1. Clone the repo and build docker image. `build_docker.sh` will clone the dependencies in `inftra-docker-build`, which is the docker build context. 

> [Optional] Copy TVM and Triton models to `inftra-docker-build/tvm-models` and `inftra-docker-build/triton-models` respectively, which will be copy to docker image. 

```
git clone --recurse-submodules git@ipads.se.sjtu.edu.cn:infer-train/gpu-colocation.git gpu-col
bash ./gpu-col/scripts/build_docker.sh
```

2. Build Triton TensorRT UM docker image.

```
bash ./gpu-col/scripts/build_triton_trt_um_docker.sh
```



### Compile From Source (using conda)

**Software Requirements**: `cmake>=3.24`, `gcc>=9.4`, `nvcc>=11.6`, `ninja`

**Create Environment and Build System**:

1. Prepare a new conda environment and install python packages, then clone the repo.

```bash
conda create -n colserve python=3.12
conda install -y conda-forge::python-devtools nvitop conda-forge::c-ares
pip install -r environment/requirements.txt

export SIRIUS_HOME=/path/to/clone/repo
git clone --recurse-submodules git@ipads.se.sjtu.edu.cn:infer-train/gpu-colocation.git $SIRIUS_HOME
```

2. Install `Boost>=1.80` from compiling source (boost installed from apt/conda may require higher version of gcc).  

```bash
export BOOST_HOME=/path/to/install/boost
$SIRIUS_HOME/scripts/install_boost.sh $BOOST_HOME
```

2. Clone and build [tvm](https://ipads.se.sjtu.edu.cn:1312/infer-train/tvm) for inference; [pytorch](https://ipads.se.sjtu.edu.cn:1312/infer-train/pytorch) and [torchvision](https://github.com/pytorch/vision/tree/v0.13.1) for training. Note CUDA backend should be enabled. Pay attention to pytorch `GLIBCXX_USE_CXX11_ABI` flag, which may cause ABI issues. To accelerate building, set `TORCH_CUDA_ARCH_LIST` flag to gpu computing capability, e.g., `TORCH_CUDA_ARCH_LIST=7.0`.

3. Set `TVM_HOME` environment, run `echo $TVM_HOME` and `echo $CONDA_REFIX` to check. Then configure cmake.

```bash
export TVM_HOME=/path/to/tvm
export TORCH_HOME=/path/to/pytorch
export BOOST_HOME=/path/to/boost
$SIRIUS_HOME/scripts/build_sirius.sh $SIRIUS_HOME $TVM_HOME $TORCH_HOME $BOOST_HOME
```

4. [Only required for Triton UM+MPS] Setup Triton TensorRT backend with Unified Memory support, clone and build [Triton TensorRT UM Backend](https://ipads.se.sjtu.edu.cn:1312/infer-train/triton_tensorrt_um).  

```bash
bash $SIRIUS_HOME/scripts/build_triton_trt_um.sh
```

5. [Only required for LLM] Install vLLM by compiling from source, clone [xFormer](git@ipads.se.sjtu.edu.cn:infer-train/xformer.git) and [vLLM](git@ipads.se.sjtu.edu.cn:infer-train/tvm.git).

```bash
export VLLM_HOME=/path/to/vllm
export XFORMER_HOME=/path/to/xformer
bash $SIRIUS_HOME/scripts/build_vllm.sh $VLLM_HOME $XFORMER_HOME
```

## Run and Evaluate

### Prepare Inference Models

**TVM Models**

Compile models using TVM (refer to [util/prepare_model_store](util/prepare_model_store)). TVM models (i.e., `mod.josn`, `mod.params` and `mod.so`) are stored at `server/models`, as shown below. 

```
server/models
├── densenet161-b1
├── distilbert_base-b1          
├── distilgpt2-b1          
├── efficientnet_v2_s-b1  
├── efficientvit_b2-b1        
└── resnet152-b1 
```

**Triton Models**

Triton models are stored at `server/triton_models`, as shown below. The model have a directory of the trtion compiled model (`model.plan`and `config.pbtxt`)

```
├── densenet161
├── distilbert_base
├── distilgpt2
├── efficientnet_v2_s
├── efficientvit_b2
├── resnet152
│   ├── 1
│   │   └── model.plan
│   └── config.pbtxt
└── config.conf
```

`config.conf` is used to configure memory usage (MiB) of model.

```ini
resnet152         = 345
distilgpt2        = 349
efficientvit_b2   = 143
efficientnet_v2_s = 114
densenet161       = 107
distilbert_base   = 278
```

**LLM**

Download Llama2 from Huggingface.

```python
from transformers import AutoConfig, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")

config = AutoConfig.from_pretrained('Qwen/Qwen2-0.5B')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B', config=config)
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

