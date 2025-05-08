## Artifact for the Paper "<u>#734 Colocating ML Inference and Training with Fast GPU Memory Handover</u>"

<!-- Intro -->

[TOC]

## Project Structure

```bash
$ tree --dirsfirst  -L 2 .
├── client                      
├── cmake                       # CMake helper files
├── common                      # Common libraries for inference/training
├── environment                 # Docker and conda environment files
├── eval
│   ├── runner                  # Automatic evaluation runner
│   └── ...                     # Evaluation scripts for test cases
├── log                         # Running logs
├── proto                       # gRPC proto
├── pytorch                     # PyTorch plugin
├── scripts                    
├── server                      # Inference server
│   ├── models                  # Contains inference models
│   └── ... 
├── train                       # PyTorch training scripts
├── third_party/mpool...        # GPU memory pool
└── ...
```

## Hardware Requirements

- 4 x NVIDIA V100 (16GB)
- 1 x NVIDIA A100 (80GB)

## Build and Install 

### Using Docker Image

**Option 1: Pull from Docker Hub**

Pull the pre-built Docker images from Docker Hub. The script `./scripts/docker.sh` is provided as a wrapper for Docker commands.

```bash
docker pull siriusinftra/sirius:latest
docker pull siriusinftra/triton-trt-um:latest # Triton TensorRT UM backend

bash ./scripts/docker.sh
```

The project is located at `/gpu-col` within the Docker container. TVM and Triton models are pre-installed in this image. 

Before running the system, activate the conda environment (e.g., `conda activate colserve`). 

To evaluate Sirius, refer to [Run Benchmark](#run-and-evaluate) and [Artifact Evaluation](artifact-evaluation/README.md) for more details.



**Option 2: Build from Dockerfile**

1. Clone the repository and build the Docker image. The `build_docker.sh` script will clone dependencies into `inftra-docker-build`, which serves as the Docker build context.

> [Optional] Copy TVM and Triton models to `inftra-docker-build/tvm-models` and `inftra-docker-build/triton-models` respectively. These will be copied into the Docker image.

```bash
git clone --recurse-submodules git@github.com:SiriusInfTra/Sirius.git gpu-col
bash ./gpu-col/scripts/build_docker.sh
```

2. Build Triton TensorRT UM Docker image.

```bash
bash ./gpu-col/scripts/build_triton_trt_um_docker.sh
```



### Compile From Source (using conda)

**Software Requirements**: `cmake>=3.24`, `gcc>=9.4`, `nvcc>=11.6`, `ninja`

**Create Environment and Build System**:

1. Prepare a new conda environment, install Python packages, and then clone the repository.

```bash
conda create -n colserve python=3.12
conda activate colserve
conda install -y conda-forge::python-devtools nvitop conda-forge::c-ares
pip install -r environment/requirements.txt

export SIRIUS_HOME=/path/to/clone/repo
git clone --recurse-submodules git@github.com:SiriusInfTra/Sirius.git $SIRIUS_HOME
```

2. Install `Boost>=1.80` by compiling from source (Boost installed via apt/conda might require a higher GCC version).

```bash
export BOOST_HOME=/path/to/install/boost
$SIRIUS_HOME/scripts/install_boost.sh $BOOST_HOME
```

3. Clone and build [TVM](git@github.com:SiriusInfTra/tvm.git) for inference, and [PyTorch](git@github.com:SiriusInfTra/pytorch.git) and [TorchVision](https://github.com/pytorch/vision/tree/v0.13.1) for training. Ensure the CUDA backend is enabled. Pay attention to the PyTorch `GLIBCXX_USE_CXX11_ABI` flag, which can cause ABI issues. To accelerate the build, set the `TORCH_CUDA_ARCH_LIST` flag to your GPU's compute capability (e.g., `TORCH_CUDA_ARCH_LIST=7.0` for V100).

4. Set the `TVM_HOME` environment variable. Verify by running `echo $TVM_HOME` and `echo $CONDA_PREFIX`. Then, configure CMake.

```bash
export TVM_HOME=/path/to/tvm
export TORCH_HOME=/path/to/pytorch
export BOOST_HOME=/path/to/boost
$SIRIUS_HOME/scripts/build_sirius.sh $SIRIUS_HOME $TVM_HOME $TORCH_HOME $BOOST_HOME
```

5. [Only required for Triton UM+MPS] Set up Triton TensorRT backend with Unified Memory support. Clone and build [Triton TensorRT UM Backend](git@github.com:SiriusInfTra/triton_tensorrt_um.git).

```bash
export TRITON_TRT_UM_HOME=/path/to/triton_tensorrt_um
export TRITON_TRT_INSTALL_HOME=/path/to/triton_tensorrt_um_install # e.g., $SIRIUS_HOME/triton/tensorrt_um/install
bash $SIRIUS_HOME/scripts/build_triton_trt_um.sh $TRITON_TRT_UM_HOME $TRITON_TRT_INSTALL_HOME
```

5. [Only required for LLM] Install vLLM by compiling from source, clone [xFormer](git@github.com:SiriusInfTra/xformer.git) and [vLLM](git@github.com:SiriusInfTra/vllm.git).

```bash
export VLLM_HOME=/path/to/vllm
export XFORMER_HOME=/path/to/xformer
bash $SIRIUS_HOME/scripts/build_vllm.sh $VLLM_HOME $XFORMER_HOME
```

## Run and Evaluate

### Prepare Inference Models

**TVM Models**

Compile models using TVM (refer to [./util/prepare_model_store](util/prepare_model_store)). TVM models (i.e., `mod.json`, `mod.params`, and `mod.so`) are stored in `server/models`, as shown below. 

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

Compile Triton models using TensorRT (refer to [./util/onnx](util/onnx)). Triton models are stored in `server/triton_models`. Each model has a directory containing the Triton compiled model (`model.plan` and `config.pbtxt`), as shown below.

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

`config.conf` is used to configure the memory usage (in MiB) for each model.

```ini
resnet152         = 345
distilgpt2        = 349
efficientvit_b2   = 143
efficientnet_v2_s = 114
densenet161       = 107
distilbert_base   = 278
```

**LLM**

Download Llama2 from Hugging Face.

```python
from transformers import AutoConfig, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")

config = AutoConfig.from_pretrained('Qwen/Qwen2-0.5B')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B', config=config)
```

### Run Benchmark

The evaluation is fully automated by the script at [./eval/runner](./eval/runner). This script will automatically launch GPU MPS, Sirius's inference server, PyTorch training tasks, and inference workloads.

For example, to evaluate Sirius with the **Light** workload:

```bash
source ./scripts/set_cuda_device.sh 0
python eval/overall_v2.py --uniform-v2 --uniform-v2-wkld-types NormalLight \
    --sirius --skip-set-mps-pct
```

The evaluation results will be saved in a directory like `log/overall-uniform-v2-1gpu-YYYYMMDD-HHMM/colsys-NormalLight`.

### Artifact Evaluation

Please refer to [./artifact-evaluation/README.md](artifact-evaluation/README.md) for more details on the artifact evaluation process.