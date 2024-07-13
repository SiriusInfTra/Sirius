#!/bin/sh

# 检查输入参数
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <GPU_IDS> <NUM_RUNS> <COMMAND>"
    exit 1
fi

# 提取GPU ID列表
GPU_IDS=$1
shift

NUM_RUNS=$1
shift

# 设置CUDA_VISIBLE_DEVICES变量
export CUDA_VISIBLE_DEVICES=$GPU_IDS
echo "\$CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export CUDA_MPS_PIPE_DIRECTORY=/home/$USER/cuda_mps-$GPU_IDS
echo "\$CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"


# 提取NUMA节点ID（例如，对于GPU 0和1，NUMA节点为0，对于GPU 2和3，NUMA节点为1）
NUMA_NODE=$((GPU_IDS / 2))

# 使用numactl设置NUMA节点
for i in `seq 1 $NUM_RUNS`; do
    echo "Run $i in GPU $GPU_IDS: numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE $@" 
    numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE $@ || echo "Error in run $i"
done