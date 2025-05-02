if [ -z "$TMPDIR" ]; then
  export TMPDIR=/tmp
fi

export TENSORRT_BACKEND_UNIFIED_MEMORY_PATH=$(pwd)/triton/tensorrt_um/install

single_gpu_env() {
  GPU_UUID=$(nvidia-smi -L | sed -n 's/.*UUID: \(.*\))/\1/p' | head -n1)
  export CUDA_VISIBLE_DEVICES=$GPU_UUID
  export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/gpu-0

  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
}

two_gpu_env() {
  GPU_UUIDS=$(nvidia-smi -L | sed -n 's/.*UUID: \(.*\))/\1/p' | head -n2)
  GPU_UUIDS=$(echo $GPU_UUIDS | tee | tr ' ' ',')
  export CUDA_VISIBLE_DEVICES=$GPU_UUIDS
  export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/gpu-0

  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
}

multi_gpu_env() {
  GPU_UUIDS=$(nvidia-smi -L | sed -n 's/.*UUID: \(.*\))/\1/p' | head -n4)
  GPU_UUIDS=$(echo $GPU_UUIDS | tee | tr ' ' ',')
  export CUDA_VISIBLE_DEVICES=$GPU_UUIDS
  export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/gpu-0

  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
}

over_all_single_gpu() {
  echo -e "\n\033[1;32m==================================================================\033[0m"
  echo -e "\033[1;32m[over_all_single_gpu]\033[0m"
  echo -e "\033[1;32m==================================================================\033[0m\n"

  single_gpu_env
  python eval/overall_v2.py --uniform-v2 --skewed-v2 --azure \
    --colsys --static-partition --task-switch --um-mps --infer-only \
    --skip-set-mps-pct --retry-limit 3
}

over_all_multi_gpu() {
  echo -e "\n\033[1;32m==================================================================\033[0m"
  echo -e "\033[1;32m[over_all_multi_gpu]\033[0m"
  echo -e "\033[1;32m==================================================================\033[0m\n"

  multi_gpu_env
  python eval/overall_v2.py --uniform-v2 --uniform-v2-wkld-types NormalLight \
    --colsys --static-partition --task-switch --um-mps --infer-only \
    --skip-set-mps-pct --multi-gpu --retry-limit 3
}

breakdown() {
  echo -e "\n\033[1;32m==================================================================\033[0m"
  echo -e "\033[1;32m[breakdown]\033[0m"
  echo -e "\033[1;32m==================================================================\033[0m\n"

  single_gpu_env
  python eval/breakdown.py --colsys --strawman --azure --retry-limit 3
  multi_gpu_env
  python eval/breakdown.py --colsys --strawman --azure --multi-gpu --retry-limit 3
}

ablation_study() {
  echo -e "\n\033[1;32m==================================================================\033[0m"
  echo -e "\033[1;32m[ablation_study]\033[0m"
  echo -e "\033[1;32m==================================================================\033[0m\n"

  single_gpu_env
  python eval/ablation.py --eval-all --retry-limit 3
}

unbalance() {
  echo -e "\n\033[1;32m==================================================================\033[0m"
  echo -e "\033[1;32m[unbalance]\033[0m"
  echo -e "\033[1;32m==================================================================\033[0m\n"

  two_gpu_env
  python eval/unbalance.py --retry-limit 3
}

memory_pressure() {
  echo -e "\n\033[1;32m==================================================================\033[0m"
  echo -e "\033[1;32m[memory_pressure]\033[0m"
  echo -e "\033[1;32m==================================================================\033[0m\n"

  single_gpu_env
  python eval/memory_pressure.py --retry-limit 3
}

for i in `seq 1 1`; do
  over_all_single_gpu
  over_all_multi_gpu
  breakdown
  ablation_study
  unbalance
  memory_pressure
done