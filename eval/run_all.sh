if [ -z "$TMPDIR" ]; then
  export TMPDIR=/tmp
fi

export TENSORRT_BACKEND_UNIFIED_MEMORY_PATH=$(pwd)/triton/tensorrt_um/install

# Function to display help information
show_help() {
  echo "Usage: $(basename $0) [OPTIONS]"
  echo "Run GPU evaluation tests."
  echo
  echo "Options:"
  echo "  -h, --help             Display this help message"
  echo "  --overall-single-gpu   Run single GPU overall evaluation"
  echo "  --overall-multi-gpu    Run multi-GPU overall evaluation"
  echo "  --breakdown            Run breakdown evaluation"
  echo "  --ablation             Run ablation study"
  echo "  --unbalance            Run unbalance test"
  echo "  --memory-pressure      Run memory pressure test"
  echo "  --llm                  Run LLM test"
  echo "  --all                  Run all tests"
  echo
  echo "If no options are specified, all tests will be run."
}

# Initialize flags for test units
run_overall_single=false
run_overall_multi=false
run_breakdown=false
run_ablation=false
run_unbalance=false
run_memory_pressure=false
run_llm=false

# Parse command line arguments
if [ $# -eq 0 ]; then
  # If no arguments, run all tests
  run_overall_single=true
  run_overall_multi=true
  run_breakdown=true
  run_ablation=true
  run_unbalance=true
  run_memory_pressure=true
  run_llm=true
else
  # Process arguments
  for arg in "$@"
  do
    case $arg in
      -h|--help)
        show_help
        exit 0
        ;;
      --overall-single-gpu)
        run_overall_single=true
        ;;
      --overall-multi-gpu)
        run_overall_multi=true
        ;;
      --breakdown)
        run_breakdown=true
        ;;
      --ablation)
        run_ablation=true
        ;;
      --unbalance)
        run_unbalance=true
        ;;
      --memory-pressure)
        run_memory_pressure=true
        ;;
      --llm)
        run_llm=true
        ;;
      --all)
        run_overall_single=true
        run_overall_multi=true
        run_breakdown=true
        run_ablation=true
        run_unbalance=true
        run_memory_pressure=true
        run_llm=true
        ;;
      *)
        echo "Unknown argument: $arg"
        echo "Available arguments: --overall-single-gpu --overall-multi-gpu --breakdown --ablation --unbalance --memory-pressure --llm --all"
        exit 1
        ;;
    esac
  done
fi

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
    --skip-set-mps-pct --retry-limit 3 --skip-fail 1 --parse-result
}

over_all_multi_gpu() {
  echo -e "\n\033[1;32m==================================================================\033[0m"
  echo -e "\033[1;32m[over_all_multi_gpu]\033[0m"
  echo -e "\033[1;32m==================================================================\033[0m\n"

  multi_gpu_env
  python eval/overall_v2.py --uniform-v2 --uniform-v2-wkld-types NormalLight \
    --colsys --static-partition --task-switch --um-mps --infer-only \
    --skip-set-mps-pct --multi-gpu --retry-limit 3 --skip-fail 1 --parse-result
}

breakdown() {
  echo -e "\n\033[1;32m==================================================================\033[0m"
  echo -e "\033[1;32m[breakdown]\033[0m"
  echo -e "\033[1;32m==================================================================\033[0m\n"

  single_gpu_env
  python eval/breakdown.py --colsys --strawman --azure --retry-limit 3 --parse-result
  multi_gpu_env
  python eval/breakdown.py --colsys --strawman --azure --multi-gpu --retry-limit 3 --parse-result
}

ablation_study() {
  echo -e "\n\033[1;32m==================================================================\033[0m"
  echo -e "\033[1;32m[ablation_study]\033[0m"
  echo -e "\033[1;32m==================================================================\033[0m\n"

  single_gpu_env
  python eval/ablation.py --eval-all --retry-limit 3 --parse-result
}

unbalance() {
  echo -e "\n\033[1;32m==================================================================\033[0m"
  echo -e "\033[1;32m[unbalance]\033[0m"
  echo -e "\033[1;32m==================================================================\033[0m\n"

  two_gpu_env
  python eval/unbalance.py --parse-result
}

memory_pressure() {
  echo -e "\n\033[1;32m==================================================================\033[0m"
  echo -e "\033[1;32m[memory_pressure]\033[0m"
  echo -e "\033[1;32m==================================================================\033[0m\n"

  single_gpu_env
  python eval/memory_pressure.py --retry-limit 3 --parse-result
}

llm() {
  echo -e "\n\033[1;32m==================================================================\033[0m"
  echo -e "\033[1;32m[llm]\033[0m"
  echo -e "\033[1;32m==================================================================\033[0m\n"
  single_gpu_env

  python eval/run_llm.py --colsys --burstgpt --burstgpt-rps 10
}

echo "TEST BEGIN: $(date)"

for i in `seq 1 1`; do
  if $run_overall_single; then
    over_all_single_gpu
  fi
  
  if $run_overall_multi; then
    over_all_multi_gpu
  fi
  
  if $run_breakdown; then
    breakdown
  fi
  
  if $run_ablation; then
    ablation_study
  fi
  
  if $run_unbalance; then
    unbalance
  fi
  
  if $run_memory_pressure; then
    memory_pressure
  fi
  
  if $run_llm; then
    llm
  fi
done

echo "TEST END: $(date)"