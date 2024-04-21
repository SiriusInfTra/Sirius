GPU_UUID=GPU-0bff5f67-7bc0-75b6-22ca-bd07b23f3482 # 0

export CUDA_VISIBLE_DEVICES=$GPU_UUID
export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/gpu-$1

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"

for i in `seq 1 3`; do
  python eval/overall_v2.py --all-sys --all-workload --retry-limit 3
  python eval/ablation.py --eval-all  --retry-limit 3
  python eval/breakdown.py --colsys --all-workload  --retry-limit 3
  python eval/memory_pressure.py --retry-limit 3
  # python eval/memory_pressure_v2.py --retry-limit 3

  # python eval/breakdown.py --colsys --all-workload  --retry-limit 3
  # python eval/overall_v2.py --all-sys --azure --azure-rps 150 --retry-limit 3


  # moti test
  # python eval/moti/infer_loading.py --model resnet152
  # python eval/moti/infer_loading.py --model efficientvit
  # python eval/moti/infer_loading.py --model efficientnet
  # python eval/moti/infer_loading.py --model distilbert
  # python eval/moti/infer_loading.py --model distilgpt2
  # python eval/moti/infer_loading.py --model densenet161
done