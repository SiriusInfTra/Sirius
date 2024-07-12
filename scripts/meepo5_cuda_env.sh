
help() {
  echo "Usage: $1 <GPU ID> ... "
  echo "       set CUDA_VISIBLE_DEVICES to the specified GPU ID"
  echo "       set CUDA_MPS_PIPE_DIRECTORY to TMPDIR ($TMPDIR)"
  echo "Example: . $1 0"
}

if [[ $# -lt 1 ]]; then
  help $0
  return
fi

GPU_UUID=""

for i in $@; do
  if [[ $i -eq 0 ]]; then
    cur_gpu_uuid=GPU-0bff5f67-7bc0-75b6-22ca-bd07b23f3482
  elif [[ $i -eq 1 ]]; then
    cur_gpu_uuid=GPU-ea8006f2-470f-f147-2425-74cede8f6cd8
  elif [[ $i -eq 2 ]]; then
    cur_gpu_uuid=GPU-2565c74c-2807-e9e9-4b9e-83c5b7edb933
  elif [[ $i -eq 3 ]]; then
    cur_gpu_uuid=GPU-76991e8b-e218-1368-e74a-bdd769009656
  else
    echo "Invalid GPU ID: $i"
  fi

  if [[ -z $GPU_UUID ]]; then
    GPU_UUID=$cur_gpu_uuid
  else
    GPU_UUID=$GPU_UUID,$cur_gpu_uuid
  fi
done

mkdir -p $TMPDIR/gpu-$1

export CUDA_VISIBLE_DEVICES=$GPU_UUID
export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/gpu-$1

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
