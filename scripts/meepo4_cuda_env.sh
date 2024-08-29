
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
    cur_gpu_uuid=GPU-b9ce1ac6-89bf-0eb6-6cab-d035581ace23
  elif [[ $i -eq 1 ]]; then
    cur_gpu_uuid=GPU-dd9deeda-76a3-30eb-c976-f9cb14500b0e
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
