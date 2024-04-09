
help() {
  echo "Usage: $1 <GPU ID>"
  echo "       set CUDA_VISIBLE_DEVICES to the specified GPU ID"
  echo "       set CUDA_MPS_PIPE_DIRECTORY to TMPDIR ($TMPDIR)"
  echo "Example: . $1 0"
}

if [[ $# -ne 1 ]]; then
  help $0
fi

GPU_UUID=""

if [[ $1 -eq 0 ]]; then
  GPU_UUID=GPU-0bff5f67-7bc0-75b6-22ca-bd07b23f3482
elif [[ $1 -eq 1 ]]; then
  GPU_UUID=GPU-ea8006f2-470f-f147-2425-74cede8f6cd8
elif [[ $1 -eq 2 ]]; then
  GPU_UUID=GPU-2565c74c-2807-e9e9-4b9e-83c5b7edb933
elif [[ $1 -eq 3 ]]; then
  GPU_UUID=GPU-76991e8b-e218-1368-e74a-bdd769009656
else
  echo "Invalid GPU ID: $1"
fi

export CUDA_VISIBLE_DEVICES=$GPU_UUID
export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
