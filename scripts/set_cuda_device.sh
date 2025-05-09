help() {
  echo "Usage: $1 <GPU ID> ... "
  echo "       set CUDA_VISIBLE_DEVICES to the specified GPU ID"
  echo "Example: . $1 0"
}

get_gpu_uuid() {
    local line=$1
    GPU_UUID=$(nvidia-smi -L | sed -n 's/.*UUID: \(.*\))/\1/p' | head -n$line | tail -n1)
    echo $GPU_UUID
}

if [[ $# -lt 1 ]]; then
  help $0
  return
fi

GPU_UUID=""

for i in $@; do
  if [[ $i -eq 0 ]]; then
    cur_gpu_uuid=$(get_gpu_uuid 1)
  elif [[ $i -eq 1 ]]; then
    cur_gpu_uuid=$(get_gpu_uuid 2)
  elif [[ $i -eq 2 ]]; then
    cur_gpu_uuid=$(get_gpu_uuid 3)
  elif [[ $i -eq 3 ]]; then
    cur_gpu_uuid=$(get_gpu_uuid 4)
  else
    echo "Invalid GPU ID: $i"
  fi

  if [[ -z $GPU_UUID ]]; then
    GPU_UUID=$cur_gpu_uuid
  else
    GPU_UUID=$GPU_UUID,$cur_gpu_uuid
  fi
done

export CUDA_VISIBLE_DEVICES=$GPU_UUID
echo "Setting CUDA_VISIBLE_DEVICES to $GPU_UUID"