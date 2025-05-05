function help
  echo "Usage: $argv[1] <GPU ID> ... "
  echo "       set CUDA_VISIBLE_DEVICES to the specified GPU ID"
  echo "Example: source $argv[1] 0"
end

function get_gpu_uuid
    set -l line $argv[1]
    set GPU_UUID (nvidia-smi -L | sed -n 's/.*UUID: \(.*\))/\1/p' | head -n$line | tail -n1)
    echo $GPU_UUID
end

if test (count $argv) -lt 1
  help $argv[1]
  exit
end

set -l GPU_UUID ""

for i in $argv
  set -l cur_gpu_uuid ""
  
  if test $i -eq 0
    set cur_gpu_uuid (get_gpu_uuid 1)
  else if test $i -eq 1
    set cur_gpu_uuid (get_gpu_uuid 2)
  else if test $i -eq 2
    set cur_gpu_uuid (get_gpu_uuid 3)
  else if test $i -eq 3
    set cur_gpu_uuid (get_gpu_uuid 4)
  else
    echo "Invalid GPU ID: $i"
  end

  if test -z "$GPU_UUID"
    set GPU_UUID $cur_gpu_uuid
  else
    set GPU_UUID "$GPU_UUID,$cur_gpu_uuid"
  end
end

# Set the environment variable
export CUDA_VISIBLE_DEVICES=$GPU_UUID
