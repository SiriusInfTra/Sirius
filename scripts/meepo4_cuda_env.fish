function help
  echo "Usage: $argv <GPU ID> ... "
  echo "       set CUDA_VISIBLE_DEVICES to the specified GPU ID"
  echo "       set CUDA_MPS_PIPE_DIRECTORY to TMPDIR ($TMPDIR)"
  echo "Example: . $argv 0"
end

if test (count $argv) -lt 1
  help (status -f)
  exit
end

set GPU_UUID ""

for i in $argv
  if test $i -eq 0
    set cur_gpu_uuid GPU-b9ce1ac6-89bf-0eb6-6cab-d035581ace23
  else if test $i -eq 1
    set cur_gpu_uuid GPU-dd9deeda-76a3-30eb-c976-f9cb14500b0e
  else
    echo "Invalid GPU ID: $i"
  end

  if test -z $GPU_UUID
    set GPU_UUID $cur_gpu_uuid
  else
    set GPU_UUID $GPU_UUID,$cur_gpu_uuid
  end
end

if test -z $TMPDIR
  set TMPDIR /tmp
end

mkdir -p $TMPDIR/gpu-$argv[1]

export CUDA_VISIBLE_DEVICES=$GPU_UUID
export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/gpu-$argv[1]

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
