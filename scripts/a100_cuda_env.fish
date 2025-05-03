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
    set cur_gpu_uuid GPU-54781309-aec7-6fb9-4bd3-cc569a0239b4
  else if test $i -eq 1
    set cur_gpu_uuid GPU-0753c94e-fc37-6269-ec06-8db7d3d25487
  else if test $i -eq 2
    set cur_gpu_uuid GPU-29027350-27fc-a949-1ce5-e22715266be0
  else if test $i -eq 3
    set cur_gpu_uuid GPU-6f25e03c-c15a-16eb-7e5d-e37931ff501c
  else if test $i -eq 4
    set cur_gpu_uuid GPU-1c1134b5-47e5-3edd-27ec-ab93a55667b3
  else if test $i -eq 5
    set cur_gpu_uuid GPU-3bd17db0-c74e-65f1-7812-0f92bc06e310
  else if test $i -eq 6
    set cur_gpu_uuid GPU-47dba085-6fa2-01d6-29db-0f14f57db1ea
  else if test $i -eq 7
    set cur_gpu_uuid GPU-c41db79b-ae61-c054-a1e4-b9ddf8f6b639
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
