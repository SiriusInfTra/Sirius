#!/bin/bash

help() {
    echo "Usage: $0 --mps-pipe [CUDA_MPS_PIPE_DIRECTORY]"
    echo "       launch mps-daemon on the specified GPUs and pipe directory"
    echo "Example: $0 --mps-pipe \"\$HOME/some-dir-for-mps-pipe\""
    exit 1
}

CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      help
      shift
      ;;
    --mps-pipe)
      CUDA_MPS_PIPE_DIRECTORY="$2"
      shift 2
      ;;
  esac
done

echo "CUDA_MPS_PIPE_DIRECTORY: ${CUDA_MPS_PIPE_DIRECTORY}"
echo quit | CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY} nvidia-cuda-mps-control