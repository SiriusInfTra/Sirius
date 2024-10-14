#!/bin/bash

help() {
    echo "Usage: $0 --device [CUDA_VISIBLE_DEVICES] --mps-pipe [CUDA_MPS_PIPE_DIRECTORY]"
    echo "       launch mps-daemon on the specified GPUs and pipe directory"
    echo "Example: $0 --device \"0\" --mps-pipe \"\$HOME/some-dir-for-mps-pipe\""
    exit 1
}

CUDA_VISIBLE_DEVICES="0,1,2,3"
CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      help
      shift
      ;;
    --device)
      CUDA_VISIBLE_DEVICES="$2"
      shift 2
      ;;
    --mps-pipe)
      CUDA_MPS_PIPE_DIRECTORY="$2"
      shift 2
      ;;
  esac
done

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}, CUDA_MPS_PIPE_DIRECTORY: ${CUDA_MPS_PIPE_DIRECTORY}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY} nvidia-cuda-mps-control -f