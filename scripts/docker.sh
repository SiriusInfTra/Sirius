USER=$USER

# docker run --gpus=all --ipc=host -it --name "gpu-col-$USER" \
#   -v /home/$USER/:/home/$USER \
#   -v /disk1/$USER/:/disk1 \
#   -v /disk2/$USER/:/disk2 \

mkdir -p $(pwd)/gpu-col-docker-log
mkdir -p $(pwd)/triton-model-wksp

docker run --gpus=all --ipc=host --network=host -it --name "gpu-col-$USER" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(pwd)/gpu-col-docker-log:/gpu-col/log \
    -v $(pwd)/triton-model-wksp:/gpu-col/triton-model-wksp \
    -w /gpu-col \
    -e HOST_DOCKER_RUN_DIR=$(pwd) \
    -e DOCKER_TRITON_MODEL_WKSP=triton-model-wksp \
    -e DOCKER_GPU_COL_LOG_DIR=gpu-col-docker-log \
    -e DOCKER_MPS_PIPE_DIRECTORY=/dev/shm/ \
    siriusinftra/sirius:latest