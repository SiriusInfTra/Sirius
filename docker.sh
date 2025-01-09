docker run --privileged=true --gpus=all --ipc=host --network host -it --name "gpu-col-$USER" \
    -v /home/$USER/:/$USER \
    -v /nvme/$USER/:/nvme \
    inf-tra