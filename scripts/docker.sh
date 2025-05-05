USER=$USER

# docker run --gpus=all --ipc=host -it --name "gpu-col-$USER" \
#   -v /home/$USER/:/home/$USER \
#   -v /disk1/$USER/:/disk1 \
#   -v /disk2/$USER/:/disk2 \

docker run --gpus=all --ipc=host --network=host -it --name "gpu-col-$USER" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    inf-tra:latest \
    bash -c "source /opt/mambaforge/bin/activate colserve && cd /gpu-col && bash"