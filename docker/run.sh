#!/bin/bash
docker run -it \
    --gpus all \
    --ipc host \
    --rm \
    --ulimit memlock=-1 \
    --name ddrive_dio \
    -v /mnt/nvme0/workspace:/workspace \
    -v /mnt/nvme1:/mnt/nvme1 \
    -v /home/chenglin/.ssh:/home/dio/.ssh \
    ddrive:v0.2
