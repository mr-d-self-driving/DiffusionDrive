docker run -it \
    --gpus all \
    --ipc host \
    --ulimit memlock=-1 \
    --name ddrive_dio \
    -v /mnt/nvme2/chenglin/workspace:/workspace \
    -v /mnt/nvme2/chenglin/data:/data \
    -v /mnt/nvme1:/mnt/nvme1 \
    -v /home/chenglin/.ssh:/home/dio/.ssh \
    ddrive_dio:latest
