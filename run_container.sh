#!/bin/bash
BASE_DIR=$(realpath "$(dirname $0)")
cd ${BASE_DIR}/..
xhost +
docker run \
    --gpus all \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    --rm \
    -it \
    --network=host \
    --name foundationpose \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v ${BASE_DIR}:/opt/nvidia/FoundationPose \
    -v /mnt:/mnt \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp:/tmp \
    --ipc=host \
    -e DISPLAY=${DISPLAY} \
    -e GIT_INDEX_FILE \
    foundationpose:latest \
    bash
