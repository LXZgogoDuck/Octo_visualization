#!/bin/bash
set -x  # Enable debugging mode (shows each command before running)

# Set Docker image tag
TAG=xuanzhuo/my_docker

# Set parent image (CUDA version)
PARENT=nvidia/cudagl:11.4.2-devel-ubuntu20.04

# Ensure compatibility with Apple Silicon (M1/M2/M3) by forcing amd64
export DOCKER_DEFAULT_PLATFORM=linux/amd64

# Get user and group ID for correct permissions
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Run the Docker build command
docker build -f docker/Dockerfile \
  --build-arg PARENT_IMAGE=${PARENT} \
  --build-arg USER_ID=${USER_ID} \
  --build-arg GROUP_ID=${GROUP_ID} \
  -t ${TAG} .
