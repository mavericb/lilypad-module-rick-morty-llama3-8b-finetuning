#!/bin/bash

# This script is used to build all of the modules in SDXL Pipeline.
# It requires *quite a lot* of disk space - be warned!

### VERSIONS ###

### NOTE ###
# Specify the versions of the Docker and Lilypad modules in VERSIONS.env

# Change to the directory that this script is in.
cd "$(dirname "$0")"

# Load the versions
source VERSIONS.env

# Check that the Docker versions are set
if [[ -z $VLLAMA3_8B ]]; then
    echo "Please set the Docker versions in VERSIONS.env before building."
    exit 1
fi

# Build the Docker containers for each model
echo "Building Docker containers..."

# Turn on Docker BuildKit and cd to the docker directory
cd ../docker/
export DOCKER_BUILDKIT=1

# Login to Docker Hub
docker login -u mavericb -p X

# Build the Docker image
docker build -f Dockerfile-llama3-8b -t mavericb/ollama:llama3-8b-lilypad$VLLAMA3_8B .

# Generate a new tag
NEW_TAG=$(date +v%Y%m%d%H%M%S)
docker build -f Dockerfile-llama3-8b -t mavericb/ollama:llama3-8b-lilypad-$NEW_TAG .

# Publish the Docker containers
echo "Publishing Docker containers..."
docker push mavericb/ollama:llama3-8b-lilypad-$NEW_TAG

echo "Done!"