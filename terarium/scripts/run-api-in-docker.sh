#!/usr/bin/env bash

# For local images:
# TARGET_IMAGE=localhost:5000/siftech/funman-api:local

# For public images:
FUNMAN_VERSION="${FUNMAN_VERSION:-1.8.0}"
TARGET_IMAGE="${TARGET_IMAGE:-ghcr.io/siftech/funman-api:$FUNMAN_VERSION}"
PLATFORM="${PLATFORM:---platform linux/amd64}"

echo "Running [" docker run $PLATFORM --rm -p 127.0.0.1:8190:8190 --pull always $TARGET_IMAGE " ]"
docker run $PLATFORM --rm -p 127.0.0.1:8190:8190 --pull always $TARGET_IMAGE