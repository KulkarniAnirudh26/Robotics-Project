#!/usr/bin/env bash
set -euxo pipefail

docker run \
  -it \
  -v $(pwd)/../catkin_ws:/opt/usc \
  usc545mcl:latest \
  bash -c "cd /opt/usc && catkin build"
