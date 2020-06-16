#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
if [[ -f "/opt/usc/devel/setup.bash" ]]; then
  source "/opt/usc/devel/setup.bash"
else
  echo "Warning: unable to find /opt/usc/devel/setup.bash. Run catkin build."
fi

exec "$@"

