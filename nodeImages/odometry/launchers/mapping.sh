#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
# export VEHICLE_NAME="db3"
rosrun customPackage odometry_with_map.py

# wait for app to end
dt-launchfile-join
