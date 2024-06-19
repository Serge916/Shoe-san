#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
# export VEHICLE_NAME="db3"
rosrun customPackage base_odometry.py

# wait for app to end
dt-launchfile-join
