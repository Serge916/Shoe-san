#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
# export VEHICLE_NAME="db3"
rosrun customPackage pid.py

# wait for app to end
dt-launchfile-join
