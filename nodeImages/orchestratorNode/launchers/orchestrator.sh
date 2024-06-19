#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch orchestrator
source /code/catkin_ws/devel/setup.bash --extend

dt-exec roslaunch --wait orchestrator orchestrator_node.launch veh:=$VEHICLE_NAME

# wait for app to end
dt-launchfile-join
