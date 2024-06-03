#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch publisher
rosrun customPackage coordinate_publisher.py

# wait for app to end
dt-launchfile-join