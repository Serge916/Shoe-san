#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun tof_driver tof_subscriber.py

# wait for app to end
dt-launchfile-join
