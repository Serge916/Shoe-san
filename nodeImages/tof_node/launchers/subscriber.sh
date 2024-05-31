#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun customPackage subscriber.py

# wait for app to end
dt-launchfile-join
