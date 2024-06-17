#!/usr/bin/env python3
import os
import time

import numpy as np

import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from geometry_msgs.msg import Polygon, Point32
from sensor_msgs.msg import PointCloud
from std_msgs.msg import Int8


from common.constants import *

class OrchestratorNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(OrchestratorNode, self).__init__(
            node_name=node_name, node_type=NodeType.BEHAVIOR
        )

        self.trigger_behavior = STOP_ROBOT
        self.current_behavior = STOP_ROBOT
        self.send_behavior    = STOP_ROBOT
        self.idle = True
        self.shoe_poses = PointCloud()

        # Create the array of all the shoes positions and set them to be uninitialized
        for _ in range(2*NUM_TYPE_SHOES):
            new_point = Point32()
            new_point.z = UNINITIALIZED
            self.shoe_poses.points.append(new_point)

        ## Construct publishers
        # Mission to execute
        self.decision_msg = rospy.Publisher(
            "~mission",
            Polygon,
            queue_size=1,
            dt_topic_type=TopicType.BEHAVIOR,
        )

        self.veh = os.environ["VEHICLE_NAME"]

        ## Construct subscribers
        # TODO: Shoe positions
        self.shoepos_msg = rospy.Subscriber(
            f"/{self.veh}/shoe_positions/shoes",
            PointCloud,
            self.shoepos_cb,
            #buff_size=10000,
            queue_size=1,
        )

        # Remote trigger to indicate which shoe to follow
        self.trigger_msg = rospy.Subscriber(
            f"~/trigger",
            Int8,
            self.trigger_cb,
            #buff_size=10000,
            queue_size=1,
        )

        # TODO: End mission
        self.end_msg = rospy.Subscriber(
            f"/{self.veh}/path_planner/end",
            Int8,
            self.end_cb,
            #buff_size=10000,
            queue_size=1,
        )

        # A first mission is sent to leave the robot still
        self._send_stop()

    def shoepos_cb(self, new_shoe_poses):
        self.log(f"New shoes detected!")

        for i, point in enumerate(new_shoe_poses.points):
            if point.z == NOT_FOUND:
                # Without removing the data, the shoe is indicated that is not seen anymore and is old data
                self.shoe_poses.points[i].z = NOT_FOUND
                continue
            
            self.shoe_poses.points[i].x = point.x
            self.shoe_poses.points[i].y = point.y
            self.shoe_poses.points[i].z = point.z
            self.log(f"The shoe id {i//2} was detected!")

        # After updating the shoe_poses array, the triggered behviour is checked whether the conditions are met to send the new behavior
        # Check if the trigger needs two shoes
        class_id = self.trigger_behavior
        self.log(f"Class ID: {class_id}. Z value of left shoe: {self.shoe_poses.points[2*class_id].z}. Right shoe: {self.shoe_poses.points[2*class_id + 1].z}. Detected {DETECTED}")
        if class_id in [TOM_CIRCLE, VASILIS_EIGHT]:
            if self.shoe_poses.points[2*class_id].z == DETECTED and self.shoe_poses.points[2*class_id + 1].z == DETECTED:
                self.log(f"Detected two shoes so {MISSION_DICT[class_id]} can be executed")
                self.current_behavior = class_id
            else:
                self.log(f"Not enogh shoes detected for {MISSION_DICT[class_id]}. Keep searching")
                self.current_behavior = NOTHING_IN_SIGHT
        # Check if the trigger needs only one shoe
        elif class_id in [SERGE_LOOK, SHASHANK_ORBIT, VARUN_FOLLOW]:
            if self.shoe_poses.points[2*class_id].z == DETECTED:
                self.log(f"Detected a shoes so {MISSION_DICT[class_id]} can be executed")
                self.current_behavior = class_id
            else:
                self.log(f"Not enogh shoes detected for {MISSION_DICT[class_id]}. Keep searching")
                self.current_behavior = NOTHING_IN_SIGHT
        else:
            # In this remaining case the trigger behavior is idle so nothing is will be sent
            self.log(f"Although new shoes arrived, the trigger behaviour is now idle.")
            self.current_behavior = self.trigger_behavior
        
                
        self.decide_mission()

    def trigger_cb(self, decision):
        self.log(f"Orchestrator was triggered to follow the behavior {MISSION_DICT[decision.data]}")
        self.trigger_behavior = decision.data

        # If a new desired behavior is given then the mission is forced back to STOP_ROBOT or NOTHING_IN_SIGHT until the trigger conditions are met
        if decision.data == STOP_ROBOT:
            self._send_stop()
        else:
            self._send_search()

    def end_cb(self, end):
        # Check if end is true just in case
        if end != 0:
            # TODO: Add some delay here in case it was one of the missions that are complex so that they do not start immediatly after?
            self.idle = True
            # The robot is given the mission of finding again the shoe
            self._send_search()
            
            self.log(f"** Received an end event from the path palnner **\n")

        else:
            self.log(f"Unexpected end message equal to 0!")

    def decide_mission(self):
        # The mission given to the path planner follows the format of a Polygon so that xy axis of the two elements
        # are the position of the shoes and z is the type of mission to execute.

        # Check if we are executing idle behavior so that the mission is not sent again
        if self.current_behavior in [STOP_ROBOT, NOTHING_IN_SIGHT]:
            return

        # If the path planner is still executing the path then no new mission is sent
        if self.idle != True:
            return

        # The way this is setup is that the behavior is also the class id of the shoe. Change this function if it is no longer the same!
        class_id = self.current_behavior

        left_shoe = self.shoe_poses.points[2*class_id]
        right_shoe = self.shoe_poses.points[2*class_id + 1]

        self._send_behaviour(left_shoe.x, left_shoe.y, right_shoe.x, right_shoe.y, self.current_behavior) 
            
    def _send_stop(self):
        self.log(f"Stopping the duckiebot!")
        self._send_behaviour(0, 0, 0, 0, STOP_ROBOT)

    def _send_search(self):
        self.log(f"Turn the duckiebot around in search of shoes!")
        self._send_behaviour(0, 0, 0, 0, NOTHING_IN_SIGHT)

    def _send_behaviour(self, left_shoe_x, left_shoe_y, right_shoe_x, right_shoe_y, behavior):
        self.log(f"Executing behavior {MISSION_DICT[behavior]}. Left shoe is on: ({left_shoe_x, left_shoe_y}). And the right shoe is on ({right_shoe_x, right_shoe_y})")
        mission = Polygon()
        mission.points = [Point32(), Point32()]
        mission.points[0].x = left_shoe_x
        mission.points[0].y = left_shoe_y
        mission.points[0].z = behavior

        mission.points[1].x = right_shoe_x
        mission.points[1].y = right_shoe_y
        mission.points[1].z = behavior

        self.idle = True if behavior in [STOP_ROBOT, NOTHING_IN_SIGHT] else False
        self.send_behavior = behavior
        self.current_behavior = behavior
        self.decision_msg.publish(mission)


if __name__ == "__main__":
    # create the node
    node = OrchestratorNode(node_name="orchestrator_node")

    # keep spinning
    rospy.spin()
