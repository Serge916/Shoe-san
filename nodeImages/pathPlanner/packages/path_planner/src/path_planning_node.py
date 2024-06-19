#!/usr/bin/env python3

import os
import rospy
import math
import time
import numpy as np
from constants import *
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Polygon, PoseStamped
from nav_msgs.msg import Odometry, Path
from duckietown_msgs.msg import Pose2DStamped
from duckietown.dtros import DTROS, NodeType


class PlannerNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(PlannerNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.trigger_decision = 0
        self.duckie_x = 0
        self.duckie_y = 0
        self.duckie_theta = 0
        self.path_decision = STOP_ROBOT
        self.frame = "map"
        self.path = {1:[self.duckie_x, self.duckie_y, self.duckie_theta]}

        # construct publisher
        path_planning_topic = f"/{self._vehicle_name}/path_planner/coordinates"
        self.pub_path_cmd = rospy.Publisher(
        path_planning_topic,
        Pose2DStamped,
        queue_size=10
        )
        rviz_path_topic = f"/{self._vehicle_name}/path_planner/path"
        self.pub_rviz_path = rospy.Publisher(
        rviz_path_topic,
        Path,
        queue_size=50
        )

        orchestrator_ack_topic = f"/{self._vehicle_name}/path_planner/end"
        self.pub_ack = rospy.Publisher(
            orchestrator_ack_topic,
            Int8,
            queue_size=1
        )

        # Construct subscribers
        self.sub_trigger_input = rospy.Subscriber(
            f"/{self._vehicle_name}/orchestrator_node/mission",
            Polygon,
            self.decision_cb,
            buff_size=1000,
            queue_size=10,
        )

        self.sub_ducky_pos = rospy.Subscriber(
            f"/{self._vehicle_name}/robot_odometry/odometry",
            Odometry,
            self.duckiePosition_cb,
            buff_size=1000,
            queue_size=10,
        )



    def run(self):  
        while not rospy.is_shutdown():
            rospy.loginfo("Alive")
            self.executePath()
            

    def duckiePosition_cb(self, PoseMsg):
        self.duckie_x = PoseMsg.pose.pose.position.x
        self.duckie_y = PoseMsg.pose.pose.position.y
        self.duckie_theta = 2*math.asin(PoseMsg.pose.pose.orientation.z)
         
    def decision_cb(self, position):
        #callback that r-eceives trigger decision from tommy
        x1 = position.points[0].x
        y1 = position.points[0].y
        path_decision = int(position.points[0].z)
        self.path_decision = path_decision

        x2 = position.points[1].x
        y2 = position.points[1].y

        self.log(f"Received decision {path_decision} at shoePosition({x1},{y1}) and ({x2},{y2})")

        self.path = self.plan_path(path_decision, x1, y1, x2, y2) 

        #send the planned path to rviz
        rviz_path = Path()
        rviz_path.poses = [PoseStamped() for _ in self.path.items()]
        rviz_path.header.frame_id = self.frame
        rviz_path.header.stamp = rospy.Time.now()
        for id, coords in self.path.items():
                rviz_path.poses[int(id)-1].header.stamp = rospy.Time.now()
                rviz_path.poses[int(id)-1].header.frame_id = self.frame
                rviz_path.poses[int(id)-1].pose.position.x = coords[0]
                rviz_path.poses[int(id)-1].pose.position.y = coords[1]
                rviz_path.poses[int(id)-1].pose.position.z = 0
                rviz_path.poses[int(id)-1].pose.orientation.x = 0
                rviz_path.poses[int(id)-1].pose.orientation.y = 0
                rviz_path.poses[int(id)-1].pose.orientation.z = np.sin(self.duckie_theta / 2)
                rviz_path.poses[int(id)-1].pose.orientation.w = np.cos(self.duckie_theta / 2)
        self.pub_rviz_path.publish(rviz_path)
        self.log(f"Map sent")

             
    def pub_car_commands(self, x, y, theta):
        car_control_msg = Pose2DStamped()
        car_control_msg.header.stamp = rospy.Time.now()
        car_control_msg.x = x
        car_control_msg.y = y
        car_control_msg.theta = theta
        self.pub_path_cmd.publish(car_control_msg)

    def plan_path(self, path_decision, x1, y1, x2, y2):

        stopPath = {
                1:[self.duckie_x, self.duckie_y, self.duckie_theta]
            }

        maxRadius = 0.5
        midx = (x1+x2)/2
        midy = (y1+y2)/2
        angle = math.atan2(y2-y1, x2-x1)
        distance = math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))
        distance_centre = math.sqrt(math.pow(midx-x1, 2) + math.pow(midy-y1, 2))

        #eqn for the eight path
        if path_decision == VASILIS_EIGHT:
            eightPath = {}
            step_size = 18
            x_path, y_path, theta_path = 0, 0, 0
            for idx, degs in enumerate(range(0, 361, step_size)):
                x = (distance / 2 + maxRadius) * math.cos(math.radians(degs))
                y = maxRadius * math.sin(math.radians(degs * 2))
                x_path = midx + x * math.cos(angle) - y * math.sin(angle)
                y_path = midy + x * math.sin(angle) + y * math.cos(angle)

                x_new = (distance / 2 + maxRadius) * math.cos(math.radians(degs+step_size))
                y_new = maxRadius * math.sin(math.radians((degs+step_size) * 2))
                x_path_next = midx + x * math.cos(angle) - y * math.sin(angle)
                y_path_next = midy + x * math.sin(angle) + y * math.cos(angle)

                try:
                    theta_path = math.atan2((y_path_next-y_path),(x_path_next-x_path))
                except ZeroDivisionError:
                    if y_path_next-y_path > 0:
                        theta_path = np.deg2rad(90)
                    else:
                        theta_path = np.deg2rad(-90)

                coordinates = [x_path, y_path, theta_path]
                eightPath[idx] = coordinates
            return eightPath
        
        #eqn for the circle
        if path_decision == TOM_CIRCLE:
            circlePath = {}
            x_path, ypath, theta_path, x_path_next, y_path_next = 0, 0, 0, 0 , 0
            step_size = 18
            for idx, degs in enumerate(range(0, 361, step_size)):
                x_path = midx + (distance_centre+0.5)*math.cos(math.radians(degs))
                y_path = midy + (distance_centre+0.5)*math.sin(math.radians(degs))

                x_path_next = midx + (distance_centre+0.5)*math.cos(math.radians(degs+step_size))
                y_path_next = midy + (distance_centre+0.5)*math.sin(math.radians(degs+step_size))

                try:
                    theta_path = math.atan2((y_path_next-y_path),(x_path_next-x_path))
                except ZeroDivisionError:
                    if y_path_next-y_path > 0:
                        theta = np.deg2rad(90)
                    else:
                        theta = np.deg2rad(-90)

                coordinates = [x_path, y_path, theta_path]
                circlePath[idx] = coordinates
            return circlePath
            
        #eqn for circle around one shoe
        if path_decision == SHASHANK_ORBIT:
            orbitPath = {}
            x_path, y_path, theta_path, x_path_next, y_path_next = 0, 0, 0, 0, 0
            step_size = 18
            for idx, degs in enumerate(range(0, 361, step_size)):
                x_path = x1 + (maxRadius)*math.cos(math.radians(degs))
                y_path = y1 + (maxRadius)*math.sin(math.radians(degs))

                x_path_next = x1 + (maxRadius)*math.cos(math.radians(degs+step_size))
                y_path_next = y1 + (maxRadius)*math.sin(math.radians(degs+step_size))

                try:
                    theta_path = math.atan2((y_path_next-y_path),(x_path_next-x_path))
                except ZeroDivisionError:
                    if y_path_next-y_path > 0:
                        theta = np.deg2rad(90)
                    else:
                        theta = np.deg2rad(-90)

                coordinates = [x_path, y_path, theta_path]
                orbitPath[idx] = coordinates
            return orbitPath

        if path_decision == VARUN_FOLLOW:  
            theta = math.atan2((y1-self.duckie_y),(x1-self.duckie_x))
            #theta in radians
            followPath = {0:[self.duckie_x, self.duckie_y, self.duckie_theta], 1:[x1-0.1*math.cos(theta), y1-0.1*math.sin(theta), theta]}
            return followPath

        if path_decision == SERGE_LOOK:
            delta_theta = self.duckie_theta - math.atan2(y1-self.duckie_y,x1-self.duckie_x)
            lookPath = {0:[self.duckie_x, self.duckie_y, delta_theta]}
            return lookPath
        
        if path_decision == NOTHING_IN_SIGHT:
            sightPath = {   
                        0:[self.duckie_x, self.duckie_y, np.deg2rad(0)],
                        1:[self.duckie_x, self.duckie_y, np.deg2rad(45)],
                        2:[self.duckie_x, self.duckie_y, np.deg2rad(90)],
                        3:[self.duckie_x, self.duckie_y, np.deg2rad(135)],
                        4:[self.duckie_x, self.duckie_y, np.deg2rad(180)],
                        5:[self.duckie_x, self.duckie_y, np.deg2rad(-135)],
                        6:[self.duckie_x, self.duckie_y, np.deg2rad(-90)],
                        7:[self.duckie_x, self.duckie_y, np.deg2rad(-45)],
                        8:[self.duckie_x, self.duckie_y, np.deg2rad(0)]}
            return sightPath
        
        if path_decision == STOP_ROBOT:
            return stopPath
        
        return stopPath
        

    def executePath(self):
        current_path_decision = self.path_decision
        current_path = self.path

        min_distance = float('inf')
        closest_point = None
        points = list(current_path.keys())
        #set starting point by finding the point in path closest to the duckie
        for id, coords in current_path.items():
            distance = math.sqrt(math.pow(self.duckie_x-coords[0], 2) + math.pow(self.duckie_y-coords[1], 2))
            if distance < min_distance:
                min_distance = distance
                closest_point = id

        if closest_point is None:
            print("No closest point found. Path execution aborted.")
            return
    
        starting_point = points.index(closest_point)
        path_points = points[starting_point:] + points[:starting_point]



        if (current_path_decision in [VASILIS_EIGHT, SHASHANK_ORBIT, TOM_CIRCLE]):
            for waypoints in path_points:
                #print(f"waypoints: {waypoints}")
                #print(f"path: {current_path}")
                self.pub_car_commands(current_path[waypoints][0], current_path[waypoints][1], current_path[waypoints][2])
                error_x = current_path[waypoints][0] - self.duckie_x
                error_y = current_path[waypoints][1] - self.duckie_y
                error = np.sqrt(error_x**2 + error_y**2)
                while abs(error) > POSE_ERROR_RADIUS :
                    #give a radius of allowed error in pose
                    error_x = current_path[waypoints][0] - self.duckie_x
                    error_y = current_path[waypoints][1] - self.duckie_y
                    error = np.sqrt(error_x**2 + error_y**2)

                    self.log(f"error: {error} duckie error x: {error_x}, duckie y: {error_y}, waypoint x: {current_path[waypoints][0]}, waypoint y: {current_path[waypoints][1]}, duckie x: {self.duckie_x}, duckie y: {self.duckie_y}")
                    if (current_path_decision != self.path_decision):
                        return
                    
                    #self.log(f"Reaching point {coords[0]},{coords[1]}")
            self.log("Path finished!!! Receiving new path")
            self.pub_ack.publish(current_path_decision)
            self.path_decision = STOP_ROBOT
            self.path = {1:[self.duckie_x, self.duckie_y, self.duckie_theta]}

        elif current_path_decision == SERGE_LOOK:
            while current_path_decision == self.path_decision:
                for waypoints in path_points:
                    self.pub_car_commands(current_path[waypoints][0], current_path[waypoints][1], current_path[waypoints][2])
                    error_theta = current_path[waypoints][2] - self.duckie_theta
                    while abs(error_theta) > ANGLE_ERROR_RADIUS:
                        #give a radius of allowed error in pose
                        error_theta = current_path[waypoints][2] - self.duckie_theta
                        self.log(f"error: {error_theta}, waypoint theta: {current_path[waypoints][2]}, theta: {self.duckie_theta}")
                        if (current_path_decision != self.path_decision):
                            return
                        #self.log(f"Reaching point {coords[0]},{coords[1]}")
                self.log("Path finished!!! Receiving new path")
                self.pub_ack.publish(current_path_decision)
                self.path_decision = STOP_ROBOT
                self.path = {1:[self.duckie_x, self.duckie_y, self.duckie_theta]}
        elif current_path_decision == VARUN_FOLLOW:
            while current_path_decision == self.path_decision:
                for waypoints in path_points:
                    self.pub_car_commands(current_path[waypoints][0], current_path[waypoints][1], current_path[waypoints][2])
                    error_x = current_path[waypoints][0] - self.duckie_x
                    error_y = current_path[waypoints][1] - self.duckie_y
                    error = np.sqrt(error_x**2 + error_y**2)
                    while abs(error) > POSE_ERROR_RADIUS:
                        #give a radius of allowed error in pose
                        error_x = current_path[waypoints][0] - self.duckie_x
                        error_y = current_path[waypoints][1] - self.duckie_y
                        error = np.sqrt(error_x**2 + error_y**2)
                        self.log(f"error: {error} duckie error x: {error_x}, duckie y: {error_y}, waypoint x: {current_path[waypoints][0]}, waypoint y: {current_path[waypoints][1]}, duckie x: {self.duckie_x}, duckie y: {self.duckie_y}")
                        if (current_path_decision != self.path_decision):
                            return
                        #self.log(f"Reaching point {coords[0]},{coords[1]}")
                self.log("Path finished!!! Receiving new path")
                self.pub_ack.publish(current_path_decision)
                self.path_decision = STOP_ROBOT
                self.path = {1:[self.duckie_x, self.duckie_y, self.duckie_theta]}
        
        elif current_path_decision == NOTHING_IN_SIGHT:
            while current_path_decision == self.path_decision:
                for waypoints in path_points:
                    self.pub_car_commands(current_path[waypoints][0], current_path[waypoints][1], current_path[waypoints][2])
                    error_theta = current_path[waypoints][2] - self.duckie_theta
                    while abs(error_theta) > ANGLE_ERROR_RADIUS:
                        #give a radius of allowed error in pose
                        error_theta = current_path[waypoints][2] - self.duckie_theta
                        self.log(f"error: {error_theta}, waypoint theta: {current_path[waypoints][2]}, theta: {self.duckie_theta}")
                        if (current_path_decision != self.path_decision):
                            return
                        #self.log(f"Reaching point {coords[0]},{coords[1]}")

        else:
            while current_path_decision == self.path_decision:
                for waypoints in path_points:
                    self.pub_car_commands(current_path[waypoints][0], current_path[waypoints][1], current_path[waypoints][2])
                    error_x = current_path[waypoints][0] - self.duckie_x
                    error_y = current_path[waypoints][1] - self.duckie_y
                    error = np.sqrt(error_x**2 + error_y**2)
                    while abs(error) > POSE_ERROR_RADIUS_STOP:
                        #give a radius of allowed error in pose
                        error_x = current_path[waypoints][0] - self.duckie_x
                        error_y = current_path[waypoints][1] - self.duckie_y
                        error = np.sqrt(error_x**2 + error_y**2)
                        self.log(f"error: {error} duckie error x: {error_x}, duckie y: {error_y}, waypoint x: {current_path[waypoints][0]}, waypoint y: {current_path[waypoints][1]}, duckie x: {self.duckie_x}, duckie y: {self.duckie_y}")
                        if (current_path_decision != self.path_decision):
                            return
                        self.log(f"Reaching point {coords[0]},{coords[1]}")
                #do not send ack for stop and look around
    

        




if __name__ == '__main__':
    # create the node
    node = PlannerNode(node_name='path_planner_node')
    # run node
    node.run()
    # keep the process from terminating
    rospy.spin()
