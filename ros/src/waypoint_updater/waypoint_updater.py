#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
import math

from scipy.spatial import KDTree
import numpy as np
from std_msgs.msg import Int32

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

'''
Message Information:

        std_msgs/Header header
          uint32 seq
          time stamp
          string frame_id
        styx_msgs/Waypoint[] waypoints
          geometry_msgs/PoseStamped pose
            std_msgs/Header header
              uint32 seq
              time stamp
              string frame_id
            geometry_msgs/Pose pose
              geometry_msgs/Point position
                float64 x
                float64 y
                float64 z
              geometry_msgs/Quaternion orientation
                float64 x
                float64 y
                float64 z
                float64 w
          geometry_msgs/TwistStamped twist
            std_msgs/Header header
              uint32 seq
              time stamp
              string frame_id
            geometry_msgs/Twist twist
              geometry_msgs/Vector3 linear
                float64 x
                float64 y
                float64 z
              geometry_msgs/Vector3 angular
                float64 x
                float64 y
                float64 z
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.waypoints_2d = None
        self.waypoint_tree = None

        self.base_lane = None
        self.stopline_wp_idx = -1

        # Original code given by Udacity
        # rospy.spin()
        # Allows control over the publishing frequency, where rospy.spin() does not.
        self.loop()

    def loop(self):
        # Set periodic run every 20ms (50Hz)
        rate = rospy.Rate(50)
        # Check ROS is still running
        while not rospy.is_shutdown():
            # Check if pose and base_lane are not None
            if self.pose and self.base_lane:
                self.publish_waypoints()

            # Sleeps for 1/rate sec
            rate.sleep()

    def get_closest_waypoint_idx(self):
        # Get current pose
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        # Closest waypoint index to current pose.
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Equation for hyperplane through closest_coordinate
        closest_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        # Cars position
        pos_vect = np.array([x, y])

        # By subtracting the two points we get the resulting vector.
        value = np.dot(closest_vect-prev_vect, pos_vect-closest_vect)
        # If the dot product is negative we know the closest waypoint is in front of the car.
        # If the dot product is positive we know the closest waypoint is behind the car.
        if value > 0:
            # Take the next index.
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        # Local Variables
        lane = Lane()
        # Get the closest index to the cars current pose (positon).
        closest_idx = self.get_closest_waypoint_idx()
        # Get X number of waypoints ahead of the closest waypoint to the cars pose.
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        # Get waypoint values from index range.
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
        # If there is no stop line  or the stop line is to far from the car to be considered.
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            # Set waypoints.
            lane.waypoints = base_waypoints
        else:
            # Traffic light requires the car to stop, so deceleration waypoints must be generated to stop the car.
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        # Local Variables
        temp = []

        for i, wp in enumerate(waypoints):
            # Create waypoint
            p = Waypoint()
            # Get current pose from waypoint.
            p.pose = wp.pose
            # Get two waypoints back from line so front of car stops at line
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            velocity = math.sqrt(2 * MAX_DECEL * dist)

            if velocity < 1.0:
                velocity = 0.0

            p.twist.twist.linear.x = min(velocity, wp.twist.twist.linear.x)
            # Append the new waypoint.
            temp.append(p)

        return temp

    def pose_cb(self, msg):
        # TODO: Implement
        # Get the message from node
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        # Store the waypoint
        self.base_lane = waypoints
        # Make sure that waypoints_2d is not None.
        if not self.waypoints_2d:
            # Extract the x and y values and store them in a list.
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            # KD Tree is a faster method for finding the closest waypoint.
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
