#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        # self.lights = []

        # Added
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.light_classifier = None

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        # Commented out as no longer needed once the classifier has been implemented.
        # This caused issues with my pervious submissions as some of the code was relying on it.  The publisher never publishes.
        # sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.detector_ready = rospy.Publisher('/detector_ready', Bool, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.loop()
        # rospy.spin()

    def loop(self):
        rate = rospy.Rate(10)
        # Check ROS is still running
        while not rospy.is_shutdown():
            # Sleeps for 1/rate sec
            # Classifier takes time to load, so don't move car until it is ready.
            if self.light_classifier != None:
                self.detector_ready.publish(True)
            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # Store the waypoints in an instance variable.
        self.waypoints = waypoints
        # If instance variable is None, collect the x and y coordinates of the waypoints.
        # waypoints_2d are also only captured once.
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            # Using KDTree to make it easier to find the closest waypoint to the current pose.
            self.waypoint_tree = KDTree(self.waypoints_2d)

    # def traffic_cb(self, msg):
    #     # Store message
    #     self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        # Using the KDTree captured in the waypoint_cb calback
        # function to find the closest waypoint.
        return self.waypoint_tree.query([x, y], 1)[1]

    def get_light_state(self):
        """Determines the current color of the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Classifier takes time to load, so don't move car until it is ready.
        if self.light_classifier is None:
            return TrafficLight.RED
        if(not self.has_image):
            self.prev_light_loc = None
            return False
        # ROS function to convert message to an OpenCV format.
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification, light state provided for debug purposes.
        # return self.light_classifier.get_classification(cv_image, light.state)
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        light_found = False
        line_wp_idx = None

        # Get a list of stop line positions for the line just before the intersection starts.
        # Positions will be used to calculate deceleration waypoints.
        stop_line_positions = self.config['stop_line_positions']
        # Check if a pose has been recieved and waypoints exist.
        # When the waypoints are not being update, self.waypoints will be None, thus causing an exception.
        if(self.pose and self.waypoints != None):
            # What is the closest waypoint to our current position.
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            wp_len = len(self.waypoints.waypoints)
            for i in range(len(stop_line_positions)):
                # Get stop line position
                line = stop_line_positions[i]
                # Find the closest waypoint index to the stop line.
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Get the index difference
                idx_distance = temp_wp_idx - car_wp_idx
                # If the difference is positive and less than the number of waypoints
                if idx_distance >= 0 and idx_distance < wp_len:
                    wp_len = idx_distance
                    light_found = True
                    line_wp_idx = temp_wp_idx
                    # rospy.logwarn('Index {0}'.format(i))
        # If a close light was found.
        if light_found:
            # Classify if the traffic light is red, yellow, or green
            state = self.get_light_state()
            # rospy.logwarn('Light State: {0}'.format(state))
            # return where the stop line is and the traffic light state.
            return line_wp_idx, state
        # Clear the waypoints.
        self.waypoints = None
        # Return an unknown state.
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
