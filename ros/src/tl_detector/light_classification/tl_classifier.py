from styx_msgs.msg import TrafficLight
import tensorflow as tf
import PIL
import cv2
import numpy as np
from PIL import Image
# import visualization_utils as vis_util
import os
import rospy

# PATH_TO_FROZEN_GRAPH = os.path.join('.', 'export', 'frozen_inference_graph.pb')

def get_class_name(value):
    value = int(value)
    if value == 1:
        return TrafficLight.RED
    elif value == 2:
        return TrafficLight.YELLOW
    elif value == 3:
        return TrafficLight.GREEN
    else:
        return TrafficLight.UNKNOWN

class TLClassifier(object):
    def __init__(self, detection_graph, session, image_tensor, detection_boxes, detection_scores, detection_classes, detection_number):
        #TODO load classifier
        self.light_gt_count = 0
        self.frame_count = 0
        self.detected_traffic_light = TrafficLight.UNKNOWN
        self.previous_detected_traffic_light = TrafficLight.UNKNOWN
        self.detection_graph = detection_graph
        self.session = session
        self.image_tensor = image_tensor
        self.detection_boxes = detection_boxes
        self.detection_scores = detection_scores
        self.detection_classes = detection_classes
        self.detection_number = detection_number
        self.visualize = False
        self.category_index = {1: {'name': 'traffic_light_red', 'id': 1}, 2: {'name': 'traffic_light_yellow', 'id': 2}, 3: {'name': 'traffic_light_green', 'id': 3}}

        if self.visualize:
            if not os.path.exists(os.path.join('.', 'images')):
                os.mkdir(os.path.join('.', 'images'))
            if not os.path.exists(os.path.join('.', 'images', 'processed')):
                os.mkdir(os.path.join('.', 'images', 'processed'))
            if not os.path.exists(os.path.join('.', 'images', 'raw')):
                os.mkdir(os.path.join('.', 'images', 'raw'))


        # Load frozen tensorflow model into memory.
        # From Tensorflow Object Detection API and Udacity's object detection lab.
        # I used the r1.5 branch since I am using Cuda Toolkit 9.0 with cuDNN 7.0
        # https://github.com/tensorflow/models/tree/r1.5
        # The virtual machine is using Tensorflow 1.3 CPU version, so modifications
        # in producing the frozen graph needed to be done to be compatiable with Tensorflow 1.3
        # self.detection_graph = tf.Graph()
        # with self.detection_graph.as_default():
        #     od_graph_def = tf.GraphDef()
        #     with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        #         serialized_graph = fid.read()
        #         od_graph_def.ParseFromString(serialized_graph)
        #         tf.import_graph_def(od_graph_def, name='')
        #
        # self.session = tf.Session(graph=self.detection_graph)
        # self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        # self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        # self.detection_number = self.detection_graph.get_tensor_by_name('num_detections:0')


    def get_classification(self, image, light_gt):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        # Process image every 4th frame
        if self.frame_count >= 3:

            self.frame_count = 0

            # if self.visualize:
            #     if light_gt == TrafficLight.RED:
            #         cv2.imwrite(os.path.join('.', 'images', 'raw', '{1}_train_2_{0}.jpg'.format('red', self.light_gt_count)), image)
            #     elif light_gt == TrafficLight.YELLOW:
            #         cv2.imwrite(os.path.join('.', 'images', 'raw', '{1}_train_2_{0}.jpg'.format('yellow', self.light_gt_count)), image)
            #     elif light_gt == TrafficLight.GREEN:
            #         cv2.imwrite(os.path.join('.', 'images', 'raw', '{1}_train_2_{0}.jpg'.format('green', self.light_gt_count)), image)


            # Convert BGR image to RGB image.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Convert to numpy array.
            image = np.asarray(image).astype(np.uint8)

            # Run tensorflow session with current graph
            with self.detection_graph.as_default():
                # The image is expected to be in the format of (batch, height width, channels)
                # However, since we only feed one image at the time batch will be set to 1.
                image_expanded = np.expand_dims(image, axis=0)
                # Run the retrained ssd_mobilenet_v1.
                (boxes, scores, classes, num) = self.session.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.detection_number],
                    feed_dict={self.image_tensor: image_expanded})

            # The output format is (batch, height width, channels, so remove the batch dimention from the arrays.
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)

            # if self.visualize:
            #     # Visual tools are provided by the Tensorflow Object Detection API.
            #     vis_util.visualize_boxes_and_labels_on_image_array(
            #         image, boxes, classes, scores,
            #         self.category_index,
            #         use_normalized_coordinates=True,
            #         min_score_thresh=0.7,
            #         line_thickness=3)
            #
            #     result = Image.fromarray((image).astype(np.uint8))
            #
            #     self.detected_traffic_light = TrafficLight.UNKNOWN
            #
            #     if light_gt == TrafficLight.RED:
            #         result.save(os.path.join('.', 'images', 'processed', '{1}_train_2_{0}.jpg'.format('red', self.light_gt_count)))
            #         # Set the ground truth as the detected light for debugging.
            #         self.detected_traffic_light  = TrafficLight.RED
            #     elif light_gt == TrafficLight.YELLOW:
            #         result.save(os.path.join('.', 'images', 'processed', '{1}_train_2_{0}.jpg'.format('yellow', self.light_gt_count)))
            #         # Set the ground truth as the detected light for debugging.
            #         self.detected_traffic_light  = TrafficLight.YELLOW
            #     elif light_gt == TrafficLight.GREEN:
            #         result.save(os.path.join('.', 'images', 'processed', '{1}_train_2_{0}.jpg'.format('green', self.light_gt_count)))
            #         # Set the ground truth as the detected light for debugging.
            #         self.detected_traffic_light = TrafficLight.GREEN
            #
            #     self.light_gt_count += 1

            detected_classes = []
            # Find the 3 top indices with the highest scores.
            top_3_score_indices = sorted(range(len(scores)), key=lambda i: scores[i])[-3:]
            # Check if the scores are greater than or equal to 80%.
            for i in top_3_score_indices:
                if scores[i] >= 0.80:
                    detected_classes.append(classes[i])
            # Check to see if there are any detections.
            if(len(detected_classes) != 0):
                # Take the first detection found.
                self.detected_traffic_light = get_class_name(detected_classes[0])
            else:
                self.detected_traffic_light = TrafficLight.UNKNOWN

            self.previous_detected_traffic_light = self.detected_traffic_light
        # Keep previous light state until next 4th frame.
        else:
            self.frame_count += 1
            self.detected_traffic_light = self.previous_detected_traffic_light

        # if self.visualize:
        #     rospy.logwarn('Detected Traffic Light: {0}'.format(self.detected_traffic_light))

        return self.detected_traffic_light
