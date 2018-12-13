# Udacity Self-Driving Car Nanodegree Final Project

This is my submission for the final project (Individual Submission).  In the **Short Summary** section, I will give an overview of the 3 main areas of this project that I had to implement.  Traffic light detection and classification was the biggest part I had to implement.

### Short Summary

1. Traffic Light Detection and Classification
  - Used the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/r1.5/research/object_detection) for traffic light classification.
    - Note that branch r1.5 was used.
    - Pretrained model used: [ssd_mobilenet_v1_coco_11_06_2017](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz)
    - Model was trained using Tensorflow GPU version 1.5 and the frozen graph was generated using Tensorflow CPU version 1.3 to satisfy project requirements.
    - Model was trained over 30,000 steps.
    - 152 images are used for training (half of the images are just flipped images to double the amount of data with little effort).
    - 36 images are used for testing.
  - [LabelImg](https://github.com/tzutalin/labelImg) Annotation Tool
    - Used for labelling images collected from the simulator.  There are better tools, but LabelImg was perfect for my needs and was free.
  - Convert XML to record
    - Used the `generate_tfrecord.py` and `xml_to_csv.py` from the [raccoon dataset](https://github.com/datitran/raccoon_dataset) to create record files needed for the Tensorflow Object Detection API.  The scripts are small and easy to modify for individual usage. I combined the scrips since there really is no need for the csv file (just an intermediary step in order to create a record file).  The [tutorial](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9) I followed was trying to classify raccoons using the API and is how I learned how to use the API.
  - Detection and Classification
    - Since I was using a virtual machine, I limited the processing to once every 4th frame.  The car responded much quicker to changing lights after this change.  Without this change, the light would turn green and the car would remain there for a few seconds before accelerating.
    - After processing the image, a detected traffic light must have a confidence of 80% or more to be considered valid for the vehicle to act on.  Generally a detection of a traffic light with lower than 80% confidence is usually too far for an action to be taken by the vehicle, so I was not too concerned about those detections.  The vehicle won't consider decelerating until it is approaching the stop line.  Also out of the 3 traffic lights only one needs to be detected for an action to be taken.  Since the model does very well, there was no need to implement addition confidence metrics.
    - In order to debug what was detected I used to `visualization_utils.py` provided by the Tensorflow Object Detection API.  This returns images with overlays of the detected objects and there confidence.
- Waypoint Updater
  - The waypoint updater node only executes every 20ms (50Hz).
  - Waypoints for the track are only published once, so there is no need to constantly update them.  
  - KDTree library is used to find the closest waypoint to the ego vehicles current position.  However, this point can be behind the ego vehicles current position, so by calculating the hyperplane we can check if this is the case.  A positive value means the closest waypoint is behind the car and therefore the next closest waypoint is chosen. If the value is negative the current closest waypoint is chosen.
  - The project provides stop line position for each traffic light.  After finding the closest waypoint to the ego vehicles position, we can calculate the next set of waypoints the car will use.  If the waypoint updater node gets a stop line position of a future red traffic light a deceleration path is created instead of an acceleration path.  Stop line positions are only issued if a red traffic light has been detected.
- Drive by Wire
  - Yaw Controller
    - Steers the car based on the linear and angular velocities.  The yaw controller was provided in the repository.
  - Low Pass Filter
    - Filters the high frequency noise out of the velocity.  The low pass filter was provided in the repository.  The only work here was to set the cutoff frequency and sample time.  I believe this low pass filter is provided for testing on a real vehicle.
  - PID Controller
    - Used to reach our target velocity.  The PID controller was provided in the repository.  The only work here is to use the base values provided in the walk-through or to use twiddle to optimize the values further.


### Noticable

1. The car seems to take a while after the light turns green, event the GT is lagging a bit.


# Udacity Original Readme Contents


For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
