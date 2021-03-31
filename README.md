# DeepRacer Inference Package

## Overview

The DeepRacer Inference ROS package creates the *inference_node* which is part of the core AWS DeepRacer application and will be launched from the deepracer_launcher. More details about the application and the components can be found [here](https://github.com/aws-deepracer/aws-deepracer-launcher).

This node is responsible for running the inference on the model that is selected using the Intel OpenVino Inference Engine APIs.

More details about the Intel OpenVino Inference Engine can be found here:
https://docs.openvinotoolkit.org/2021.1/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html

## License

The source code is released under Apache 2.0 (https://aws.amazon.com/apache-2-0/).

## Installation

### Prerequisites

The DeepRacer device comes with all the pre-requisite packages and libraries installed to run the inference_pkg. More details about pre installed set of packages and libraries on the DeepRacer, and installing required build systems can be found in the [Getting Started](https://github.com/aws-deepracer/aws-deepracer-launcher/blob/main/getting-started.md) section of the AWS DeepRacer Opensource page.

The inference_pkg specifically depends on the following ROS2 packages as build and execute dependencies:

1. *deepracer_interfaces_pkg* - This packages contains the custom message and service type definitions used across the AWS DeepRacer core application.
1. *cv_bridge* - This contains CvBridge, which converts between ROS Image messages and OpenCV images.
1. *image_transport* - It provides transparent support for transporting images in low-bandwidth compressed formats.
1. *sensor_msgs* - This package defines messages for commonly used sensors, including cameras and scanning laser rangefinders.

## Downloading and Building

Open up a terminal on the DeepRacer device and run the following commands as root user.

1. Switch to root user before you source the ROS2 installation:

        sudo su

1. Source the ROS2 Foxy setup bash script:

        source /opt/ros/foxy/setup.bash 

1. Set the environment variables required to run Intel OpenVino scripts:

        source /opt/intel/openvino_2021/bin/setupvars.sh

1. Create a workspace directory for the package:

        mkdir -p ~/deepracer_ws
        cd ~/deepracer_ws

1. Clone the inference_pkg on the DeepRacer device:

        git clone https://github.com/aws-deepracer/aws-deepracer-inference-pkg.git

1. Fetch unreleased dependencies:

        cd ~/deepracer_ws/aws-deepracer-inference-pkg
        rosws update

1. Resolve the inference_pkg dependencies:

        cd ~/deepracer_ws/aws-deepracer-inference-pkg && rosdep install -i --from-path . --rosdistro foxy -y

1. Build the inference_pkg and deepracer_interfaces_pkg:

        cd ~/deepracer_ws/aws-deepracer-inference-pkg && colcon build --packages-select inference_pkg deepracer_interfaces_pkg

## Usage

The inference_node provides a very specific and core functionality to run inference on the Reinforcement learning models that are trained on the AWS DeepRacer Simulator. Intel OpenVino provides APIs to load an intermediate representation file for the model and create a core object which can be used to run the inference. Although the node is built to work with the AWS DeepRacer application, it can be run independently for development/testing/debugging purposes.

### Run the node

To launch the built inference_node as root user on the DeepRacer device open up another terminal on the DeepRacer device and run the following commands as root user:

1. Switch to root user before you source the ROS2 installation:

        sudo su

1. Source the ROS2 Foxy setup bash script:

        source /opt/ros/foxy/setup.bash 

1. Set the environment variables required to run Intel OpenVino scripts:

        source /opt/intel/openvino_2021/bin/setupvars.sh

1. Source the setup script for the installed packages:

        source ~/deepracer_ws/aws-deepracer-inference-pkg/install/setup.bash  

1. Launch the inference_pkg using the launch script:

        ros2 launch inference_pkg inference_pkg_launch.py

## Launch Files

The  inference_pkg_launch.py is also included in this package that gives an example of how to launch the nodes independently from the core application.

    from launch import LaunchDescription
    from launch_ros.actions import Node

    def generate_launch_description():
        return LaunchDescription([
            Node(
                package='inference_pkg',
                namespace='inference_pkg',
                executable='inference_node',
                name='inference_node'
            )
        ])

## Node Details

### inference_node

#### Subscribed Topics

| Topic Name | Message Type | Description |
| ---------- | ------------ | ----------- |
|/sensor_fusion_pkg/sensor_msg|EvoSensorMsg|Message with the combined sensor data. Contains single camera/two camera images and LiDAR distance data.|


#### Published Topics

| Topic Name | Message Type | Description |
| ---------- | ------------ | ----------- |
|/inference_pkg/rl_results|InferResultsArray|Publish a message with the reinforcement learning inference results with class probabilities for the state input passed through the current model that is selected in the device console.|


#### Services

| Service Name | Service Type | Description |
| ---------- | ------------ | ----------- |
|load_model|LoadModelSrv|Service that is responsible for setting pre-processing algorithm and inference task for the specific type of model loaded.|
|inference_state|InferenceStateSrv|Service that is responsible for starting and stopping inference tasks.|

## Resources

* AWS DeepRacer Opensource getting started: [https://github.com/aws-deepracer/aws-deepracer-launcher/blob/main/getting-started.md](https://github.com/aws-deepracer/aws-deepracer-launcher/blob/main/getting-started.md)
