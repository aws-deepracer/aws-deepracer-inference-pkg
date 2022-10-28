# AWS DeepRacer inference package

## Overview

The AWS DeepRacer inference ROS package creates the `inference_node`, which is part of the core AWS DeepRacer application and launches from the `deepracer_launcher`. For more information about the application and the components, see the [aws-deepracer-launcher repository](https://github.com/aws-deepracer/aws-deepracer-launcher).

This node is responsible for running the inference on the model that is selected using the Intel OpenVino Inference Engine APIs.

For more information about the Intel OpenVino Inference Engine, see the [Inference Engine Developer Guide](https://docs.openvinotoolkit.org/2021.1/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html).

## License

The source code is released under Apache 2.0 (https://aws.amazon.com/apache-2-0/).

## Installation
Follow these steps to install the AWS DeepRacer inference package.

### Prerequisites

The AWS DeepRacer device comes with all the prerequisite packages and libraries installed to run the `inference_pkg`. For more information about the pre-installed set of packages and libraries on the AWS DeepRacer, and about installing the required build systems, see [Getting started with AWS DeepRacer OpenSource](https://github.com/aws-deepracer/aws-deepracer-launcher/blob/main/getting-started.md).

The `inference_pkg` specifically depends on the following ROS 2 packages as build and run dependencies:

1. `deepracer_interfaces_pkg`: This package contains the custom message and service type definitions used across the AWS DeepRacer core application.
1. `cv_bridge`: This package contains CvBridge, which converts between ROS image messages and OpenCV images.
1. `image_transport`: This package provides transparent support for transporting images in low-bandwidth compressed formats.
1. `sensor_msgs`: This package defines messages for commonly used sensors, including cameras and scanning laser rangefinders.

## Downloading and building

Open a terminal on the AWS DeepRacer device and run the following commands as the root user.

1. Switch to the root user before you source the ROS 2 installation:

        sudo su

1. Source the ROS 2 Foxy setup bash script:

        source /opt/ros/foxy/setup.bash 

1. Set the environment variables required to run Intel OpenVino scripts:

        source /opt/intel/openvino_2021/bin/setupvars.sh

1. Create a workspace directory for the package:

        mkdir -p ~/deepracer_ws
        cd ~/deepracer_ws

1. Clone the `inference_pkg` on the AWS DeepRacer device:

        git clone https://github.com/aws-deepracer/aws-deepracer-inference-pkg.git

1. Fetch unreleased dependencies:

        cd ~/deepracer_ws/aws-deepracer-inference-pkg
        rosws update

1. Resolve the `inference_pkg` dependencies:

        cd ~/deepracer_ws/aws-deepracer-inference-pkg && rosdep install -i --from-path . --rosdistro foxy -y

1. Build the `inference_pkg` and `deepracer_interfaces_pkg`:

        cd ~/deepracer_ws/aws-deepracer-inference-pkg && colcon build --packages-select inference_pkg deepracer_interfaces_pkg

## Usage

The `inference_node` provides a very specific and core functionality to run inference on the reinforcement learning models that are trained on the AWS DeepRacer Simulator. Intel OpenVino provides APIs to load an intermediate representation file for the model and create a core object which can be used to run the inference. Although the node is built to work with the AWS DeepRacer application, it can be run independently for development, testing, and debugging purposes.

### Run the node

To launch the built `inference_node` as the root user on the AWS DeepRacer device, open another terminal on the AWS DeepRacer device and run the following commands as the root user:

1. Switch to the root user before you source the ROS 2 installation:

        sudo su

1. Source the the ROS 2 Foxy setup bash script:

        source /opt/ros/foxy/setup.bash 

1. Set the environment variables required to run Intel OpenVino scripts:

        source /opt/intel/openvino_2021/bin/setupvars.sh

1. Source the setup script for the installed packages:

        source ~/deepracer_ws/aws-deepracer-inference-pkg/install/setup.bash  

1. Launch the `inference_pkg` using the launch script:

        ros2 launch inference_pkg inference_pkg_launch.py

## Launch files

The `inference_pkg_launch.py`, included in this package, provides an example demonstrating how to launch the nodes independently from the core application.

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

## Node details

### `inference_node`

#### Subscribed topics

| Topic name | Message type | Description |
| ---------- | ------------ | ----------- |
|/`sensor_fusion_pkg`/`sensor_msg`|`EvoSensorMsg`|Message with the combined sensor data. Contains single camera or two camera images and LiDAR distance data.|


#### Published topics

| Topic name | Message type | Description |
| ---------- | ------------ | ----------- |
|/`inference_pkg`/`rl_results`|`InferResultsArray`|Publish a message with the reinforcement learning inference results with class probabilities for the state input passed through the current model that is selected in the device console.|


#### Services

| Service name | Service type | Description |
| ---------- | ------------ | ----------- |
|`load_model`|`LoadModelSrv`|Service that is responsible for setting pre-processing algorithm and inference tasks for the specific type of model loaded.|
|`inference_state`|`InferenceStateSrv`|Service that is responsible for starting and stopping inference tasks.|


### Parameters

| Parameter name   | Description  |
| ---------------- |  ----------- |
| `device` | String that is either `CPU`, `GPU` or `MYRIAD`. Default is `CPU`. `MYRIAD` is the Intel Compute Stick 2. |


## Resources

* [Getting started with AWS DeepRacer OpenSource](https://github.com/aws-deepracer/aws-deepracer-launcher/blob/main/getting-started.md)
