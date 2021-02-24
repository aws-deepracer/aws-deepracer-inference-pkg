///////////////////////////////////////////////////////////////////////////////////
//   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.          //
//                                                                               //
//   Licensed under the Apache License, Version 2.0 (the "License").             //
//   You may not use this file except in compliance with the License.            //
//   You may obtain a copy of the License at                                     //
//                                                                               //
//       http://www.apache.org/licenses/LICENSE-2.0                              //
//                                                                               //
//   Unless required by applicable law or agreed to in writing, software         //
//   distributed under the License is distributed on an "AS IS" BASIS,           //
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    //
//   See the License for the specific language governing permissions and         //
//   limitations under the License.                                              //
///////////////////////////////////////////////////////////////////////////////////

#ifndef IMAGE_PROCESS_HPP
#define IMAGE_PROCESS_HPP

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include <unordered_map>

namespace InferTask {
    class ImgProcessBase
    {
    /// Simple base class for image process algorithms
    public:
        ImgProcessBase() = default;
        ~ImgProcessBase() = default;
        /// Applies the image processing algorithm.
        /// @param frameData ROS message containing the image data.
        /// @param retImg Open CV Mat object that will be used to store the post processed image
        /// @param params Hash map containing relevant pre-processing parameters
        virtual void processImage(const sensor_msgs::msg::Image &frameData, cv::Mat& retImg,
                                  const std::unordered_map<std::string, int> &params) = 0;
        virtual void processImageVec(const std::vector<sensor_msgs::msg::Image> &frameDataArr, cv::Mat& retImg,
                                     const std::unordered_map<std::string, int> &params) = 0;
        /// Resets the image processing algorithms data if any.
        virtual void reset() = 0;
        /// @returns String containing the color encoding of the image after processing
        virtual const std::string getEncode() const = 0;
    };

    class RGB : public ImgProcessBase
    {
    /// Algorithm for down sampling RGB images.
    public:
        RGB() = default;
        virtual ~RGB() = default;
        virtual void processImage(const sensor_msgs::msg::Image &frameData, cv::Mat& retImg,
                                  const std::unordered_map<std::string, int> &params) override;
        virtual void processImageVec(const std::vector<sensor_msgs::msg::Image> &frameDataArr, cv::Mat& retImg,
                                     const std::unordered_map<std::string, int> &params) override {(void)frameDataArr;(void)retImg;(void)params;}
        virtual void reset() override {}
        virtual const std::string getEncode() const;
    };

    class Grey : public ImgProcessBase
    {
    /// Algorithm for converting BGR images to greyscale, will create a stack
    /// of images based on the number of channels. Can be configured to threshold
    /// and mask.
    public:
        /// @param isThreshold True if thresholding should be performed on the image
        /// @param isMask True if background masking should be performed on the image.
        Grey(bool isThreshold, bool isMask);
        virtual ~Grey() = default;
        virtual void processImage(const sensor_msgs::msg::Image &frameData, cv::Mat& retImg,
                                  const std::unordered_map<std::string, int> &params) override;
        virtual void processImageVec(const std::vector<sensor_msgs::msg::Image> &frameDataArr, cv::Mat& retImg,
                                     const std::unordered_map<std::string, int> &params);
        virtual void reset() override;
        virtual const std::string getEncode() const;
    private:
        /// This container will hold all the necessary images, it will work as a queue.
        /// A vector is used instead of a queue to avaid extra cv:Mat copeis. cv:merge
        /// has no std:queue implementation.
        std::vector<cv::Mat> imageStack_;
        /// If true perform thresholding.
        const bool isThreshold_;
        /// If true mask out the background.
        const bool isMask_;
    };

    class GreyDiff : public ImgProcessBase
    {
    /// Algorithm for converting BGR images to greyscale and generating an image that is a
    /// diff between the current and previous image.
    public:
        GreyDiff() = default;
        virtual ~GreyDiff() = default;
        virtual void processImage(const sensor_msgs::msg::Image &frameData, cv::Mat& retImg,
                                  const std::unordered_map<std::string, int> &params) override;
        virtual void processImageVec(const std::vector<sensor_msgs::msg::Image> &frameDataArr, cv::Mat& retImg,
                                     const std::unordered_map<std::string, int> &params) override {(void)frameDataArr;(void)retImg;(void)params;}
        virtual void reset() override;
        virtual const std::string getEncode() const;
    private:
        /// Store the previous image in this member, so that we can diff the current image.
        cv::Mat prevImage_;
    };
}
#endif