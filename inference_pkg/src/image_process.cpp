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

#include "inference_pkg/image_process.hpp"

namespace {
    /// Constants fro pre-processing
    //! TODO Make these configurable once we have a handle on what the user will be
    //! expected to input.
    const int THRESHOLD = 0;
    const int PXL_MAX_VALUE = 255;
    const int ROW_IDX = 40;
    const int MASK_VALUE = 0;
    /// Helper method that converts ROS image messages to CV objects and resizes the
    /// cv object according to the params map.
    /// @param frameData ROS image message containing the image data.
    /// @param retImg Reference to CV object to be populated the with resized image.
    /// @param params Hash map containing resize information.
    bool cvtToCVObjResize (const sensor_msgs::msg::Image &frameData, cv::Mat &retImg,
                           const std::unordered_map<std::string, int> &params) {

        cv_bridge::CvImagePtr cvPtr;

        try {
            cvPtr = cv_bridge::toCvCopy(frameData, "bgr8");
        }
        catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "cv_bridge exception: %s", e.what());
            return false;
        }

        auto itWidth = params.find("width");
        if (itWidth == params.end()) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Image processing params has no width entry");
            return false; 
        }

        auto itHeight = params.find("height");
        if (itHeight == params.end()) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Image processing params has no height entry");
            return false; 
        }

        try {
             cv::resize(cvPtr->image, retImg, cv::Size(itWidth->second, itHeight->second));
        }
        catch (...) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Resize failed");
            return false;
        }
        return true;
    }
    /// Helper method that places the currImg at the from of imageStack.
    /// @param currImg Image to be placed at the front of the stack.
    /// @param retImg Reference to the image that will be passed to the inference engine
    /// @param imageStack Vector who's first entry is to be currImg
    /// @param params Hash map of the relevant model parameters.
    void stack(cv::Mat &currImg, cv::Mat &retImg, 
               std::vector<cv::Mat> &imageStack, const std::unordered_map<std::string, int> &params) {
        auto itChannels = params.find("channels");
        if (itChannels == params.end()) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Image processing params has no channel entry");
            return;
        }
        // Add image to the stack.
        if (imageStack.empty()) {
            for (int i = 0; i < itChannels->second; ++i) {
                imageStack.push_back(currImg);
            }
        }
        else {
        // Remove the oldest image which should be in the back
            imageStack.pop_back();
            // Add the current image to the front of the vector, this will be of order
            // N, if OpenCV refactors cv::merge to used std::deque refactor immediately.
            imageStack.insert(imageStack.begin(), currImg);
        }
        // Populate the return image with the image stack
        cv::merge(imageStack, retImg);
    }
    /// Helper method that masks the top half of an image.
    /// @param retImg Reference to the image that will be passed to the inference engine
    /// @param rowStopIdx Row index to terminate the masking.
    /// @param maskValue Pixel value for the mask
    void masking(cv::Mat &retImg, int rowStopIdx, int maskValue) {
        if (rowStopIdx > retImg.rows) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Invalid mask range");
            return;
        }
        for(int j = 0; j < rowStopIdx; j++) {
            for (int i = 0; i < retImg.cols; i++) {
                retImg.at<uchar>(j,i) = maskValue;
            }
        }
    }
    /// Helper method that applies thresholding
    /// @param retImg Reference to the image that will be passed to the inference engine
    /// @param thresh Threshold value
    /// @param maximum value to use with cv::THRESH_BINARY
    void threshold(cv::Mat &retImg, int thresh, int maxValue) {
        try {
            cv::threshold(retImg, retImg, thresh, maxValue, cv::THRESH_BINARY + cv::THRESH_OTSU);
        }
        catch (...) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Threshold failed");
        }
    }
}

namespace InferTask {
    void RGB::processImage(const sensor_msgs::msg::Image &frameData, cv::Mat &retImg,
                           const std::unordered_map<std::string, int> &params) {
        cvtToCVObjResize(frameData, retImg, params);
    }

    const std::string RGB::getEncode() const {
        return sensor_msgs::image_encodings::BGR8;
    }

    Grey::Grey(bool isThreshold, bool isMask)
     : isThreshold_(isThreshold),
       isMask_(isMask)
    {

    }

    void Grey::processImage(const sensor_msgs::msg::Image &frameData, cv::Mat &retImg,
                            const std::unordered_map<std::string, int> &params) {
        cv::Mat currImg;
        if (cvtToCVObjResize(frameData, currImg, params)) {
            try {
                // Convert to greyscale
                cv::cvtColor(currImg, currImg, cv::COLOR_BGR2GRAY);
                // Perform desired pre processing
                if (isThreshold_) {
                    threshold(currImg, THRESHOLD, PXL_MAX_VALUE);
                }
                if (isMask_) {
                    masking(currImg, ROW_IDX, MASK_VALUE);
                }
                stack(currImg, retImg, imageStack_, params);
            }
            catch (...) {
                RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Conversion to Grey Scale and vector stacking failed");
                return;
            }
        }
    }

    void Grey::processImageVec(const std::vector<sensor_msgs::msg::Image> &frameDataArr, cv::Mat &retImg,
                            const std::unordered_map<std::string, int> &params) {
        // Left camera image is sent as the top image and the right camera image is sent as second in the vector.
        // Stack operation replaces the beginning values as we loop through and hence we loop in decreasing order
        try {
              std::vector<cv::Mat> channels;
              for (const auto& image_msg : frameDataArr) {
                cv::Mat img;
                cvtToCVObjResize(image_msg, img, params);
                cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
                channels.push_back(img);
              }
              cv::merge(channels, retImg);
        }
        catch (...) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Conversion to Grey Scale and vector stacking failed");
            return;
        }
    }

    void Grey::reset() {
        imageStack_.clear();
    }

    const std::string Grey::getEncode() const {
        return sensor_msgs::image_encodings::MONO8;
    }

    void GreyDiff::processImage(const sensor_msgs::msg::Image &frameData, cv::Mat &retImg,
                            const std::unordered_map<std::string, int> &params) {
        (void)retImg;                    
        cv::Mat currImg;
        if (cvtToCVObjResize(frameData, currImg, params)) {
            try {
                // Convert to greyscale
                cv::cvtColor(currImg, currImg, cv::COLOR_BGR2GRAY);
                bool isFirstImg = prevImage_.empty();
                prevImage_.copyTo(currImg);
                if (!isFirstImg) {
                    currImg = currImg - prevImage_;
                }
            }
            catch (...) {
                RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Conversion to Grey Scale and vector stacking failed");
                return;
            }
        }
    }

    void GreyDiff::reset() {
        prevImage_.release();
    }

    const std::string GreyDiff::getEncode() const {
        return sensor_msgs::image_encodings::MONO8;
    }
}
