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

#ifndef INFERENCE_BASE_HPP
#define INFERENCE_BASE_HPP

#include "inference_pkg/image_process.hpp"
#include <memory>

namespace InferTask {
    class InferenceBase
    {
    /// Interface for all inference task. Subclasses are responsible for 
    /// specifying a device to run inference on, publishing topics, and 
    /// parsing the results.
    public:
        InferenceBase() = default;
        ~InferenceBase() = default;
        /// @returns True if model loaded successfully, false otherwise
        /// @param artifactPath Path to the model artifact.
        /// @param imgProcess Pointer to the image processing algorithm
        virtual bool loadModel(const char* artifactPath,
                               std::shared_ptr<ImgProcessBase> imgProcess) = 0;
        /// Starts the inference task until stopped.
        virtual void startInference() = 0;
        /// Stops the inference task if running.
        virtual void stopInference() = 0;
    };
}
#endif