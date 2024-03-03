cmake_minimum_required(VERSION 3.5)
project(inference_pkg)
include(FetchContent)

set(ABSL_PROPAGATE_CXX_STD ON)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wno-deprecated-declarations -Wno-ignored-attributes)
endif()

FetchContent_Declare(tensorflow-lite
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
)
FetchContent_Populate(tensorflow-lite)

add_subdirectory(
  ${tensorflow-lite_SOURCE_DIR}/tensorflow/lite 
  ${tensorflow-lite_BINARY_DIR}
  EXCLUDE_FROM_ALL
)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(deepracer_interfaces_pkg REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ngraph REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV 4.2 QUIET
  COMPONENTS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
  CONFIG
)
if(NOT OpenCV_FOUND)
  find_package(OpenCV 3 REQUIRED
    COMPONENTS
      opencv_core
      opencv_imgproc
      opencv_imgcodecs
    CONFIG
  )
endif()

add_executable(inference_node
    src/inference_node.cpp
    src/tflite_inference_eng.cpp
    src/intel_inference_eng.cpp
    src/image_process.cpp
)

target_include_directories(inference_node PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${InferenceEngine_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers/include
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(inference_node  -lm -ldl
  ${OpenCV_LIBRARIES}
  tensorflow-lite
  ${InferenceEngine_LIBRARIES}
  ${NGRAPH_LIBRARIES}  
)

ament_target_dependencies(inference_node rclcpp deepracer_interfaces_pkg sensor_msgs std_msgs cv_bridge image_transport OpenCV InferenceEngine ngraph) 

install(TARGETS
  inference_node
  DESTINATION
  lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include)

# Install launch files.
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_export_include_directories(include)
ament_export_dependencies(inference_pkg sensor_msgs std_msgs)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()



ament_package()
