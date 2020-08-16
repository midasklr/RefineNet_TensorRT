CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(refinenet)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

file(GLOB headers "${CMAKE_CURRENT_LIST_DIR}/*.h)")
set(exported_headers ${headers})

include_directories(${PROJECT_SOURCE_DIR})
include_directories(/home/kong/TensorRT-5.1.5.0/samples/common)
set(TENSORRT_INCLUDE_DIR /home/kong/TensorRT-5.1.5.0/include)
include_directories(TENSORRT_INCLUDE_DIR)
link_directories(/home/kong/TensorRT-5.1.5.0/lib)


find_package(OpenCV 3.4.8 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs)
find_package(CUDA REQUIRED)
set(TENSORRT_LIBRARY_INFER  /home/kong/TensorRT-5.1.5.0/include)
#set(TENSORRT_LIBRARY_INFER_PLUGIN  /home/kong/TensorRT-5.1.5.0/include)
#set(TENSORRT_LIBRARY_PARSER  /home/kong/TensorRT-5.1.5.0/include)
#set(TENSORRT_LIBRARY_ONNXPARSER  /home/kong/TensorRT-5.1.5.0/include)

#find_path(TENSORRT_INCLUDE_DIR NvInfer.h
#  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
#  PATH_SUFFIXES include)
#MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
#find_library(TENSORRT_LIBRARY_INFER nvinfer
#  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
#  PATH_SUFFIXES lib lib64 lib/x64)
#find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
#  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
#  PATH_SUFFIXES lib lib64 lib/x64)
#find_library(TENSORRT_LIBRARY_PARSER nvparsers
#  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
#  PATH_SUFFIXES lib lib64 lib/x64)
#find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
#  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
#  PATH_SUFFIXES lib lib64 lib/x64)


set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER} ${TENSORRT_LIBRARY_ONNXPARSER})
set(TENSORRT_LIBRARY nvinfer nvinfer_plugin nvparsers nvonnxparser)
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()


ADD_EXECUTABLE(RefineNet RefineNet.cpp image.cpp ${headers})

target_include_directories(RefineNet PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
# my cuda is 10.1 ，if your cuda isn‘t cuda10.1， please change this line
TARGET_LINK_LIBRARIES(RefineNet ${OpenCV_LIBS} ${TENSORRT_LIBRARY} /usr/local/cuda-10.1/lib64/libcudart.so.10.1)
