cmake_minimum_required(VERSION 3.22)
project(nano_track_trt_text)

set(CMAKE_CXX_STANDARD 14)

# OpenCV
# change this DIR to your opencv dir
set(OpenCV_DIR /home/lizhang/opencv-4.x/build)
find_package(OpenCV)

# TensorRT
# change this dir to your tensorrt dir
set(TensorRT_INCLUDE_DIRS /home/lizhang/TensorRT-8.5.2.2/include)
set(TensorRT_LIBRARIES /home/lizhang/TensorRT-8.5.2.2/lib)

#CUDA
find_package(CUDA REQUIRED)
link_directories(/usr/local/cuda/lib64)

include_directories(
        ${TensorRT_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/common.h
        ${CMAKE_SOURCE_DIR}/include/Nano_track.h
)

add_executable(nano_track_trt main.cpp)

target_link_libraries(nano_track_trt
        ${OpenCV_LIBS}
        libcudart.so
        libnvinfer.so
        libnvinfer_plugin.so
        ${TensorRT_LIBRARIES}
        ${CUDA_LIBRARIES}
        )
