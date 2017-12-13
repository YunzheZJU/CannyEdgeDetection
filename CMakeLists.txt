cmake_minimum_required(VERSION 3.9)
project(OpenCV_A1)

set(OpenCV_DIR "D:\\opencv\\mingw-build\\install")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 98)

add_executable(OpenCV_A1 main.cpp head.h)

# Add libs we need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_videoio)

# Linking
target_link_libraries(OpenCV_A1 ${OpenCV_LIBS})