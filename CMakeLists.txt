cmake_minimum_required(VERSION 3.19)
project(mnist_neural_net)

set(CMAKE_CXX_STANDARD 14)

set(EIGEN_DIR "C:/Users/Will/OneDrive/Projects/include/eigen")
include_directories(${EIGEN_DIR})

add_executable(mnist_neural_net main.cpp CSVReader.h EigenUtils.h PreprocessingUtils.h Metrics.h)