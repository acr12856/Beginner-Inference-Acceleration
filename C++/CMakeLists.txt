cmake_minimum_required(VERSION 3.14)
project(onnx_cpp_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ORT_LIB_PATH "<path to onnxruntime build library>")
set(ORT_INCLUDE_PATH "<path to onnxruntime include library>")

add_subdirectory(src)