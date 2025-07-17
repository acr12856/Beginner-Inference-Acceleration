cmake_minimum_required(VERSION 3.14)
project(onnx_cpp_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ORT_LIB_PATH "/Users/amogh/Documents/Projects/onnxruntime/build/MacOS/Release")
set(ORT_INCLUDE_PATH "/Users/amogh/Documents/Projects/onnxruntime/include/onnxruntime/core/session")

add_subdirectory(src)