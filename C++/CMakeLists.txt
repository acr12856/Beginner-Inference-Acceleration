add_executable(main main.cpp)

target_include_directories(main PRIVATE ${ORT_INCLUDE_PATH})
target_link_directories(main PRIVATE ${ORT_LIB_PATH})
target_link_libraries(main PRIVATE onnxruntime)

add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/onnx_files/resnet-18.onnx"
    "${CMAKE_BINARY_DIR}/resnet-18.onnx"
)

set_target_properties(main PROPERTIES
    INSTALL_RPATH "${ORT_LIB_PATH}"
    BUILD_WITH_INSTALL_RPATH TRUE
)