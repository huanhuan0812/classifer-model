# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(FileProcessor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

# 使用Homebrew的ONNX Runtime
set(ONNXRUNTIME_INCLUDE_DIR "/opt/homebrew/include/onnxruntime")
set(ONNXRUNTIME_LIBRARY "/opt/homebrew/lib/libonnxruntime.dylib")

# 检查ONNX Runtime是否存在
if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h" AND EXISTS "${ONNXRUNTIME_LIBRARY}")
    message(STATUS "找到ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
    set(ONNX_FOUND TRUE)
else()
    message(WARNING "未找到ONNX Runtime，将编译简化版本")
    set(ONNX_FOUND FALSE)
endif()

# cppjieba路径
set(CPPJIEBA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cppjieba")
set(CPPJIEBA_INCLUDE_DIR "${CPPJIEBA_DIR}/include")

# 包含目录
include_directories(${CPPJIEBA_INCLUDE_DIR})
if(ONNX_FOUND)
    include_directories(${ONNXRUNTIME_INCLUDE_DIR})
endif()

# 源文件
set(SOURCES
    cpp/file_processor.cpp
)

# 创建共享库
add_library(FastProcessor SHARED ${SOURCES})

# 链接库
if(ONNX_FOUND)
    target_link_libraries(FastProcessor ${ONNXRUNTIME_LIBRARY})
endif()
target_link_libraries(FastProcessor pthread)

# 设置输出名称
set_target_properties(FastProcessor PROPERTIES
    OUTPUT_NAME "FastProcessor"
    VERSION 1.0.0
    SOVERSION 1
)

# 编译测试程序
add_executable(test_processor cpp/test_processor.cpp)
if(ONNX_FOUND)
    target_link_libraries(test_processor FastProcessor)
else()
    target_link_libraries(test_processor pthread)
endif()

# 复制词典文件到构建目录
if(EXISTS ${CPPJIEBA_DIR}/dict/)
    file(COPY ${CPPJIEBA_DIR}/dict/ DESTINATION ${CMAKE_BINARY_DIR}/dict)
    message(STATUS "词典已复制到: ${CMAKE_BINARY_DIR}/dict")
endif()