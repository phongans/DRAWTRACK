cmake_minimum_required(VERSION 3.10)
project(inferencetrt VERSION 1.0.0)

unset(CMAKE_C_FLAGS CACHE)
unset(CMAKE_CXX_FLAGS CACHE)
unset(CMAKE_CXX_FLAGS_RELEASE CACHE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W4 -DGTL_STATIC" CACHE STRING COMPILE_FLAGS FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD /Ox /Ob2 /Oi /Ot /arch:AVX2 /fp:fast /DNDEBUG /std:c++latest" CACHE STRING COMPILE_FLAGS FORCE)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Linker external library
set(OpenCV_DIR "C:/opencv")
find_package(OpenCV REQUIRED)

set(inference_sources
	src/inference_trt.cpp
)

set(inference_headers
	include/inference_trt.h
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include  ${CMAKE_CURRENT_SOURCE_DIR}/libtensorrt/include ${TensorRT_INCLUDE_DIR} ${CUDA_TOOLKIT_INCLUDE})

find_library(TRT_API_LIBRARY NAMES tensorrt_cpp_api PATHS libtensorrt/build/Release)

# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR C:/TensorRT-8.5.1.7)
include(cmake/FindTensorRT.cmake)

# Build the inferenceTRT library
add_library(${PROJECT_NAME} SHARED ${inference_sources} ${CMAKE_CURRENT_SOURCE_DIR}/libtensorrt/src/engine.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include PRIVATE src ${TensorRT_INCLUDE_DIR} ${CUDA_TOOLKIT_INCLUDE})
target_link_libraries(${PROJECT_NAME} PUBLIC ${TRT_API_LIBRARY} ${OpenCV_LIBS} ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES})