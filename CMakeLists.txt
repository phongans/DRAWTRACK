cmake_minimum_required(VERSION 3.10)

project(drawtrack VERSION 1.0.0)

unset(CMAKE_C_FLAGS CACHE)
unset(CMAKE_CXX_FLAGS CACHE)
unset(CMAKE_CXX_FLAGS_RELEASE CACHE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W4 -DGTL_STATIC" CACHE STRING COMPILE_FLAGS FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD /Ox /Ob2 /Oi /Ot /arch:AVX2 /fp:fast /DNDEBUG /std:c++latest" CACHE STRING COMPILE_FLAGS FORCE)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

SET(Eigen3_DIR "C:/eigen-3.4.0/build")
find_package(Eigen3 REQUIRED)
set(OpenCV_DIR "C:/opencv")
find_package(OpenCV REQUIRED)
# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR C:/TensorRT-8.5.1.7)

set(main_headers
	detector/src/inference_trt.h
    detector/libtensorrt/src/engine.h
)

set(main_sources
	detector/src/inference_trt.cpp
    detector/libtensorrt/src/engine.cpp
)

find_library(DETECTOR_LIBRARY NAMES inferencetrt PATHS C:/Projects/ANS-DRAWTRACK/detector/build/Release)
find_library(OCSORT_LIBRARY NAMES ocsort PATHS C:/Projects/ANS-DRAWTRACK/ocsort/build/Release)
find_library(TRT_API_LIBRARY NAMES tensorrt_cpp_api PATHS C:/Projects/ANS-DRAWTRACK/detector/libtensorrt/build/Release)

set(LIBS
    ${OpenCV_LIBS} 
    Eigen3::Eigen
    ${OCSORT_LIBRARY}
    ${DETECTOR_LIBRARY}
)

add_executable(${PROJECT_NAME} drawtrack.cpp ${main_sources} ${main_headers})
target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC ${main_headers})