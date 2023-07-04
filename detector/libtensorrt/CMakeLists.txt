cmake_minimum_required(VERSION 3.18)
project(tensorrt_cpp_api VERSION 1.0.0)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# TODO: Specify the path to TensorRT root dir
if (NOT TensorRT_DIR)
    set(TensorRT_DIR "C:/TensorRT-8.5.1.7")
endif()
# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR C:/Program\ Files/NVIDI\ GPU\ Computing\ Toolkit/CUDA/v11.7)

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
set(OpenCV_DIR "C:/opencv")
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})

file(GLOB SRC_LIST src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SRC_LIST} ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
target_include_directories(${PROJECT_NAME} PUBLIC include PRIVATE src)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES DEFINE_SYMBOL  "COMPILING_DLL" )
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)