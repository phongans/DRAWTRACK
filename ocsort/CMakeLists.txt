cmake_minimum_required(VERSION 3.10)
project(ocsort)

unset(CMAKE_C_FLAGS CACHE)
unset(CMAKE_CXX_FLAGS CACHE)
unset(CMAKE_CXX_FLAGS_RELEASE CACHE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W4 -DGTL_STATIC" CACHE STRING COMPILE_FLAGS FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD /Ox /Ob2 /Oi /Ot /arch:AVX2 /fp:fast /DNDEBUG /std:c++latest" CACHE STRING COMPILE_FLAGS FORCE)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(tracker_sources
	src/Utilities.cpp
	src/OCSort.cpp
	src/lapjv.cpp
	src/KalmanFilter.cpp
	src/KalmanBoxTracker.cpp
	src/Association.cpp
)

set(tracker_headers
	include/Utilities.hpp
	include/OCSort.hpp
	include/lapjv.hpp
	include/KalmanFilter.hpp
	include/KalmanBoxTracker.hpp
	include/Association.hpp
)

# Linker external library
SET(Eigen3_DIR "C:/eigen-3.4.0/build")
find_package(Eigen3 REQUIRED)
set(OpenCV_DIR "C:/opencv")
find_package(OpenCV REQUIRED)

set(LIBS ${OpenCV_LIBS})
set(LIBS ${LIBS} Eigen3::Eigen)

add_library(${PROJECT_NAME} SHARED ${tracker_sources})
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${tracker_headers}")

add_executable(sample sample.cpp)
target_link_libraries(sample PUBLIC Eigen3::Eigen ${PROJECT_NAME} ${OpenCV_LIBS})



