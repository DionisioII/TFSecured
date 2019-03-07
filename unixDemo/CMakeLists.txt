cmake_minimum_required(VERSION 2.8)
project(app_project)


set(CMAKE_BUILD_TYPE Debug)

add_executable(demo main.cpp)

set( OPENCV_VERSION  "2.4.13.1" )
set( OPENCV_PATH     /usr/local/Cellar/opencv/${OPENCV_VERSION} )
set( OPENCV_PATH_INCLUDE ${OPENCV_PATH}/include)

set( OPENCV_PATH_LIBS /usr/local/lib )


message(STATUS "OPENCV_VERSION      = ${OPENCV_VERSION}")
message(STATUS "OPENCV_PATH         = ${OPENCV_PATH}")
message(STATUS "OPENCV_PATH_LIBS    = ${OPENCV_PATH_LIBS}")
message(STATUS "OpenCV_LIBS    = ${OpenCV_LIBS}")



include_directories( ${OPENCV_PATH_INCLUDE} )


file(GLOB OPENCV_LIBS ${OPENCV_PATH_LIBS}/libopencv_*.a  ${OPENCV_PATH_LIBS}/libtiff.a)

message(STATUS "Adding libs:  ${OPENCV_LIBS}")



target_link_libraries(demo ${OPENCV_LIBS})

install(TARGETS demo DESTINATION bin)
