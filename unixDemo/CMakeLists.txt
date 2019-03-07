cmake_minimum_required(VERSION 3.6)
project(demo)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)

set(OPENCV_VERSION "2.4.13.1")
set(OPENCV_PATH /usr/local/Cellar/opencv/${OPENCV_VERSION})
set(OPENCV_INCLUDE ${OPENCV_PATH}/include)
set(OPENCV_LIBS
    ${OPENCV_PATH}/lib/libopencv_imgproc.2.4.13.dylib
    ${OPENCV_PATH}/lib/libopencv_highgui.2.4.13.dylib
    ${OPENCV_PATH}/lib/libopencv_core.2.4.13.dylib
    ${OPENCV_PATH}/lib/libopencv_features2d.2.4.13.dylib
)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


message(STATUS "OPENCV_PATH    = ${OPENCV_PATH}")
message(STATUS "OPENCV_LIBS    = ${OPENCV_LIBS}")
message(STATUS "OPENCV_INCLUDE = ${OPENCV_INCLUDE}")
message(STATUS "DYLD_FALLBACK_LIBRARY_PATH = ${DYLD_FALLBACK_LIBRARY_PATH}")



include_directories(${OPENCV_INCLUDE})
link_directories(${OPENCV_INCLUDE})

add_executable(demo ${SOURCE_FILES})
target_link_libraries(demo ${OPENCV_LIBS})

install(TARGETS demo DESTINATION bin)




