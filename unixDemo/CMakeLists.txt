cmake_minimum_required(VERSION 3.6)
project(demo)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)

set(PLATFORM mac)
set(OPENCV_INCLUDE include)
set(OPENCV_LIB_DIR libs/${PLATFORM}/opencv)

file(GLOB OPENCV_LIBS ${OPENCV_LIB_DIR}/*.a)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


message(STATUS "OPENCV_LIB_DIR    = ${OPENCV_LIB_DIR}")
message(STATUS "OPENCV_LIBS       = ${OPENCV_LIBS}")
message(STATUS "OPENCV_INCLUDE   = ${OPENCV_INCLUDE}")

find_library(corefoundation_lib CoreFoundation)
find_library(cfnetwork_lib CFNetwork)

set(frameworks
    ${cfnetwork_lib}
    ${corefoundation_lib})


include_directories(${OPENCV_INCLUDE})
link_directories(${OPENCV_INCLUDE})

add_executable(demo ${SOURCE_FILES})
target_link_libraries(demo ${OPENCV_LIBS} ${frameworks})

install(TARGETS demo DESTINATION bin)




