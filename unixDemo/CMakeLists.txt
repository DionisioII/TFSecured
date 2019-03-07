cmake_minimum_required(VERSION 3.6)
project(demo)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)

if(APPLE)
    set(PLATFORM mac)
elseif()
    set(PLATFORM linux)
endif()




######## OpenCV ########

set(OPENCV_INCLUDE include)
set(OPENCV_LIB_DIR libs/${PLATFORM}/opencv)
file(GLOB OPENCV_LIBS ${OPENCV_LIB_DIR}/*.a)

message(STATUS "OPENCV_LIBS      = ${OPENCV_LIBS}")
message(STATUS "OPENCV_INCLUDE   = ${OPENCV_INCLUDE}")

include_directories(${OPENCV_INCLUDE})
link_directories(${OPENCV_INCLUDE})


######## Tensorflow ########

set(TF_INCLUDE include)
set(TF_LIB_DIR libs/${PLATFORM}/tensorflow)
file(GLOB TF_LIBS ${TF_LIB_DIR}/*.so)

message(STATUS "TF_LIBS    = ${TF_LIBS}")
message(STATUS "TF_INCLUDE = ${TF_INCLUDE}")

include_directories(${TF_INCLUDE})
link_directories(${TF_INCLUDE})


if(APPLE)
    message(STATUS "Setup RPATH for macos...")

    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_INSTALL_NAME_DIR ${TF_LIB_DIR})
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_INSTALL_RPATH ${TF_LIB_DIR})

    message(STATUS "CMAKE_INSTALL_RPATH = ${CMAKE_INSTALL_RPATH}")
endif()


add_executable(demo ${SOURCE_FILES})
target_link_libraries(demo ${OPENCV_LIBS} ${TF_LIBS})

install(TARGETS demo DESTINATION bin)




