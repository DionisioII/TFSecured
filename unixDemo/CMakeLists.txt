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

message(STATUS "[+] Using OpenCV version 4.0.1 ....")

set(OPENCV_INCLUDE include)
set(OPENCV_LIB_DIR libs/${PLATFORM}/opencv)
file(GLOB OPENCV_LIBS ${OPENCV_LIB_DIR}/*.a)

include_directories(${OPENCV_INCLUDE})
link_directories(${OPENCV_INCLUDE})


######## Tensorflow ########

message(STATUS "[+] Using Tensorflow version 1.12.0 ....")

set(TF_INCLUDE include)
set(TF_LIB_DIR libs/${PLATFORM}/tensorflow)

add_library( libtensorflow SHARED IMPORTED)
set_target_properties( libtensorflow PROPERTIES IMPORTED_LOCATION
                       ${TF_LIB_DIR}/libtensorflow.so )

add_library( libtensorflow_framework SHARED IMPORTED)
set_target_properties( libtensorflow_framework PROPERTIES IMPORTED_LOCATION
                       ${TF_LIB_DIR}/libtensorflow_framework.so )

include_directories(${TF_INCLUDE})
link_directories(${TF_INCLUDE})





add_executable(demo ${SOURCE_FILES})

target_link_libraries(
        demo
        ${OPENCV_LIBS}
        libtensorflow libtensorflow_framework
)

install(TARGETS demo DESTINATION bin)




