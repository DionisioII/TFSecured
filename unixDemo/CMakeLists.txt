cmake_minimum_required(VERSION 3.6)
project(demo)

set(CMAKE_CXX_STANDARD 11)

file(GLOB SOURCE_FILES src/*)


if(APPLE)
    set(PLATFORM_TYPE mac)
elseif()
    set(PLATFORM_TYPE linux)
endif()

set(INCLUDE_DIR include)
include_directories(${INCLUDE_DIR})
link_directories(${INCLUDE_DIR})



######## OpenCV ########

message(STATUS "[+] Using OpenCV version 4.0.1 ....")

set(OPENCV_LIB_DIR libs/${PLATFORM_TYPE}/opencv)
file(GLOB OPENCV_LIBS ${OPENCV_LIB_DIR}/*.a)


######## Tensorflow ########

message(STATUS "[+] Using Tensorflow version 1.1.0 ....")

set(TF_LIB_DIR libs/${PLATFORM_TYPE}/tensorflow)

add_library( libtensorflow SHARED IMPORTED)
set_target_properties( libtensorflow PROPERTIES IMPORTED_LOCATION
                       ${TF_LIB_DIR}/libtensorflow.so )


add_executable(demo ${SOURCE_FILES})

target_link_libraries(
        demo
        ${OPENCV_LIBS}
        libtensorflow
)

install(TARGETS demo DESTINATION bin)




