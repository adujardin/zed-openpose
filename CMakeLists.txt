SET(execName zed_openpose)
cmake_minimum_required(VERSION 2.4)
PROJECT(${execName})

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")

IF(WIN32) # Windows
    SET(ZED_INCLUDE_DIRS $ENV{ZED_INCLUDE_DIRS})
    SET(ZED_LIBRARIES $ENV{ZED_LIBRARIES_64})
    SET(ZED_LIBRARY_DIR $ENV{ZED_LIBRARY_DIR})
    SET(GLUT_INCLUDE_DIRS $ENV{GLUT_INCLUDE_DIRS})
    SET(GLUT_LIBRARY_DIRS $ENV{GLUT_LIBRARY_DIRS})
    SET(GLUT_LIBRARY $ENV{GLUT_LIBRARY})
    SET(GLEW_INCLUDE_DIRS $ENV{GLEW_INCLUDE_DIRS})
    SET(GLEW_LIBRARY_DIRS $ENV{GLEW_LIBRARY_DIRS})
    SET(GLEW_LIBRARY $ENV{GLEW_LIBRARY})
ELSE() # Linux
    find_package(ZED 2 REQUIRED)
    find_package(GLUT REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(OpenPose REQUIRED)
    
    exec_program(uname ARGS -p OUTPUT_VARIABLE CMAKE_SYSTEM_NAME2)
    if ( CMAKE_SYSTEM_NAME2 MATCHES "aarch64" )
        link_directories("/usr/lib/aarch64-linux-gnu/tegra") ## on Jetson TX1 64bits, the correct libGL.so is located here (the default one will lead to linking error (undef ref))
        unset(CUDA_USE_STATIC_CUDA_RUNTIME CACHE)
        option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
        message ("CUDA_USE_STATIC_CUDA_RUNTIME : ${CUDA_USE_STATIC_CUDA_RUNTIME}")
    endif()

    SET(SPECIAL_OS_LIBS "pthread" "X11")
    add_definitions(-std=c++0x -g -O3 -pthread)
ENDIF(WIN32)

find_package(CUDA REQUIRED)
find_package(OpenGL)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

FILE(GLOB_RECURSE SRC_FILES src/*.cpp)
FILE(GLOB_RECURSE HDR_FILES include/*.hpp)

ADD_EXECUTABLE(${execName} ${HDR_FILES} ${SRC_FILES})

target_link_libraries(${execName}
                        ${OpenPose_LIBS} ${GLOG_LIBRARY} ${GFLAGS_LIBRARY} ${Caffe_LIBS}
                        ${ZED_LIBRARIES}
                        ${GLUT_LIBRARY}
                        ${GLEW_LIBRARY}
                        ${SPECIAL_OS_LIBS}
                        ${OPENGL_LIBRARIES}
                        ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_NPP_LIBRARIES_ZED} )
