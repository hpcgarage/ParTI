cmake_minimum_required(VERSION 3.0)
project(ParTI)

option(USE_CUDA "Use NVIDIA CUDA library" ON)
option(USE_MKL "Use Intel MKL library" OFF)
option(USE_OpenBLAS "Use OpenBLAS library" ON)

if(USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DPARTI_ENABLE_CUDA)
endif()

# A bug in FindCUDA module forces us to put "link_libraries" before "cuda_add_library".
# So we cannot use "target_link_libraries" for target-wise library tracking.
if(USE_MKL)
    find_package(MKL REQUIRED)
    add_definitions(-DPARTI_ENABLE_MKL)
    include_directories(${MKL_INCLUDE_DIRS})
    link_libraries(${MKL_LIBRARIES})
endif()
if(USE_OpenBLAS)
    find_package(OpenBLAS REQUIRED)
    add_definitions(-DPARTI_ENABLE_OpenBLAS)
    include_directories(${OpenBLAS_INCLUDE_DIRS})
    link_libraries(${OpenBLAS_LIBRARIES})
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fopenmp")

# A bug in FindCUDA module forces us to put "include_directories" before "cuda_add_library".
# So we cannot use "target_include_directories" for target-wise include tracking.
include_directories("include")
link_libraries("m")

if(USE_CUDA)
    file(GLOB_RECURSE PARTI_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.c" "src/*.cu" "src/*.h" "include/*.h")
    cuda_add_library(ParTI SHARED ${PARTI_SRC})
    cuda_add_library(ParTI_s STATIC ${PARTI_SRC})
else()
    file(GLOB_RECURSE PARTI_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.c" "src/*.h" "include/*.h")
    add_library(ParTI SHARED ${PARTI_SRC})
    add_library(ParTI_s STATIC ${PARTI_SRC})
endif()

set_target_properties(ParTI PROPERTIES C_STANDARD 99)
set_target_properties(ParTI_s PROPERTIES C_STANDARD 99)

add_subdirectory("examples")
add_subdirectory("tests")
