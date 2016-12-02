cmake_minimum_required(VERSION 3.0)
project(ParTI)

find_package(CUDA)
find_package(OpenBLAS)  # Sets OpenBLAS_INCLUDE_DIRS and OpenBLAS_LIBRARIES

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fopenmp")

file(GLOB_RECURSE PARTI_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.c" "src/*.cu" "src/*.h" "include/*.h")
include_directories("include" ${OpenBLAS_INCLUDE_DIRS})

cuda_add_library(ParTI SHARED ${PARTI_SRC})
cuda_add_library(ParTI_s STATIC ${PARTI_SRC})

set_target_properties(ParTI PROPERTIES C_STANDARD 99)
set_target_properties(ParTI_s PROPERTIES C_STANDARD 99)

target_link_libraries(ParTI ${OpenBLAS_LIBRARIES})
target_link_libraries(ParTI_s ${OpenBLAS_LIBRARIES})

add_subdirectory("examples")
add_subdirectory("tests")
