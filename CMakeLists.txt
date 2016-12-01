cmake_minimum_required(VERSION 3.0)
project(SpTOL)

find_package(CUDA)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fopenmp")

file(GLOB_RECURSE SPTOL_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.c" "src/*.cu" "src/*.h" "include/*.h")
include_directories("include" "/usr/include")
cuda_add_library(SpTOL SHARED ${SPTOL_SRC})
cuda_add_library(SpTOL_s STATIC ${SPTOL_SRC})

set_target_properties(SpTOL PROPERTIES C_STANDARD 99)
set_target_properties(SpTOL_s PROPERTIES C_STANDARD 99)

target_link_libraries(SpTOL blas lapack)
target_link_libraries(SpTOL_s blas lapack)

add_subdirectory("examples")
add_subdirectory("tests")
