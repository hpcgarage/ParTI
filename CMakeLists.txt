cmake_minimum_required(VERSION 3.0)
project(SpTOL)

find_package(CUDA)

file(GLOB_RECURSE SPTOL_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.c" "*src/*.cu" "src/*.h" "include/*.h")
include_directories("include")
cuda_add_library(SpTOL SHARED ${SPTOL_SRC})
cuda_add_library(SpTOL_s STATIC ${SPTOL_SRC})
