cmake_minimum_required(VERSION 3.0)
project(SpTOL)

file(GLOB_RECURSE SPTOL_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.c" "src/*.h" "include/*.h")
add_library(SpTOL SHARED ${SPTOL_SRC})
add_library(SpTOL_s STATIC ${SPTOL_SRC})

target_include_directories(SpTOL PUBLIC "include")
target_include_directories(SpTOL_s PUBLIC "include")
