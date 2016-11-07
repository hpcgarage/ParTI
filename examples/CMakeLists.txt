cmake_minimum_required(VERSION 3.0)
project(SpTOL)

file(GLOB_RECURSE EXAMPLE_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.c" "*.cu")
foreach(EXAMPLE_SRC IN LISTS EXAMPLE_SRCS)
  get_filename_component(EXAMPLE_EXE "${EXAMPLE_SRC}" NAME_WE)
  cuda_add_executable("${EXAMPLE_EXE}" "${EXAMPLE_SRC}")
  target_link_libraries("${EXAMPLE_EXE}" SpTOL_s)
endforeach(EXAMPLE_SRC)
