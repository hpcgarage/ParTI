cmake_minimum_required(VERSION 3.0)
project(SpTOL)

file(GLOB_RECURSE TEST_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.c" "*.cu")
foreach(TEST_SRC IN LISTS TEST_SRCS)
  get_filename_component(TEST_EXE "${TEST_SRC}" NAME_WE)
  cuda_add_executable("${TEST_EXE}" "${TEST_SRC}")
  target_link_libraries("${TEST_EXE}" SpTOL_s)
endforeach(TEST_SRC)
