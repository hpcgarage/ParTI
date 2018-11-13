#!/bin/bash

set -e

echo "This script will do an out-of-tree build of ParTI into the 'build' directory by calling CMake."
echo
echo "To make things easier, you can write your personalized CMake options in build.config and it will be passed to CMake."
echo
echo "If you have GCC >= 6.0 and CUDA <= 8.0, write this into 'build.config': -DCUDA_HOST_COMPILER=gcc-5"
echo "If you want to use alternate compilers (e.g. Intel C++ Compilers), write this into 'build.config': -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DUSE_ICC=ON "
echo "================================================================================"
echo

declare -a CMAKE_FLAGS
[ -e build.config ] && CMAKE_FLAGS=("${CMAKE_FLAGS[@]}" $(<build.config))
CMAKE_FLAGS=("${CMAKE_FLAGS[@]}" "$@")

mkdir -p build
cd build

cmake "${CMAKE_FLAGS[@]}" ..

make

echo "Finished. Check the 'build' directory for results."
