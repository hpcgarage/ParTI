#!/bin/sh

set -e

echo "This script will do an out-of-tree build of SpTOL into the 'build' directory."

mkdir -p build
cd build

cmake .. "$@"
# Use this if you have GCC >= 6.0 and CUDA <= 8.0
#cmake .. -DCMAKE_C_COMPILER=gcc-5

make

echo "Finished. Check the 'build' directory for results."
