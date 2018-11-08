#!/bin/bash

set -e

echo "This script will call 'ctest' to perform automatic unit tests and functional tests."
echo "================================================================================"
echo

export CTEST_OUTPUT_ON_FAILURE=1

if cd build
then
    ctest -M Experimental -T MemCheck "$@"
    ctest -M Experimental -T Coverage "$@"
else
    echo "Please run './build.sh' first."
fi
