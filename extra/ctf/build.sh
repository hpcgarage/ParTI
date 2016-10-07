#!/bin/bash

mpicxx -std=c++0x -std=c++0x -fopenmp -Wall  -D_POSIX_C_SOURCE=200112L -D__STDC_LIMIT_MACROS -DFTN_UNDERSCORE=1   bench_ttm.cxx -o ../bin/bench_ttm -I../include/ -L../lib -lctf -lblas

