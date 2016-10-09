#!/bin/bash

mpiicpc -std=c++0x -qopenmp -O3 -Wall  -D_POSIX_C_SOURCE=200112L -D__STDC_LIMIT_MACROS -DUSE_SP_MKL=1 -DFTN_UNDERSCORE=1   bench_ttm.cxx -o bench_ttm -I../include/ -L/home/jli/Software/ctf-1.4.1/lib -lctf -mkl

