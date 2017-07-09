ParTI!
------

A Parallel Tensor Infrastructure (ParTI!), is to support fast essential sparse tensor operations and tensor decompositions on multicore CPU and GPU architectures. These basic tensor operations are critical to the overall performance of tensor analysis algorithms (such as tensor decomposition). ParTI! is formerly known as SpTOL. 


## Supported sparse tensor operations:

* Scala-tensor mul/div (CPU)
* Kronecker product (CPU)
* Khatri-Rao product (CPU)
* Sparse tensor matricization (CPU)
* Element-wise tensor add/sub/mul/div (CPU, OMP, GPU)
* Sparse tensor-times-dense matrix (SpTTM) (CPU, OMP, GPU)
* Sparse matricized tensor times Khatri-Rao product (SpMTTKRP) (CPU, OMP, GPU)


## Build requirements:

- C Compiler (GCC or Clang)

- [CUDA SDK](https://developer.nvidia.com/cuda-downloads)

- [CMake](https://cmake.org) (>v3.0)

- [OpenBLAS](http://www.openblas.net)

- [MAGMA](http://icl.cs.utk.edu/magma/) (not required)


## Build:

1. Create a file by `touch build.config' to define OpenBLAS_DIR and MAGMA_DIR

2. Type `./build.sh`

3. Check `build` for resulting library

4. Check `build/tests` for testing programs

5. Check `build/examples` for example programs

## Build MATLAB interface:

1. `cd matlab`

2. export LD_LIBRARY_PATH=../build:$LD_LIBRARY_PATH

3. Type `make` to build all functions into MEX library.

4. matlab

    1. In matlab environment, type `addpath(pwd)`
   
    2. Play with ParTI MATLAB inferface.
    

## Build docs:

1. Install Doxygen

2. Go to `docs`

3. Type `make`


## Run examples:

1. MTTKRP: 
    * Usage: ./build/examples/mttkrp tsr mode impl_num [cuda_dev_id, R, output]
    * tsr: input sparse tensor
    * mode: specify tensor mode, e.g. (0, or 1, or 2) for third-order tensors
    * impl_num: 11, 12, 15, where 15 should be the best case
    * cuda_dev_id: -2, -1, or 0, 1, -2: sequential code; -1: omp code; 0, or other possible integer: GPU devide id. [Optinal, -2 by default]
    * R: rank number (matrix column size), an integer. [Optinal, 16 by default]
    * output: the file name for output. [Optinal]
    * An example: ./build/examples/mttkrp example.tns 0 15 0 16 result.txt
    
2. TTM: 
    * Usage: ./build/examples/ttm tsr mode impl_num smem_size [cuda_dev_id, R, output]
    * tsr: input sparse tensor
    * mode: specify tensor mode, e.g. (0, or 1, or 2) for third-order tensors
    * impl_num: 11, 12, 13, 14, 15, where either 14 or 15 should be the best case
    * smem_size: shared memory size in bytes (0, or 16000, or 32000, or 48000) 
    * cuda_dev_id: -2, -1, or 0, 1, -2: sequential code; -1: omp code; 0, or other possible integer: GPU devide id. [Optinal, -2 by default]
    * R: rank number (matrix column size), an integer. [Optinal, 16 by default]
    * output: the file name for output. [Optinal]
    * An example: ./build/examples/ttm example.tns 0 15 16000 0 16 result.txt
    

<br/>The algorithms and details are described in the following publications.
## Publication
* **Optimizing Sparse Tensor Times Matrix on multi-core and many-core architectures**. Jiajia Li, Yuchen Ma, Chenggang Yan, Richard Vuduc. The sixth Workshop on Irregular Applications: Architectures and Algorithms (IA^3), co-located with SC’16. 2016. [[pdf]](http://fruitfly1026.github.io/static/files/sc16-ia3.pdf)

* **ParTI!: a Parallel Tensor Infrastructure for Data Analysis**. Jiajia Li, Yuchen Ma, Chenggang Yan, Jimeng Sun, Richard Vuduc. Tensor-Learn Workshop @ NIPS'16. [[pdf]](http://fruitfly1026.github.io/static/files/nips16-tensorlearn.pdf)


## Contributiors

* Jiajia Li (Contact: jiajiali@gatech.edu)
* Yuchen Ma (Contact: m13253@hotmail.com)

## License

ParTI! is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
