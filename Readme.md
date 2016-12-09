ParTI!
------

A Parallel Tensor Infrastructure (ParTI!), is to support fast essential sparse tensor operations and tensor decompositions on multicore CPU and GPU architectures. These basic tensor operations are critical to the overall performance of tensor analysis algorithms (such as tensor decomposition). ParTI! is formerly known as SpTOL. 


## Supported sparse tensor operations:

* Scala-tensor mul/div
* Element-wise tensor add/sub/mul/div
* Kronecker product
* Khatri-Rao product
* Sparse tensor-times-dense matrix (SpTTM)
* Sparse matricized tensor times Khatri-Rao product (SpMTTKRP)
* Sparse tensor matricization
* Sparse CANDECOMP/PARAFAC decomposition

## Build requirements:

- C Compiler (GCC or Clang)

- [CUDA SDK](https://developer.nvidia.com/cuda-downloads)

- [CMake](https://cmake.org) (>v3.0)

- [OpenBLAS](http://www.openblas.net)

- [MAGMA](http://icl.cs.utk.edu/magma/)


## Build:

1. Create a file by `touch build.config' to define OpenBLAS_DIR and MAGMA_DIR

2. Type `./build.sh`

3. Check `build` for resulting library

4. Check `build/examples` for example programs

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



<br/>The algorithms and details are described in the following publications.
## Publication
* **Optimizing Sparse Tensor Times Matrix on multi-core and many-core architectures**. Jiajia Li, Yuchen Ma, Chenggang Yan, Richard Vuduc. The sixth Workshop on Irregular Applications: Architectures and Algorithms (IA^3), co-located with SC’16. 2016. [[pdf]](http://fruitfly1026.github.io/static/files/sc16-ia3.pdf)

* **ParTI!: a Parallel Tensor Infrastructure for Data Analysis**. Jiajia Li, Yuchen Ma, Chenggang Yan, Jimeng Sun, Richard Vuduc. Tensor-Learn Workshop @ NIPS'16. [[pdf]](http://fruitfly1026.github.io/static/files/nips16-tensorlearn.pdf)


## Contributiors

* Yuchen Ma (Contact: m13253@hotmail.com)
* Jiajia Li (Contact: jiajiali@gatech.edu)
