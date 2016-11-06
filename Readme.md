SpTOL
-----

Sparse Tensor Operation Library

## Supported sparse tensor operations:

* Scala-tensor mul/div
* Element-wise tensor add/sub/mul/div
* Kronecker product
* Khatri-Rao product
* Sparse tensor-times-dense matrix (TTM)
* Matricized tensor times Khatri-Rao product (MTTKRP)
* Tensor matricization

## Build requirements:

- C Compiler (GCC or Clang)

- [CUDA SDK](https://developer.nvidia.com/cuda-downloads)

- [CMake](https://cmake.org)


## Build:

1. Type `cmake .`

2. Type `make`

You may also create an empty directory, type `cmake <path to SpTOL>` there, followed by `make`, which will prevent polluting source tree.

## Build docs:

1. Install Doxygen

2. Go to `docs`

3. Type `make`

