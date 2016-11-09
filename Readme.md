SpTOL
-----

Sparse Tensor Operation Library (SpTOL) is to support fast essential sparse tensor operations on multicore CPU and GPU architectures. These basic tensor operations are critical to the overall performance of tensor analysis algorithms (such as tensor decomposition).


## Supported sparse tensor operations:

* Scala-tensor mul/div
* Element-wise tensor add/sub/mul/div
* Kronecker product
* Khatri-Rao product
* Sparse tensor-times-dense matrix (SpTTM)
* Sparse matricized tensor times Khatri-Rao product (SpMTTKRP)
* Sparse tensor matricization

## Build requirements:

- C Compiler (GCC or Clang)

- [CUDA SDK](https://developer.nvidia.com/cuda-downloads)

- [CMake](https://cmake.org)


## Build:

1. Type `./build.sh`

2. Check `build` for resulting library

3. Check `build/examples` for example programs

## Build docs:

1. Install Doxygen

2. Go to `docs`

3. Type `make`



<br/>The algorithms and details are described in the following publications.
## Publication
* **Optimizing Sparse Tensor Times Matrix on multi-core and many-core architectures**. Jiajia Li, Yuchen Ma, Chenggang Yan, Richard Vuduc. The sixth Workshop on Irregular Applications: Architectures and Algorithms (IA^3), co-located with SC’16. 2016. [[pdf]](http://fruitfly1026.github.io/static/files/sc16-ia3.pdf)


## Contributiors

* Yuchen Ma (Contact: m13253@hotmail.com)
* Jiajia Li (Contact: jiajiali@gatech.edu)
