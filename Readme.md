Supported sparse tensor operations:

* Scala-tensor mul/div
* Tensor matricization
* Element-wise tensor add/sub/mul/div
* Kronecker product
* Khatri-Rao product
* Tensor-times-dense matrix (TTM)
* Matricized tensor times Khatri-Rao product (MTTKRP)

Build requirements:

- C Compiler (GCC or Clang)

- CMake ( https://cmake.org )


Build tests:

1. Go to `tests`

2. Type `make`

Build docs:

1. Install Doxygen

2. Go to `docs`

3. Type `make`

TODO:

1. Sequential
  * TTV
  * single-node tensor contraction
2. multi-threading
  * Element-wise functions
  * TTM, TTV
  * MTTKRP
3. GPU (high->low priority)
  1. TTM
  2. TTV
  3. Element-wise functions
  4. MTTKRP

Future TODO:

1. Tensor Contraction

Qs:

1. Haven't check "sptSparseTensorKroneckerMul" and "sptSparseTensorKhatriRaoMul" functions. (TODO)
