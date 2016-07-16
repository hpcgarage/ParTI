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

TODO:

1. TTV
2. multi-threading
  * Element-wise functions
  * TTM, TTV
  * MTTKRP
3. GPU

Future TODO:

1. Tensor Contraction
