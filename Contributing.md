Contributing to ParTI!
======================

This is the guide aboue contributing to ParTI!.


Language standard
-----------------

ParTI! mainly follows C99 standard, and must be compatible with GCC 4.9 through 6.2.

CUDA code follows C++03 standard, which is the default of NVCC compiler.


Indentation and format
----------------------

Feel free to use any indent style, but please respect the original style of an existing file.


Naming convention
-----------------

C does not have namespace, thus it is important to keep names from conflicting. All ParTI! functions have names starting with `spt`. Private funcions start with `spt_`.

Names of functions and types follow `PascalCase`. Constants and enumerations follow `UPPER_CASE`. While variables are not restricted to a naming convention.


Error checking
--------------

`spt_CheckError`, `spt_CheckOSError`, `spt_CheckCudaError` are used to check for invalid input or environmental exceptions.

Use `assert` to check for some conditions that should never happen on a production system, such as wrong data produced by other parts of ParTI!. I/O error or invalid data from the outside should not go into this category.


Using `const`
-------------

`const` provides immutability check, optimizes code, and improves documentation clarity. Correct usage of `const` against pointers and arrays are required.


Licensing and copyright
-----------------------

Contribution to ParTI! must license the code under LGPL version 3. Put a copyright notice alongside with your name at the top of each file you modify.
