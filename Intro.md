Getting started with SpTOL
==========================


To use SpTOL in a C/C++ project, simply add
```c
#include <SpTOL.h>
```
to your source code.

Link your code with
```sh
-fopenmp -lSpTOL -lm
```

I hope this intro document can help you get used to the basics of SpTOL.


Data types
----------

`sptScalar`: the default real value data type. It is defined as `double` type. For some devices without 64-bit float point support, you might want to define `sptScalar` as `float`.

`sptVector`: dense dynamic array of scalars. It is implemented as a one-dimensional array. It uses preallocation to reduce the overhead of the append operation.

`sptSizeVector`: the same thing, but with `size_t`. C does not have templates, that is why they are implemented twice.

`sptMatrix`: dense matrix type. It is implemented as a two-dimensional array. Column count is rounded up to multiples of 8, to optimize for CPU and GPU cache.

`sptSparseMatrix`: sparse matrix type. It uses COO storage format, which stores the coordinate and the value of each non-zero element.

`sptSparseTensor`: sparse tensor type. It works similar to `sptSparseMatrix`, but the modes (number of dimensions) can be arbitary.

`sptSemiSparseTensor`: semi sparse tensor type. Can be considered as "sparse tensor of dense fiber".


Creating objects
----------------

Most data types can fit themselves into stack memory, as local variables. They will handle extra memory allocations on demand.

For example, to construct an `sptVector` and use it.

```c
// Construct it
sptVector my_vector;
sptNewVector(&my_vector, 0, 0);

// Add values to it
sptAppendVector(&my_vector, 42);
sptAppendVector(&my_vector, 31);

// Copy it to another uninitialized vector
sptVector another_vector;
sptCopyVector(&another_vector, &my_vector);

// Access data
printf("%lf %lf\n", another_vector.data[0], another_vector.data[1]);

// Free memory
sptFreeVector(&my_vector);
sptFreeVector(&another_vector);
```

Most functions require initialized data structures. While functions named `New` or `Copy` require uninitialized data structions. They are states in the Doxygen document on a function basis. Failing to supply data with correct initialization state may result in memory leak or program crash.


Validation
----------

For the sake of simplicity, properties are not designed. You can directly modify any field of any struct.

Every function assumes the input is valid, and gurantees the output is valid. This reduces the the need to check the input for most of the time, and improves the performance as a math library. 

But if you modify the data structure directly, you must keep it valid. Some functions expect ordered input, you should sort them with functions like `sptSparseTensorSortIndex` after your modification, or the functions may not work correctly. These functions usually also produces ordered output.


Error reporting
---------------

Most functions return 0 when it succeeded, non-zero when failed.

By invoking `sptGetLastError`, you can extract the last error information.

Operating system `errno` and CUDA error code are also captured and converted.

If you need to make sure a procedure produces no error, call `sptClearLastError` first, since succes procedures does not clear last error status automatically.

Limitation: memory might not be released properly when an error happened. This might be fixed in future releases.
