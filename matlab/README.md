This directory contains the MATLAB wrapper of ParTI! library.

## Building

Type `make` to build all functions into MEX library.

## Running

If `libParTI.so` is not in the default library search path of your system, you
will need to set `LD_LIBRARY_PATH` to the directory containing `libParTI.so`
before starting MATLAB.

```bash
export LD_LIBRARY_PATH=../build:$LD_LIBRARY_PATH
matlab
```

## Interface

This wrapper is a one-to-one mapping of the original C interface.

However there are some differences between C interface and MATLAB interface:

- Argument mapping

  - Input arguments are mapped into right-hand-side arguments
  - Output arguments are mapped into left-hand-side arguments
  - Bidirectional arguments are placed both left and right
  - Return value is mapped into left-hand-side argument, you will need to allocate a variable to receive this

- Language convensions

  MATLAB matrices are in FORTRAN style: indices start from 1, memory storage is column major. The MATLAB wrapper for ParTI follows this kind of conversion.

  When you call `sptLoadSparseTensor` or `sptDumpSparseTensor`, you can choose the indexing style for your file by specifying a value in `start_idx`.

- Data structure exposing

  Data structure is not exposed to MATLAB interface. However you can access the individual fields of each C structure by using the getter/setter functions.

- Memory safety

  - If the C function requires an initialized data structure, remember to call `sptNew*` functions to initialize them.
  - If the data structures are not used any longer, remember to call `sptFree*` functions to finalize them.
  - Be aware of double-freeing: calling `sptFree*` on the same object twice will crash MATLAB interpreter.

## Type conversion

Convert MATLAB 1xN matrix to `sptVector`:

```matlab
mxvec = [1 2 3 5 8 13 21];
sptvec = sptNewVector(7, 7);
sptvec.setdata(mxvec);
```

Then convert it back:

```matlab
mxvec1 = sptvec.data;
sptFreeVector(sptvec);
```

Convert MATLAB MxN matrix to `sptMatrix`:

```matlab
mxmtx = [1 5 9; 6 7 2; 8 3 4];
sptmtx = sptNewMatrix(3, 3);
sptmtx.setvalues(mxmtx);
```

Then convert it back:

```matlab
mxmtx1 = sptmtx.values;
sptFreeMatrix(sptmtx);
```

## Todo

- Not all functions are wrapped, we are working on this.
- Error checking and recovering is not fully functional.
