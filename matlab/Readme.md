# MATLAB toolbox ported from SpTOL

## Usage

When you start, change your current directory here.

MATLAB will be able to read class declarations and definitions in directories
starting with `@` when you use the corresponding data type.

A trivial example:

```matlab
spt = spTensor([1 2 3])

...

fo = fopen('tensor.tsr', 'w')
spt.dump(0, fo)
fclose(fo)

fi = fopen('tensor.tsr', 'r')
spt1 = spt.load(0, fi)
fclose(fi)
```

## Class `spTensor`

A data type representing a sparse tensor

### Property `nmodes`, protected

The number of modes

### Property `ndims`, protected

The dimension of each mode

Size: (1 x `nmodes`)

### Property `sortkey`, public

On which mode is the tensor sorted

### Property `nnz`, public

Number of non-zero elements in this sparse tensor

### Property `inds`, public

The index of each element, on each mode

Size: (`nmodes` x `nnz`)

### Property `values`, public

The numeral data of the elements corresponding to `inds`

Size: (`nnz` x 1)

### Method constructor

Construct an `spTensor` object using the given `ndims`

The length of `ndims` will be `nmodes`.

```matlab
tsr = spTensor(ndims)
```

where `ndims` is a vector

### Method `load`, static

Construct an `spTensor` object using the given file


```matlab
fp = fopen('tensor.tsr', 'r')
start_index = 0
tsr = spTensor.load(start_index, fp)
fclose(fp)
```

Where `start_index` may be 0 or 1:
MATLAB starts a vector with 1 while C/C++/Python starts an array/vector/list
with 0. By setting `start_index` to 0, all indices will be increased by 1.

### Method `dump`

Write the contents of an `spTensor` object to a given file

```matlab
tsr = spTensor(...)
fp = fopen('tensor.tsr', 'w')
start_index = 0
nwritten = tsr.dump(start_index, fp)
fclose(fp)
```

Where `start_index` may be 0 or 1:
MATLAB starts a vector with 1 while C/C++/Python starts an array/vector/list
with 0. By setting `start_index` to 0, all indices will be decreased by 1.

### Method `sort`

Apply the quick sort algorithm on the `spTensor` object, on the last mode

After sorting, the `sortkey` property will be set to `nmodes`

```matlab
tsr = spTensor(...)
tsr.sort()
```

### Method `sortAtMode`


Apply the quick sort algorithm on the `spTensor` object, on the given mode

After sorting, the `sortkey` property will be set to `mode`

```matlab
tsr = spTensor(...)
mode = 1
tsr.sort(mode)
```

Not implemented

### Method `mulMatrix`

Multiply a `spTensor` with a dense matrix, the TTM algorithm

The multiplication will be done on the given `mode`

Return a `sspTensor` as the result

```matlab
tsr = spTensor(...)
mtx = [...]
result = tsr.mulMatrix(mtx, mode)

```

Not implemented

## Class `sspTensor`

A data type representing a semi-sparse tensor

Not implemented

## License

It's not determined, yet.
