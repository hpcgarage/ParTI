#!/usr/bin/env python

import math
import random
import sys


if sys.version_info < (3,):
    iter_range, list_range = xrange, range
else:
    iter_range, list_range = range, lambda stop: list(range(stop))


def randround(x):
    int_part = math.floor(x)
    frac_part = x - int_part
    return int(math.ceil(x) if random.random() < frac_part else int_part)


def main(argv):
    if len(argv) < 2:
        print('Usage:   %s output.tns [nonzero_rate%%]mode_dimension ...' % argv[0])
        print()
        print('Example: %s output.tns 256 50%%1024 2%%16384' % argv[0])
        print()
        print('Each non-zero element will be a gaussian random number (mu=0, sigma=1).')
        print()
        return 1

    output = argv[1]
    rates = []
    dims = []
    for i in argv[2:]:
        if '%' in i:
            rate, dim = i.split('%', 1)
            rates.append(float(rate) * 0.01)
            dims.append(int(dim))
        else:
            rates.append(1)
            dims.append(int(i))
    ndims = len(dims)

    nnz = 1
    for i in iter_range(ndims):
        nnz *= rates[i] * dims[i]
    print('%d non-zero elements estimated.' % round(nnz))
    written = 0
    percent = 0

    f = open(output, 'w')
    f.write('%d\n' % ndims)
    f.write('\t'.join(map(str, dims)))

    inds = [None] * ndims
    ptrs = [0] * ndims
    for i in iter_range(ndims):
        if rates[i] == 1:
            inds[i] = list_range(dims[i])
        else:
            inds[i] = random.sample(iter_range(dims[i]), randround(rates[i] * dims[i]))
            inds[i].sort()
    while ptrs[0] != len(inds[0]):
        for i in iter_range(ndims):
            f.write('%d\t' % (inds[i][ptrs[i]]+1))
        f.write('% .16f\n' % random.gauss(0, 1))
        ptrs[ndims-1] += 1
        written += 1
        if nnz != 0:
            new_percent = int(written * 100.0 / nnz)
            if new_percent < 100 and new_percent != percent:
                percent = new_percent
                print('%3d%% completed, %d generated, %.1f MiB written.' % (percent, written, f.tell() / 1048576.0))
        for i in iter_range(ndims-1, 0, -1):
            if ptrs[i] == len(inds[i]):
                if rates[i] == 1:
                    inds[i] = list_range(dims[i])
                else:
                    inds[i] = random.sample(iter_range(dims[i]), randround(rates[i] * dims[i]))
                    inds[i].sort()
                ptrs[i] = 0
                ptrs[i-1] += 1
    
    print('100%% completed, %d generated, %.1f MiB written.' % (written, f.tell() / 1048576.0))
    f.close()
    print('Successfully written into %s.' % output)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
