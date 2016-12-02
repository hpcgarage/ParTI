#!/usr/bin/env python3

# This file is part of ParTI!.
#
# ParTI! is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# ParTI! is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with ParTI!.
# If not, see <http://www.gnu.org/licenses/>.

import math
import random
import sys


if sys.version_info < (3,):
    range = xrange


def randround(x):
    int_part = math.floor(x)
    frac_part = x - int_part
    return int(math.ceil(x) if random.random() < frac_part else int_part)


def human_size(nbytes):
    if nbytes < 1024000:
        if nbytes < 1000:
            return '%d bytes' % nbytes
        else:
            return '%.1f KiB' % (nbytes / 1024.0)
    else:
        if nbytes < 1048576000:
            return '%.1f MiB' % (nbytes / 1048576.0)
        else:
            return '%.1f GiB' % (nbytes / 1073741824.0)

def main(argv):
    if len(argv) < 3:
        print('Usage:   %s output.tns [nonzero_rate%%]mode_dimension ...' % argv[0])
        print()
        print('Example: %s output.tns 256 50%%1024 2%%4096' % argv[0])
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
    for i in range(ndims):
        nnz *= rates[i] * dims[i]
    print('%d non-zero elements estimated.' % round(nnz))
    written = 0
    percent = 0

    f = open(output, 'w')
    f.write('%d\n' % ndims)
    f.write('\t'.join(map(str, dims)))
    f.write('\n')

    inds = [None] * ndims
    ptrs = [0] * ndims
    for i in range(ndims):
        if rates[i] == 1:
            inds[i] = range(dims[i])
        else:
            inds[i] = random.sample(range(dims[i]), randround(rates[i] * dims[i]))
            inds[i].sort()

    while ptrs[0] != len(inds[0]):
        for i in range(ndims):
            f.write('%d\t' % (inds[i][ptrs[i]]+1))
        f.write('% .16f\n' % random.gauss(0, 1))
        ptrs[ndims-1] += 1

        written += 1
        if nnz != 0:
            new_percent = int(written * 100.0 / nnz)
            if new_percent < 100 and new_percent != percent:
                percent = new_percent
                print('%3d%% completed, %d generated, %s written.' % (percent, written, human_size(f.tell())), end='\r', flush=True)

        for i in range(ndims-1, 0, -1):
            if ptrs[i] == len(inds[i]):
                if rates[i] == 1:
                    inds[i] = range(dims[i])
                else:
                    inds[i] = random.sample(range(dims[i]), randround(rates[i] * dims[i]))
                    inds[i].sort()
                ptrs[i] = 0
                ptrs[i-1] += 1

    print('100%% completed, %d generated, %s written.' % (written, human_size(f.tell())))
    f.close()
    print('Successfully written into %s.' % output)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
