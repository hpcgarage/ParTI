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
        print('Usage:   %s output.tns ndims ndiags mode_dim' % argv[0])
        print()
        print('Example: %s output.tns 3 5 1024' % argv[0])
        print()
        print('Each non-zero element will be a gaussian random number (mu=0, sigma=1).')
        print()
        return 1

    output = argv[1]
    ndims = int(argv[2])
    ndiags = int(argv[3])
    mode_dim = int(argv[4])
    stencil_length = (int)(mode_dim ** (1.0/3))
    print('stencil_length: %d' % (stencil_length))
    assert(ndims == 3)
    assert(ndiags == 5)

    dims = []
    for i in range(ndims):
        dims.append (mode_dim)

    nnz = mode_dim * ndiags
    print('%d non-zero elements estimated.' % nnz)

    f = open(output, 'w')
    f.write('%d\n' % ndims)
    f.write('\t'.join(map(str, dims)))
    f.write('\n')

    inds = [[]]
    offset = (int)(ndiags / (2 * (ndims - 1)))
    for m in range(ndims-1):
        inds.append ([])
    z = 0
    for i in range(mode_dim):
        if ndims == 3:
            # Write (i, i-stencil_length, i), (i, i, i-stencil_length), (i, i, i), (i, i, i+stencil_length), (i, i+stencil_length, i)
            if i - stencil_length >= 0:
                inds[0].append (i); inds[1].append (i - stencil_length); inds[2].append (i);
                z = z + 1;
                inds[0].append (i); inds[1].append (i); inds[2].append (i - stencil_length);
                z = z + 1;
            inds[0].append (i); inds[1].append (i); inds[2].append (i);  
            z = z + 1;
            if i + stencil_length < mode_dim:
                inds[0].append (i); inds[1].append (i); inds[2].append (i + stencil_length);  
                z = z + 1;
                inds[0].append (i); inds[1].append (i + stencil_length); inds[2].append (i);
                z = z + 1;

    assert (z <= nnz)
    nnz = z
    print('%d non-zero elements real.' % nnz)

    for z in range(nnz):
        for m in range(ndims):
            f.write('%d\t' % ((int)(inds[m][z] + 1)))
        f.write('% .16f\n' % random.gauss(0, 1))

    print('100%% completed, %s written.' % human_size(f.tell()))
    f.close()
    print('Successfully written into %s.' % output)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
