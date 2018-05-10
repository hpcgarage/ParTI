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


import sys
import sortedcontainers


def main(argv: [str]) -> int:
    if len(argv) < 3:
        print('Usage: {} input.tns output.tns SLICE ...'.format(argv[0]))
        print()
        print('SLICE may be either any of the following:')
        print('        BEGIN:END    for any begin ≤ INDEX < END')
        print('        BEGIN:       for any begin ≤ INDEX')
        print('        :END         for any INDEX < END')
        print('        INDEX        for an index strictly equal to INDEX')
        print('        :            do not crop this dimension')
        print('        -            remove this dimension')
        print('This program sums elements with identical coordinates prior to output.')
        print()
        return 1
    tensor_file = open(argv[1], 'r')
    tensor_file.seek(0, 2)
    tensor_size = tensor_file.tell()
    tensor_file.seek(0)
    ndims = int(tensor_file.readline())
    if len(argv) != ndims + 3:
        raise ValueError(
            'Error: you specified {} ranges, while the tensor has {} modes.'.
            format(len(argv) - 3, ndims))
    shape = list(map(int, tensor_file.readline().split()))
    if len(shape) != ndims:
        raise ValueError('Incomplete definition of tensor shape')
    limits = [None] * ndims
    squash = [False] * ndims
    for i in range(ndims):
        limit = argv[i + 3]
        if limit == '-':
            limits[i] = slice(None)
            squash[i] = True
        elif limit == ':':
            limits[i] = slice(None)
        elif limit.startswith(':'):
            limits[i] = slice(None, int(limit[1:]))
        elif limit.endswith(':'):
            limits[i] = slice(int(limit[:-1]), None)
        elif ':' in limit:
            start, stop = tuple(map(int, limit.split(':', 1)))
            limits[i] = slice(start, stop)
        else:
            start = int(limit)
            limits[i] = slice(start, start + 1)
    tensor = sortedcontainers.SortedDict()
    percent = 0
    while True:
        line = tensor_file.readline()
        if not line:
            break
        line_split = line.split()
        if not line_split:
            continue
        coord = tuple(
            (0 if s else int(c) for c, s in zip(line_split[:ndims], squash)))
        value = float(line_split[ndims])
        if all(((l.start is None or l.start <= c)
                and (l.stop is None or c < l.stop)
                for c, l in zip(coord, limits))):
            tensor[coord] = tensor.get(coord, 0) + value
        new_percent = round(tensor_file.tell() * 50 / tensor_size, 1)
        if new_percent != percent:
            print('{:5.1f}% completed.'.format(new_percent), end='\r')
            percent = new_percent
    print(' 50.0% completed.', end='\r')
    tensor_file.close()
    tensor_file = open(argv[2], 'w')
    tensor_file.write(str(ndims - sum(squash)) + '\n')
    new_shape = shape.copy()
    for i in range(ndims - 1, -1):
        if squash[i]:
            del new_shape[i]
    tensor_file.write('{}\n'.format('\t'.join(map(str, new_shape))))
    percent = 50
    tensor_size = len(tensor)
    for count, (coord, value) in enumerate(tensor.iteritems()):
        for i in range(ndims):
            if not squash[i]:
                tensor_file.write(str(coord[i]) + '\t')
        tensor_file.write('{: .16g}\n'.format(value))
        new_percent = round(50 + count * 50 / tensor_size, 1)
        if new_percent != percent:
            print('{:5.1f}% completed.'.format(new_percent), end='\r')
            percent = new_percent
    print('100.0% completed.')
    tensor_file.close()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
