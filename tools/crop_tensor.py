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
        print('This program removes out-of-bound elements and sums elements with identical coordinates prior to output.')
        print()
        return 1
    tensor_file = open(argv[1], 'rb')
    tensor_file.seek(0, 2)
    tensor_size = tensor_file.tell()
    tensor_file.seek(0)
    ndims = int(tensor_file.readline().decode('iso-8859-1', 'replace'))
    if len(argv) != ndims + 3:
        raise ValueError(
            'Error: you specified {} ranges, while the tensor has {} modes.'.
            format(len(argv) - 3, ndims))
    shape = list(map(int, tensor_file.readline().decode('iso-8859-1', 'replace').split()))
    if len(shape) != ndims:
        raise ValueError('Incomplete definition of tensor shape')
    limits = [None] * ndims
    squash = [False] * ndims
    for i in range(ndims):
        limit = argv[i + 3]
        if limit == '-':
            limits[i] = 0, shape[i] + 1
            squash[i] = True
        elif limit == ':':
            limits[i] = 0, shape[i] + 1
        elif limit.startswith(':'):
            limits[i] = 0, int(limit[1:])
        elif limit.endswith(':'):
            limits[i] = int(limit[:-1]), shape[i] + 1
        elif ':' in limit:
            limits[i] = tuple(map(int, limit.split(':', 1)))
        else:
            start = int(limit)
            limits[i] = start, start + 1
    tensor = sortedcontainers.SortedDict()
    percent = 0
    while True:
        line = tensor_file.readline()
        if not line:
            break
        line_split = line.decode('iso-8859-1', 'replace').split()
        if not line_split:
            continue
        coord = tuple(
            (0 if s else int(c) for c, s in zip(line_split[:ndims], squash)))
        if all((start <= c < stop for c, (start, stop) in zip(coord,limits))):
            value = float(line_split[ndims])
            tensor[coord] = tensor.get(coord, 0) + value
        new_percent = tensor_file.tell() * 80 // tensor_size
        if new_percent != percent:
            print('{:3d}% completed.'.format(new_percent), end='\r')
            percent = new_percent
    tensor_file.close()
    tensor_file = open(argv[2], 'wb')
    tensor_file.write(str(ndims - sum(squash)).encode('iso-8859-1', 'replace') + b'\n')
    new_shape = shape.copy()
    for i in range(ndims - 1, -1):
        if squash[i]:
            del new_shape[i]
    tensor_file.write('{}\n'.format('\t'.join(map(str, new_shape))).encode('iso-8859-1', 'replace'))
    tensor_size = len(tensor)
    for count, (coord, value) in enumerate(tensor.iteritems()):
        for i in range(ndims):
            if not squash[i]:
                tensor_file.write(str(coord[i]).encode('iso-8859-1', 'replace') + b'\t')
        tensor_file.write('{: .16g}\n'.format(value).encode('iso-8859-1', 'replace'))
        new_percent = 80 + count * 20 // tensor_size
        if new_percent != percent:
            print('{:3d}% completed.'.format(new_percent), end='\r')
            percent = new_percent
    print('100% completed.')
    tensor_file.close()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
