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
        print('Usage: {} input.tns output.tns'.format(argv[0]))
        print()
        return 1
    tensor_file = open(argv[1], 'rb')
    tensor_file.seek(0, 2)
    tensor_size = tensor_file.tell()
    tensor_file.seek(0)
    ndims = int(tensor_file.readline().decode('iso-8859-1', 'replace'))
    shape = list(
        map(int,
            tensor_file.readline().decode('iso-8859-1', 'replace').split()))
    if len(shape) != ndims:
        raise ValueError('Incomplete definition of tensor shape')
    indices = [sortedcontainers.SortedSet() for i in range(ndims)]
    tensor = sortedcontainers.SortedDict()
    percent = 0
    while True:
        line = tensor_file.readline()
        if not line:
            break
        line_split = line.decode('iso-8859-1', 'replace').split()
        if not line_split:
            continue
        coord = tuple(map(int, line_split[:ndims]))
        value = float(line_split[ndims])
        for i in range(ndims):
            indices[i].add(coord[i])
        tensor[coord] = tensor.get(coord, 0) + value
        new_percent = tensor_file.tell() * 60 // tensor_size
        if new_percent != percent:
            print('{:3d}% completed.'.format(new_percent), end='\r')
            percent = new_percent
    tensor_file.close()
    idx_lut = [sortedcontainers.SortedDict(((k, i + 1) for i, k in enumerate(m))) for m in indices]
    new_shape = [len(i) for i in indices]
    tensor_file = open(argv[2], 'wb')
    tensor_file.write(
        str(sum((i > 1
                 for i in new_shape))).encode('iso-8859-1', 'replace') + b'\n')
    tensor_file.write('{}\n'.format('\t'.join(
        (str(i) for i in new_shape if i > 1))).encode('iso-8859-1', 'replace'))
    tensor_size = len(tensor)
    for count, (coord, value) in enumerate(tensor.iteritems()):
        if value == 0:
            continue
        new_coord = [idx_lut[i][c] for i, c in enumerate(coord)]
        tensor_file.write('\t'.join(
            (str(c) for i, c in zip(new_shape, new_coord) if i > 1)).encode(
                'iso-8859-1', 'replace'))
        tensor_file.write('\t{: .16g}\n'.format(value).encode(
            'iso-8859-1', 'replace'))
        new_percent = 60 + count * 40 // tensor_size
        if new_percent != percent:
            print('{:3d}% completed.'.format(new_percent), end='\r')
            percent = new_percent
    print('100% completed.')
    tensor_file.close()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
