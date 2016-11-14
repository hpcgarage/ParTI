#!/usr/bin/env python2

# This file is part of SpTOL.
#
# SpTOL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# SpTOL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with SpTOL.
# If not, see <http://www.gnu.org/licenses/>.

import numpy
import sktensor
import itertools

range = xrange
map = itertools.imap

def load_sptensor(fp, start_index=1, dtype=None):
    nmodes = int(fp.readline())
    ndims = tuple(map(int, fp.readline().split()))
    assert nmodes == len(ndims)
    subs = tuple(([] for m in ndims))
    vals = []
    while True:
        line = fp.readline()
        if not line:
            break
        linesep = line.split()
        for m, x in enumerate(linesep[:-1]):
            subs[m].append(int(x) - start_index)
        vals.append((dtype or float)(linesep[-1]))
    return sktensor.sptensor(subs, vals, shape=ndims, dtype=dtype)

def dump_sptensor(tsr, fp, start_idx=1):
    fp.write('%s\n' % tsr.ndim)
    fp.write('%s\n' % ' '.join(map(str, tsr.shape)))
    for idx, val in enumerate(tsr.vals):
        fp.write('\t'.join((str(tsr.subs[m][idx]+start_idx) for m in range(tsr.ndim))))
        fp.write('\t%g\n' % val)

