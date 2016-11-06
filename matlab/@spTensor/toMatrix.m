%{
    This file is part of SpTOL.

    SpTOL is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    SpTOL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with SpTOL.
    If not, see <http://www.gnu.org/licenses/>.
%}

function mtx = toMatrix(tsr)
    if tsr.nmodes ~= 2
        throw(MException('dim mismatch', 'dimension mismatch'));
    end
    mtx = zeros(tsr.ndims(1), tsr.ndims(2));
    for i = 1:tsr.nnz
        mtx(tsr.inds(1, i), tsr.inds(2, i)) = tsr.values(i);
    end
end
