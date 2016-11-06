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

function tsr = fromSspTensor(src, epsilon)
    tsr = spTensor(src.ndims);
    for i = 1:src.nnz
        for j = 1:src.ndims(src.mode)
            data = src.values(i, j);
            if isnan(data) || isinf(data) || ~(data < epsilon && data > -epsilon)
                tsr.inds(:, tsr.nnz+1) = src.inds(:, i);
                tsr.inds(src.mode, tsr.nnz+1) = j;
                tsr.values(tsr.nnz+1, 1) = data;
                tsr.nnz = tsr.nnz + 1;
            end
        end
    end
    tsr = tsr.sort();
end
