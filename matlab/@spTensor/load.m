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

function tsr = load(start_index, fp)
    nmodes = fscanf(fp, '%u', 1);
    ndims = fscanf(fp, '%u', nmodes);
    tsr = spTensor(ndims);
    while true
        [ind, nread] = fscanf(fp, '%u', nmodes);
        if nread ~= nmodes
            break;
        end
        tsr.inds(:, tsr.nnz+1) = ind + 1 - start_index;
        tsr.values(tsr.nnz+1, 1) = fscanf(fp, '%f', 1);
        tsr.nnz = tsr.nnz + 1;
    end
    tsr = tsr.sort();
end
