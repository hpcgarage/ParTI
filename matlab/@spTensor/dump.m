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

function nwritten = dump(tsr, start_index, fp)
    fprintf(fp, '%u\n', tsr.nmodes);
    fprintf(fp, '%u\t', tsr.ndims);
    fprintf(fp, '\n');
    for i = 1:tsr.nnz
        fprintf(fp, '%u\t', tsr.inds(:, i) + start_index - 1);
        fprintf(fp, '%f\n', tsr.values(i));
    end
    nwritten = tsr.nnz;
end