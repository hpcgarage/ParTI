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

function Y = timesMatrix(X, U, mode)
    if mode > X.nmodes
        throw(MException('invalid mode', 'Invalid mode'));
    end
    if X.ndims(mode) ~= nrows(U)
        throw(MException('dim mismatch', 'dimension mismatch'));
    end
    if X.sortkey ~= mode
        X = X.sortAtMode(mode);
    end
    ind_buf = X.ndims;
    ind_buf(1, mode)  = ncols(U);
    Y = sspTensor(ind_buf, mode);
    [Y, fiberidx] = Y.setIndices(X);

    for i = 1:Y.nnz
        inz_begin = fiberidx(i);
        inz_end = fiberidx(i+1);
        for j = inz_begin:(inz_end-1)
            r = X.inds(mode, j);
            for k = 1:ncols(U)
                Y.values(i, k) = Y.values(i, k) + X.values(j) * U(r, k);
            end
        end
    end
end
