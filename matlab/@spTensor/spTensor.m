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

classdef spTensor
    properties (SetAccess=protected)
        nmodes = 0;
        ndims = zeros(1, 0, 'uint64');
    end
    properties
        sortkey = 0;
        nnz = 0;
        inds = zeros(0, 0, 'uint64');
        values = zeros(0, 1);
    end
    methods
        function tsr = spTensor(ndims)
            tsr.ndims = ndims(:)';
            tsr.nmodes = length(tsr.ndims);
        end
        nwritten = dump(tsr, start_index, fp)
        tsr = sort(tsr)
        tsr = sortAtMode(tsr, mode)
        mtx = toMatrix(tsr)
        Y = timesMatrix(X, U, mode)
    end
    methods (Static)
        tsr = load(start_index, fp)
        tsr = fromSspTensor(src, epsilon)
    end
end
