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

classdef sspTensor
    properties (SetAccess=protected)
        nmodes = 0;
        ndims = zeros(1, 0, 'uint64');
        mode = 0;
        stride = 0;
    end
    properties
        nnz = 0;
        inds = zeros(0, 0, 'uint64');
        values = zeros(0, 0);
    end
    methods
        function tsr = sspTensor(ndims, mode)
            tsr.ndims = ndims(:)';
            tsr.nmodes = length(tsr.ndims);
            tsr.mode = mode;
            tsr.stride = ceil(tsr.ndims(mode)/8)*8;
            tsr.values = zeros(0, tsr.stride);
        end
    end
end
