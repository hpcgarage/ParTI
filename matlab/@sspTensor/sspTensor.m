classdef sspTensor
    properties (SetAccess=protected)
        nmodes = 0;
        ndims = zeros(1, 0);
        mode = 0;
        stride = 0;
    end
    properties
        nnz = 0;
        inds = zeros(0, 0);
        values = zeros(0, 0);
    end
    methods
        function tsr = sspTensor(ndims, mode)
            tsr.ndims = ndims(:)';
            tsr.nmodes = length(tsr.ndims);
            tsr.mode = mode;
            tsr.stride = ceil(tsr.ndims(mode)/8)*8;
            tsr.values = zero(0, tsr.stride);
        end
    end
end
