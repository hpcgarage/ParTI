classdef spTensor
    properties (SetAccess=protected)
        nmodes = 0;
        ndims = zeros(1, 0);
    end
    properties
        sortkey = 0;
        nnz = 0;
        inds = zeros(0, 0);
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
