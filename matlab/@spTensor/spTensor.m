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
        function obj = spTensor(ndims)
            obj.ndims = ndims(:)';
            obj.nmodes = length(obj.ndims);
        end
        nwritten = dump(obj, start_index, fp)
    end
    methods (Static)
        obj = load(start_index, fp)
    end
end
