function obj = load(start_index, fp)
    nmodes = fscanf(fp, '%u', 1);
    ndims = fscanf(fp, '%u', nmodes);
    obj = spTensor(ndims);
    while true
        [ind, nread] = fscanf(fp, '%u', nmodes);
        if nread ~= nmodes
            break;
        end
        obj.inds(:, obj.nnz+1) = ind + 1 - start_index;
        obj.values(obj.nnz+1, 1) = fscanf(fp, '%f', 1);
        obj.nnz = obj.nnz + 1;
    end
end
