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
