function tsr = fromSspTensor(src, epsilon)
    tsr = spTensor(src.ndims);
    for i = 1:src.nnz
        for j = 1:src.ndims(src.mode)
            data = src.values(i, j);
            if (isnan(data) || isinf(data) ||
                ~(data < epsilon && data > -epsilon))
                tsr.inds(:, tsr.nnz+1) = src.inds(:, i);
                tsr.inds(src.mode, tsr.nnz+1) = j;
                tsr.values(tsr.nnz+1, 1) = data;
                tsr.nnz = tsr.nnz + 1;
            end
        end
    end
    tsr = tsr.sort();
end
