function mtx = toMatrix(tsr)
    if tsr.nmodes ~= 2
        throw(MException('dim mismatch', 'dimension mismatch'));
    end
    mtx = zeros(tsr.ndims(1), tsr.ndims(2));
    for i = 1:tsr.nnz
        mtx(tsr.inds(1, i), tsr.inds(2, i)) = tsr.values(i);
    end
end
tsr
