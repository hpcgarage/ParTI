function Y = timesMatrix(X, U, mode)
    if mode > X.nmodes
        throw(MException('invalid mode', 'Invalid mode'));
    end
    if  X.ndims(mode) ~= U.nrows
        throw(MException('dim mismatch', 'dimension mismatch'));
    end
    if X.sortkey ~= mode
        X = X.sortAtMode(mode);
    end
    ind_buf = X.ndims;
    ind_buf(1, mode)  = U.ncols;
    Y = sspTensor(ind_buf, mode);
    fiberidx = Y.setIndices(X);

    for i = 1:Y.nnz
        inz_begin = fiberidx(i);
        inz_end = fiberidx(i+1);
        for j = inz_begin:(inz_end-1)
            r = X.inds(mode, j);
            Y.values(i, k) = Y.values(i, j) + X.values(j) * U(r, k);
        end
    end
end
