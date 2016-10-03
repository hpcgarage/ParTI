function Y = timesMatrix(X, U, mode)
    if mode > X.nmodes
        throw(MException('invalid mode', 'Invalid mode'));
    end
    if X.ndims(mode) ~= nrows(U)
        throw(MException('dim mismatch', 'dimension mismatch'));
    end
    if X.sortkey ~= mode
        X = X.sortAtMode(mode);
    end
    ind_buf = X.ndims;
    ind_buf(1, mode)  = ncols(U);
    Y = sspTensor(ind_buf, mode);
    [Y, fiberidx] = Y.setIndices(X);

    for i = 1:Y.nnz
        inz_begin = fiberidx(i);
        inz_end = fiberidx(i+1);
        for j = inz_begin:(inz_end-1)
            r = X.inds(mode, j);
            for k = 1:ncols(U)
                Y.values(i, k) = Y.values(i, k) + X.values(j) * U(r, k);
            end
        end
    end
end
