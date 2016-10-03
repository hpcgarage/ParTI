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

    Y.values = zeros(Y.nnz, Y.stride, 'gpuArray');
    X.values = gpuArray(X.values);
    X_inds_m = gpuArray(X.inds(mode, :));
    U = gpuArray(U);
    fiberidx = gpuArray(fiberidx);

    max_nblocks = 32768;
    max_nthreads = 1024;
    nthreadsX = 32;
    sizeof_scalar = 8;
    sharedMem = nthreadsX * ncols(U) * sizeof_scalar;

    all_nblocks = ceil(Y.nnz / nthreadsX);
    use_naive_kernel = false;
    if ~use_naive_kernel
        kernel = parallel.gpu.CUDAKernel('ttm.ptx', 'ttm.cu', 'spt_TTMKernel');
    else
        kernel = parallel.gpu.CUDAKernel('ttm_naive.ptx', 'ttm_naive.cu', 'spt_TTMNaiveKernel');
    end

    block_offset = 0;
    while block_offset < all_nblocks
        nblocks = all_nblocks - block_offset;
        if nblocks > max_nblocks
            nblocks = max_nblocks;
        end
        if ~use_naive_kernel
            kernel.ThreadBlockSize = [nthreadsX ncols(U) 1];
            kernel.GridSize = [nblocks 1 1];
            kernel.SharedMemorySize = sharedMem;
            Y.values = feval(kernel, ...
                Y.values, Y.stride, Y.nnz, ...
                X.values, X.nnz, X_inds_m, ...
                fiberidx, length(fiberidx), ...
                U, nrows(U), ncols(U), ncols(U), ...
                block_offset ...
            )
        else
            kernel.ThreadBlockSize = [nthreadsX ncols(U) 1];
            kernel.GridSize = [nblocks 1 1];
            Y.values = feval(kernel, ...
                Y.values, Y.stride, Y.nnz, ...
                X.values, X.nnz, X_inds_m, ...
                fiberidx, length(fiberidx), ...
                U, nrows(U), ncols(U), ncols(U), ...
                block_offset ...
            )
        end
        block_offset = block_offset + max_nblocks;
    end
end
