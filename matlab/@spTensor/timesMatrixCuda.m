function Y = timesMatrix(X, U, mode)
    if mode > X.nmodes
        throw(MException('invalid mode', 'Invalid mode'));
    end
    if X.ndims(mode) ~= size(U)(1)
        throw(MException('dim mismatch', 'dimension mismatch'));
    end
    if X.sortkey ~= mode
        X = X.sortAtMode(mode);
    end
    ind_buf = X.ndims;
    ind_buf(1, mode)  = size(U)(2);
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
    sharedMem = nthreadsX * size(U)(2) * sizeof_scalar;

    if mod(Y.nnz, nthreadsX) == 0
        all_nblocks = Y.nnz / nthreadsX;
    else
        all_nblocks = Y.nnz / nthreadsX + 1;
    end
    use_naive_kernel = 0;
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
            kernel.ThreadBlockSize = [nthreadsX size(U)(2) 1];
            kernel.GridSize = [nblocks 1 1];
            % kernel.sharedMemory = ...
            Y.values = feval(kernel,
                Y.stride, Y.nnz,
                X.values, X.nnz, X_inds_m,
                fiberidx_val, length(fiberidx_val),
                U, size(U)(1), size(U)(2), size(U)(2),
                block_offset
            )
        else
            kernel.ThreadBlockSize = [nthreadsX size(U)(2) 1];
            kernel.GridSize = [nblocks 1 1];
            Y.values = feval(kernel,
                Y.stride, Y.nnz,
                X.values, X.nnz, X_inds_m,
                fiberidx_val, length(fiberidx_val),
                U, size(U)(1), size(U)(2), size(U)(2),
                block_offset
            )
        end
        block_offset = block_offset + max_nblocks;
    end
end
