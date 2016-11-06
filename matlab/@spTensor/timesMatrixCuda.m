%{
    This file is part of SpTOL.

    SpTOL is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    SpTOL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with SpTOL.
    If not, see <http://www.gnu.org/licenses/>.
%}

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

    Y.values = zeros(1, Y.nnz*Y.stride, 'gpuArray');
    X.values = gpuArray(flatten(X.values));
    X_inds_m = gpuArray(flatten(X.inds(mode, :) - 1));
    U_stride = ceil(ncols(U) / 8) * 8;
    U_vals = zeros(nrows(U), U_stride);
    U_vals(:, 1:ncols(U)) = U;
    U_vals = gpuArray(flatten(U_vals));
    fiberidx = gpuArray(flatten(fiberidx - 1));

    max_nblocks = 32768;
    max_nthreads = 1024;
    env_SPTOL_TTM_NTHREADS = getenv('SPTOL_TTM_NTHREADS');
    nthreadsX = str2num(env_SPTOL_TTM_NTHREADS);
    if length(nthreadsX) == 0
        nthreadsX = 32;
    end
    sizeof_scalar = 8;
    sharedMem = nthreadsX * Y.stride * sizeof_scalar;

    all_nblocks = ceil(Y.nnz / nthreadsX);
    env_SPTOL_TTM_KERNEL = getenv('SPTOL_TTM_KERNEL');
    if length(env_SPTOL_TTM_KERNEL) ~= 0 && env_SPTOL_TTM_KERNEL == 'naive'
        use_naive_kernel = true;
    else
        use_naive_kernel = false;
    end
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
                U_vals, nrows(U), ncols(U), U_stride, ...
                block_offset ...
            );
        else
            kernel.ThreadBlockSize = [nthreadsX ncols(U) 1];
            kernel.GridSize = [nblocks 1 1];
            Y.values = feval(kernel, ...
                Y.values, Y.stride, Y.nnz, ...
                X.values, X.nnz, X_inds_m, ...
                fiberidx, length(fiberidx), ...
                U_vals, nrows(U), ncols(U), U_stride, ...
                block_offset ...
            );
        end
        block_offset = block_offset + max_nblocks;
    end
    Y.values = reshape(Y.values, [Y.stride Y.nnz])';
end

function mtx = flatten(mtx)
    mtx = reshape(mtx', [1 numel(mtx)]);
end
