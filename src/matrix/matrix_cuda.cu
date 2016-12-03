/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <ParTI.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../error/error.h"
#include <cuda_runtime.h>


__global__ static void spt_MatrixDotMulSeqKernel(
    size_t const mode,
    size_t const nmodes, 
    const size_t rank, 
    const size_t stride, 
    sptScalar ** dev_ata)
{
    const size_t tidx = threadIdx.x;
    const size_t tidy = threadIdx.y;

    sptScalar * ovals = dev_ata[nmodes];
    for(size_t m=1; m < nmodes; ++m) {
        size_t const pm = (mode + m) % nmodes;
        sptScalar const * vals = dev_ata[pm];
        ovals[tidx * stride + tidy] *= vals[tidx * stride + tidy];
        __syncthreads();
    }
}


int sptCudaMatrixDotMulSeq(
    size_t const mode,
    size_t const nmodes, 
    const size_t rank, 
    const size_t stride, 
    sptScalar ** dev_ata)
{
    sptScalar * ovals = dev_ata[nmodes];
    for(size_t i=0; i < rank; ++i) {
        for(size_t j=0; j < rank; ++j) {
            ovals[i * stride + j] = 1;
        }
    }

    dim3 nthreads(rank, rank);  // rank <=  16
    dim3 nblocks(1, 1);

    spt_MatrixDotMulSeqKernel<<<nblocks, nthreads>>> (mode, nmodes, rank, stride, dev_ata);
    
    return 0;
}