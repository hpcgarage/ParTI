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

#include <math.h>
#include <ParTI.h>
#include "../sptensor.h"
#include "hicoo.h"

/**
 * Compare two coordinates, in the order of mode-0,...,N-1. One from the sparse tensor, the other is specified.
 * @param tsr    a pointer to a sparse tensor
 * @return      1, z > item; otherwise, 0.
 */
static int sptLargerThanCoordinates(
    sptSparseTensor *tsr,
    const sptNnzIndex z,
    const sptIndex * item)
{
    sptIndex nmodes = tsr->nmodes;
    sptIndex i1, i2;

    for(sptIndex m=0; m<nmodes; ++m) {
        i1 = tsr->inds[m].data[z];
        i2 = item[m];
        if(i1 > i2) {
            return 1;
            break;
        }
    }
    return 0;
}


/**
 * Compare two coordinates, in the order of mode-0,...,N-1. One from the sparse tensor, the other is specified.
 * @param tsr    a pointer to a sparse tensor
 * @return      1, z < item; otherwise, 0.
 */
static int sptSmallerThanCoordinates(
    sptSparseTensor *tsr,
    const sptNnzIndex z,
    const sptIndex * item)
{
    sptIndex nmodes = tsr->nmodes;
    sptIndex i1, i2;

    for(sptIndex m=0; m<nmodes; ++m) {
        i1 = tsr->inds[m].data[z];
        i2 = item[m];
        if(i1 < i2) {
            return 1;
            break;
        }
    }
    return 0;
}


/**
 * Compare two coordinates, in the order of mode-0,...,N-1. One from the sparse tensor, the other is specified.
 * @param tsr    a pointer to a sparse tensor
 * @return      1, z = item; otherwise, 0.
 */
static int sptEqualWithCoordinates(
    sptSparseTensor *tsr,
    const sptNnzIndex z,
    const sptIndex * item)
{
    sptIndex nmodes = tsr->nmodes;
    sptIndex i1, i2;

    for(sptIndex m=0; m<nmodes; ++m) {
        i1 = tsr->inds[m].data[z];
        i2 = item[m];
        if(i1 != i2) {
            return 0;
            break;
        }
    }
    return 1;
}


/**
 * Compare two specified coordinates.
 * @param tsr    a pointer to a sparse tensor
 * @return      1, z == item; otherwise, 0.
 */
static int sptEqualWithTwoCoordinates(
    const sptIndex * item1,
    const sptIndex * item2,
    const sptIndex nmodes)
{
    sptIndex i1, i2;
    for(sptIndex m=0; m<nmodes; ++m) {
        i1 = item1[m];
        i2 = item2[m];
        if(i1 != i2) {
            return 0;
            break;
        }
    }
    return 1;
}

/**
 * Check if a nonzero item is in the range of two given coordinates, in the order of mode-0,...,N-1. 
 * @param tsr    a pointer to a sparse tensor
 * @return      1, yes; 0, no.
 */
static int sptCoordinatesInRange(
    sptSparseTensor *tsr,
    const sptNnzIndex z,
    const sptIndex * range_begin,
    const sptIndex * range_end)
{
    if ( (sptLargerThanCoordinates(tsr, z, range_begin) == 1 ||
        sptEqualWithCoordinates(tsr, z, range_begin) == 1) &&
        sptSmallerThanCoordinates(tsr, z, range_end) == 1) {
        return 1;
    }
    return 0;
}

/**
 * Compute the beginning of the next block
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the beginning indices of the next block
 */
static int sptNextBlockBegin(
    sptIndex * out_item,
    sptSparseTensor *tsr,
    const sptIndex * in_item,
    const sptElementIndex sb)
{
    sptIndex nmodes = tsr->nmodes;

    for(int32_t m=nmodes-1; m>=0; --m) {
        if(in_item[m] < tsr->ndims[m]-1) {
            out_item[m] = in_item[m]+sb-1 < tsr->ndims[m] ? in_item[m]+sb-1 : tsr->ndims[m] - 1;
            break;
        }
    }

    return 0;
}


/**
 * Compute the end of this block
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the end indices of this block
 */
static int sptBlockEnd(
    sptIndex * out_item,
    sptSparseTensor *tsr,
    const sptIndex * in_item,
    const sptElementIndex sb)
{
    sptIndex nmodes = tsr->nmodes;

    for(sptIndex m=0; m<nmodes; ++m) {
        sptAssert(in_item[m] < tsr->ndims[m]);
        out_item[m] = in_item[m]+sb < tsr->ndims[m] ? in_item[m]+sb : tsr->ndims[m];    // exclusive
    }

    return 0;
}


/**
 * Locate the beginning of the block containing the coordinates
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the beginning indices of this block
 */
static int sptLocateBlockBegin(
    sptIndex * out_item,
    sptSparseTensor *tsr,
    const sptIndex * in_item,
    const sptElementIndex sb)
{
    sptIndex nmodes = tsr->nmodes;
    
    // TODO: efficiently use bitwise operation
    for(sptIndex m=0; m<nmodes; ++m) {
        out_item[m] = in_item[m] - in_item[m] % sb;
    }

    return 0;
}


/**
 * Compute the strides for kernels, mode order N-1, ..., 0 (row-like major)
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the beginning indices of this block
 */
static int sptKernelStrides(
    sptIndex * strides,
    sptSparseTensor *tsr,
    const sptBlockIndex sk)
{
    sptIndex nmodes = tsr->nmodes;
    sptBlockIndex sk_bit = log2((float)sk);
    sptIndex kernel_size = 0;
    
    // TODO: efficiently use bitwise operation
    strides[nmodes-1] = 1;
    for(sptIndex m=nmodes-2; m>=1; --m) {
        kernel_size = (sptIndex)(tsr->ndims[m+1] + sk - 1) / sk;
        strides[m] = strides[m+1] * kernel_size;
    }
    kernel_size = (sptIndex)(tsr->ndims[1] + sk - 1) / sk;
    strides[0] = strides[1] * kernel_size;

    return 0;
}


/**
 * Compute the kernel location in kptr
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the beginning indices of this block
 */
static sptIndex sptLocateKernel(
    sptSparseTensor *tsr,
    const sptIndex * in_item,
    const sptIndex * strides,
    const sptElementIndex sb,
    const sptBlockIndex sk)
{
    sptIndex nmodes = tsr->nmodes;
    sptBlockIndex sk_bit = log2((float)sk) - log2((float)sb);
    sptNnzIndex kidx = 0;
    
    for(sptIndex m=0; m<nmodes; ++m) {
        kidx += (in_item[m] >> sk_bit) * strides[m];    // index * stride
    }

    return 0;
}


/**
 * Record mode pointers for block rows, from a sorted tensor.
 * @param mptr  a vector of pointers as a dense array
 * @param tsr    a pointer to a sparse tensor
 * @return      mode pointers
 */
int sptGetBlockFiberPointers(
    sptNnzIndexVector *mptr,
    sptSparseTensor *tsr, 
    const sptElementIndex sb)
{
    sptNnzIndex nnz = tsr->nnz;
    sptIndex i = tsr->inds[0].data[0];
    sptNnzIndex b = 0;
    sptNnzIndex bnnz = 0;
    mptr->data[0] = 0;
    while(1) {
        if(i >= sb * b && i < sb * (b+1)) {
            ++ bnnz;
            break;
        } else {
            ++ b;
            mptr->data[b] = bnnz + mptr->data[b-1];
            bnnz = 0;
        }
    }

    
    for(sptNnzIndex z=1; z<nnz; ++z) {
        i = tsr->inds[0].data[z];
        /* Compare with the next block row index */
        while(1) {
            if(i >= sb * b && i < sb * (b+1)) {
                ++ bnnz;
                break;
            } else {
                ++ b;
                mptr->data[b] = bnnz + mptr->data[b-1];
                bnnz = 0;
            }
        }
    }
    sptAssert(b < mptr->len);
    sptAssert(mptr->data[mptr->len-1] + bnnz == nnz);

    return 0;
}

/**
 * Pre-process COO sparse tensor by permuting, sorting, and record pointers to blocked rows.
 * @param tsr    a pointer to a sparse tensor
 * @return      mode pointers
 */
int sptPreprocessSparseTensor(
    sptNnzIndexVector *mptr,
    sptSparseTensor *tsr, 
    const sptElementIndex sb)
{
    sptNnzIndex nnz = tsr->nnz;
    int result;

    // TODO: possible permute modes to improve parallelism

    /* Sort tsr only in mode-0 */
    sptSparseTensorSortIndexSingleMode(tsr, 1, 0);

    sptIndex num_mb = (sptIndex)(tsr->ndims[0] / sb + 0.5);
    result = sptNewNnzIndexVector(mptr, num_mb, num_mb);
    spt_CheckError(result, "HiSpTns Preprocess", NULL);
    /* Morton order conserves the block-sorted order in mode-0. */
    result = sptGetBlockFiberPointers(mptr, tsr, sb);
    spt_CheckError(result, "HiSpTns Preprocess", NULL);

    /* Sort tsr in a Morton order for each blocked-mode0 subtensor */
    sptNnzIndex mb_begin, mb_end;
    /* Loop for all mode blocks, mptr.len -> nk for OMP code */
    for(sptNnzIndex mb=0; mb<mptr->len; ++mb) {
        mb_begin = mptr->data[mb];
        mb_end = mb < mptr->len - 1 ? mptr->data[mb+1] : nnz;   // exclusive
        sptSparseTensorSortIndexMorton(tsr, 1, mb_begin, mb_end, sb);
    }

    return 0;
}


int sptSparseTensorToHiCOO(
    sptSparseTensorHiCOO *hitsr,
    sptSparseTensor *tsr, 
    const sptElementIndex sb,
    const sptBlockIndex sk,
    const sptBlockNnzIndex sc)
{
    sptIndex i;
    int result;
    sptIndex nmodes = tsr->nmodes;
    sptNnzIndex nnz = tsr->nnz;

    sptElementIndex sb_bit = log2((float)sb);
    sptBlockIndex sk_bit = log2((float)sk);
    sptAssert(pow(2, sb_bit) == (float)sb);
    sptAssert(pow(2, sk_bit) == (float)sk);

    /* Set HiCOO parameters, without allocation */
    sptIndex * ndims = malloc(nmodes * sizeof *ndims);
    spt_CheckOSError(!ndims, "HiSpTns Convert");
    for(i = 0; i < nmodes; ++i) {
        ndims[i] = (sptIndex)tsr->ndims[i];
    }

    result = sptNewSparseTensorHiCOO(hitsr, (sptIndex)tsr->nmodes, ndims, (sptNnzIndex)tsr->nnz, sb, sk, sc);
    spt_CheckError(result, "HiSpTns Convert", NULL);

    /* Pre-process tensor to get pointers of mode-0 blocks */
    sptNnzIndexVector mptr;
    sptPreprocessSparseTensor(&mptr, tsr, sb);
    sptDumpNnzIndexVector(&mptr, stdout);

    /* Temporary storage */
    sptIndex * block_begin = (sptIndex *)malloc(nmodes * sizeof(*block_begin));
    sptIndex * block_end = (sptIndex *)malloc(nmodes * sizeof(*block_end));
    sptIndex * block_begin_prior = (sptIndex *)malloc(nmodes * sizeof(*block_begin_prior));
    sptIndex * block_coord = (sptIndex *)malloc(nmodes * sizeof(*block_coord));
    sptIndex * kernel_strides = (sptIndex *)malloc(nmodes * sizeof(*kernel_strides));

    sptNnzIndex mb_begin, mb_end;
    sptBlockNnzIndex chunk_size_limit = sc * sb;  // #Blocks per chunk
    // TODO: chunk_size_limit >= max block nnzs
    sptNnzIndex nk = 1; // #Kernels  // counting from the first nnz
    sptNnzIndex nc = 1; // #Chunks  // counting from the first nnz
    sptNnzIndex nb = 1; // #Blocks  // counting from the first nnz
    sptNnzIndex ne = 0; // #Nonzeros per block
    sptIndex eindex = 0;
    sptIndex kindex = 0, kindex_prior = 0;
    sptBlockNnzIndex chunk_size = 0;
    sptNnzIndex num_all_kernels = hitsr->kptr.len;
    printf("num_all_kernels: %lu\n", num_all_kernels);

    result = sptKernelStrides(kernel_strides, tsr, sk);
    spt_CheckError(result, "HiSpTns Convert", NULL);
    printf("kernel_strides:\n");
    sptAssert(sptDumpIndexArray(kernel_strides, nmodes, stdout) == 0);

    /* different appending methods:
     * elements: append every nonzero entry
     * blocks: append when seeing a new block.
     * chunks: appending when seeting a new chunk. Notice the boundary of kernels.
     * kernels: append when seeing a new kernel. Not appending a vector, just write data into an allocated array.
     */
    /* Process first nnz */
    for(sptIndex m=0; m<nmodes; ++m) 
        block_coord[m] = tsr->inds[m].data[0];    // first nonzero indices
    result = sptLocateBlockBegin(block_begin_prior, tsr, block_coord, sb);
    spt_CheckError(result, "HiSpTns Convert", NULL);
    for(sptIndex m=0; m<nmodes; ++m)
        sptAppendBlockIndexVector(&hitsr->binds[m], (sptBlockIndex)block_begin_prior[m]);
    sptAppendBlockIndexVector(&hitsr->cptr, 0);
    kindex_prior = sptLocateKernel(tsr, block_begin_prior, kernel_strides, sb, sk);
    /* kptr already allocated */
    hitsr->kptr.data[kindex_prior] = 0;


    /* Loop for all mode blocks, mptr.len -> nk for OMP code */
    for(sptNnzIndex mb=0; mb<mptr.len; ++mb) {
        mb_begin = mptr.data[mb];
        mb_end = mb < mptr.len - 1 ? mptr.data[mb+1] : nnz; // exclusive

        /* Loop nonzeros in each block */
        for(sptNnzIndex z = mb_begin; z < mb_end; ++z) {

            for(sptIndex m=0; m<nmodes; ++m) 
                block_coord[m] = tsr->inds[m].data[z];    // first nonzero indices
            printf("block_coord:\n");
            sptAssert(sptDumpIndexArray(block_coord, nmodes, stdout) == 0);

            result = sptLocateBlockBegin(block_begin, tsr, block_coord, sb);
            spt_CheckError(result, "HiSpTns Convert", NULL);
            // printf("block_begin_prior:\n");
            // sptAssert(sptDumpIndexArray(block_begin_prior, nmodes, stdout) == 0);
            printf("block_begin:\n");
            sptAssert(sptDumpIndexArray(block_begin, nmodes, stdout) == 0);

            result = sptBlockEnd(block_end, tsr, block_begin, sb);  // exclusive
            spt_CheckError(result, "HiSpTns Convert", NULL);

            /* Append einds and values */
            for(sptIndex m=0; m<nmodes; ++m) {
                eindex = tsr->inds[m].data[z] - block_begin[m];
                sptAssert(eindex < sb);
                sptAppendElementIndexVector(&hitsr->einds[m], (sptElementIndex)eindex);
            }
            sptAppendValueVector(&hitsr->values, tsr->values.data[z]);


            /* z in the same block with last z */
            if (sptEqualWithTwoCoordinates(block_begin, block_begin_prior, nmodes) == 1)
            {
                ++ ne;
            } else { /* New block */
                /* Append block indices */
                for(sptIndex m=0; m<nmodes; ++m)
                    sptAppendBlockIndexVector(&hitsr->binds[m], (sptBlockIndex)block_begin[m]);
                for(sptIndex m=0; m<nmodes; ++m)
                    block_begin_prior[m] = block_begin[m];

                kindex = sptLocateKernel(tsr, block_begin, kernel_strides, sb, sk);
                sptAssert(kindex >= kindex_prior);
                printf("kindex: %lu\n", kindex);

                /* Add a new kernel */
                if( kindex > kindex_prior) {
                    hitsr->kptr.data[kindex] = nb;
                    /* Not append a chunk, the last chunk may be larger than chunk_size_limit */
                    chunk_size = 0;
                    ++ nk;
                } else {
                    if(chunk_size + ne > chunk_size_limit) {    // calculate the prior block
                        /* Append a chunk */
                        sptAppendBlockIndexVector(&hitsr->cptr, nb);
                        ++ nc;
                        chunk_size = 0;
                    } else {
                        chunk_size += ne;
                    }
                }

                ++ nb;
                ne = 1;              
            }
            
            printf("nk: %u, nc: %u, nb: %u, ne: %u\n\n", nk, nc, nb, ne);

        }

        sptAssert(nb <= nnz);
        sptAssert(nb == hitsr->binds[0].len); 
        sptAssert(nc <= nb);
        sptAssert(nk <= num_all_kernels);
    }


    free(block_begin);
    free(block_end);
    free(block_begin_prior);
    free(block_coord);
    free(kernel_strides);

	return 0;
}
