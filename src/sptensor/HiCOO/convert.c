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
        sptEqualWithCoordinates(tsr, z, range_begin) == 1)
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
    int result;

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
    sptNnzIndex z;
    int result;

    sptIndex nmodes = tsr->nmodes;
    sptNnzIndex nnz = tsr->nnz;

    sptElementIndex sb_bit = log2((float)sb);
    sptBlockIndex sk_bit = log2((float)sk);
    sptAssert(pow(2, sb_bit) == (float)sb);
    sptAssert(pow(2, sk_bit) == (float)sk);
    printf("sb_bit: %u, sk_bit: %u\n", sb_bit, sk_bit);
    fflush(stdout);

    /* Set HiCOO parameters, without allocation */
    sptIndex * ndims = malloc(nmodes * sizeof *ndims);
    spt_CheckOSError(!ndims, "HiSpTns Convert");
    for(i = 0; i < nmodes; ++i) {
        ndims[i] = (sptIndex)tsr->ndims[i];
    }

    result = sptNewSparseTensorHiCOO(hitsr, (sptIndex)tsr->nmodes, ndims, (sptNnzIndex)tsr->nnz, sb, sk, sc);
    spt_CheckError(result, "HiSpTns Convert", NULL);

    sptNnzIndexVector mptr;
    sptPreprocessSparseTensor(&mptr, tsr, sb);
    sptDumpNnzIndexVector(&mptr, stdout);
    sptAssert(sptDumpSparseTensor(tsr, 0, stdout) == 0);

    sptIndex * block_begin = (sptIndex *)malloc(nmodes * sizeof(*block_begin));
    sptIndex * block_end = (sptIndex *)malloc(nmodes * sizeof(*block_end));
    sptIndex * block_begin_next = (sptIndex *)malloc(nmodes * sizeof(*block_begin_next));
    sptIndex * block_coord = (sptIndex *)malloc(nmodes * sizeof(*block_coord));

    sptAppendBlockIndexVector(&hitsr->cptr, 0);

    sptNnzIndex mb_begin, mb_end;
    sptBlockNnzIndex chunk_size_limit = sc * sb;  //#Blocks in a chunk
    sptNnzIndex nk = 0; // #Kernels
    sptNnzIndex nc = 0; // #Chunks
    sptNnzIndex nb = 0; // #Blocks
    sptNnzIndex ne = 0; // #Nonzeros in a block
    sptIndex eindex = 0;
    sptBlockNnzIndex chunk_size = 0;

    /* Loop for all mode blocks, mptr.len -> nk for OMP code */
    for(sptNnzIndex mb=0; mb<mptr.len; ++mb) {
        mb_begin = mptr.data[mb];
        mb_end = mb < mptr.len - 1 ? mptr.data[mb+1] : nnz; // exclusive

        /* Find nonzeros in each block */
        for(sptNnzIndex z = mb_begin; z <= mb_end; ) {
            for(sptIndex m=0; m<nmodes; ++m) 
                block_coord[m] = tsr->inds[m].data[z];    // first nonzero indices

            result = sptLocateBlockBegin(block_begin, tsr, block_coord, sb);
            spt_CheckError(result, "HiSpTns Convert", NULL);
            sptAssert(sptDumpIndexArray(block_begin, nmodes, stdout) == 0);

            result = sptBlockEnd(block_end, tsr, block_begin, sb);
            spt_CheckError(result, "HiSpTns Convert", NULL);
            sptAssert(sptDumpIndexArray(block_end, nmodes, stdout) == 0);

            for(sptIndex m=0; m<nmodes; ++m)
                sptAppendBlockIndexVector(&hitsr->binds[m], (sptBlockIndex)block_begin[m]);

            if (sptCoordinatesInRange(tsr, z, block_begin, block_end) == 1)
            {
                for(sptIndex m=0; m<nmodes; ++m) {
                    eindex = tsr->inds[m].data[z] - block_begin[m];
                    sptAssert(eindex < sb);
                    sptAppendElementIndexVector(&hitsr->einds[m], (sptElementIndex)eindex);
                }
                sptAppendValueVector(&hitsr->values, tsr->values.data[z]);
                ++ ne;
                ++ z;
            }
            /* Next block */
            if (sptLargerThanCoordinates(tsr, z, block_end) == 1)
            {
                for(sptIndex m=0; m<nmodes; ++m)
                    block_begin_coord[m] = tsr->inds[m].data[z];
                result = sptLocateBlockBegin(block_begin_next, tsr, block_begin_coord, sb);
                spt_CheckError(result, "HiSpTns Convert", NULL);
                for(sptIndex m=0; m<nmodes; ++m) 
                    block_begin[m] = block_begin_next[m];

                if(ne > 0) {
                    for(sptIndex m=0; m<nmodes; ++m) {
                        sptAppendBlockIndexVector(&hitsr->binds[m], (sptBlockIndex)block_begin[m]);
                    }
                    ++ nb;
                    
                    if(chunk_size + ne > chunk_size_limit) {
                        ++ nc;
                        sptAppendBlockIndexVector(&hitsr->cptr, nb);
                    } else {
                        chunk_size += ne;
                    }
                    ne = 0;
                }
                
            }

        }
        sptAssert(nb <= mb_end - mb_begin + 1);
        sptAssert(nb == hitsr->binds[0].len); 
    }


    free(block_begin);
    free(block_end);
    free(block_begin_next);
    free(block_begin_coord);

	return 0;
}
