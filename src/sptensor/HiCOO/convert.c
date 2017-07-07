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
#include <assert.h>
#include <ParTI.h>
#include "../sptensor.h"
#include "hicoo.h"

/**
 * Compare two coordinates of a sparse tensor, in the order of mode-0,...,N-1. One from the sparse tensor, the other is specified.
 * @param tsr    a pointer to a sparse tensor
 * @return      1, z > item; otherwise, 0.
 */
int sptCompareCoordinates(
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
 * Check if a nonzero item is in the range of two given coordinates, in the order of mode-0,...,N-1. 
 * @param tsr    a pointer to a sparse tensor
 * @return      1, yes; 0, no.
 */
int sptCoordinatesInRange(
    sptSparseTensor *tsr,
    const sptNnzIndex z,
    const sptIndex * range_begin,
    const sptIndex * range_end)
{
    if (sptCompareCoordinates(tsr, z, range_begin) == 1 &&
        sptCompareCoordinates(tsr, z, range_end) == 0) {
        return 1;
    }
    return 0;
}

/**
 * Compute the beginning of the next block
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the beginning indices of the next block
 */
int sptNextBlockBegin(
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
int sptBlockEnd(
    sptIndex * out_item,
    sptSparseTensor *tsr,
    const sptIndex * in_item,
    const sptElementIndex sb)
{
    sptIndex nmodes = tsr->nmodes;

    for(sptIndex m=0; m<nmodes; ++m) {
        assert(in_item[m] < tsr->ndims[m] - 1);
        out_item[m] = in_item[m]+sb-1 < tsr->ndims[m] ? in_item[m]+sb-1 : tsr->ndims[m] - 1;
    }

    return 0;
}


/**
 * Locate the beginning of the block containing the coordinates
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the beginning indices of this block
 */
int sptLocateBlockBegin(
    sptIndex * out_item,
    sptSparseTensor *tsr,
    const sptIndex * in_item,
    const sptElementIndex sb)
{
    sptIndex nmodes = tsr->nmodes;

    for(sptIndex m=0; m<nmodes; ++m) {
        out_item[m] = in_item[m] - in_item[m] % sb;
    }

    return 0;
}

/**
 * Record mode pointers for block rows, from a sorted tensor.
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
    sptIndex oldi = i;
    sptNnzIndex oldz = 0;
    sptNnzIndex b = 0;
    if(i >= sb * (b+1)) {
        mptr->data[b] = oldz;
        ++ b;
    }

    
    for(sptNnzIndex z=0; z<nnz; ++z) {
        i = tsr->inds[0].data[z];
        if(i >= sb * (b+1)) {
            mptr->data[b] = oldz;
            ++ b;
        }
        oldi = i;
        oldz = z;
    }
    assert(b <= mptr->len);

    return 0;
}

/**
 * Pre-process COO sparse tensor by permuting, sorting, and record mode pointers.
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

    /* Sort tsr in a particular order */
    sptSparseTensorSortIndexBlocked(tsr, 1, sb);

    sptIndex num_mb = (sptIndex)(nnz / sb + 0.5);
    result = sptNewNnzIndexVector(mptr, num_mb, num_mb);
    spt_CheckError(result, "HiSpTns Preprocess", NULL);
    result = sptGetBlockFiberPointers(mptr, tsr, sb);
    spt_CheckError(result, "HiSpTns Preprocess", NULL);


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
    assert(pow(2, sb_bit) == (float)sb);
    assert(pow(2, sk_bit) == (float)sk);
    printf("%u, %u\n", sb_bit, sk_bit);
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

#if 0
    sptIndex * block_begin = (sptIndex *)malloc(nmodes * sizeof(*block_begin));
    sptIndex * block_end = (sptIndex *)malloc(nmodes * sizeof(*block_end));
    sptIndex * block_begin_next = (sptIndex *)malloc(nmodes * sizeof(*block_begin_next));
    sptIndex * min_block_begin = (sptIndex *)malloc(nmodes * sizeof(*min_block_begin));
    for(sptIndex m=0; m<nmodes; ++m) 
        block_begin[m] = 0;

    sptNnzIndex mb_begin, mb_end;
    sptNnzIndex nb = 0;
    sptNnzIndex ne = 0;
    sptIndex eindex = 0;
    /* Loop for all mode blocks */
    for(sptNnzIndex mb=0; mb<mptr.len; ++mb) {
        mb_begin = mptr.data[mb];
        mb_end = mb < mptr.len - 1 ? mptr.data[mb+1] : nnz;

        /* Find all blocks for each mode block */
        while(block_begin_next[0] < mb_end) {
            /* Loop mode block for each block range */
            result = sptBlockEnd(block_end, tsr, block_begin, sb);
            spt_CheckError(result, "HiSpTns Convert", NULL);
            for(sptIndex m=0; m<nmodes; ++m) 
                min_block_begin[m] = ndims[m] - 1;  // Set the largest possible indices
            ne = 0;

            /* Find nonzeros in each block */
            for(sptNnzIndex z = mb_begin; z<=mb_end; ++z) {
                if (sptCoordinatesInRange(tsr, z, block_begin, block_end) == 1)
                {
                    for(sptIndex m=0; m<nmodes; ++m) {
                        eindex = tsr->inds[m].data[z] - block_begin[m];
                        assert(eindex < sb);
                        sptAppendElementIndexVector(&hitsr->einds[m], (sptElementIndex)eindex);
                    }
                    sptAppendValueVector(&hitsr->values, tsr->values.data[z]);
                    ++ ne;
                }
                if (sptCompareCoordinates(tsr, z, block_end) == 1 &&
                    sptCompareCoordinates(tsr, z, min_block_begin) == 0)
                {
                    for(sptIndex m=0; m<nmodes; ++m)
                        min_block_begin[m] = tsr->inds[m].data[z];
                }
            }

            /* Record nonzero blocks in binds */
            if(ne > 0) {
                for(sptIndex m=0; m<nmodes; ++m) {
                    sptAppendBlockIndexVector(&hitsr->binds[m], (sptBlockIndex)block_begin[m]);
                }
                ++ nb;
            }

            /* Calculate the next block, from the min_block_begin.
             * Can jump some zero blocks.
             */
            // result = sptNextBlockBegin(block_begin_next, tsr, block_begin, sb);
            // spt_CheckError(result, "HiSpTns Convert", NULL);
            result = sptLocateBlockBegin(block_begin_next, tsr, min_block_begin, sb);
            spt_CheckError(result, "HiSpTns Convert", NULL);
            assert(ne == mb_end - mb_begin + 1);
        }

    assert(nb == hitsr->binds[0].len);
    }


    free(block_begin);
    free(block_end);
    free(block_begin_next);
    free(min_block_begin);
#endif

	return 0;
}
