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
#include "../sptensor.h"
#include "hicoo.h"


/*************************************************
 * PRIVATE FUNCTIONS
 *************************************************/
/**
 * Compare two specified coordinates.
 * @param[in]   item1 first tuple
 * @param[in]   item2 second tuple
 * @return      1, item1 == item2; otherwise, 0.
 */
static int spt_EqualWithTwoCoordinates(
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
 * Compute the end of this block
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the end indices of this block
 */
static int spt_BlockEnd(
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
 * Locate the beginning of the block/kernel containing the coordinates
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the beginning indices of this block
 */
static int spt_LocateBeginCoord(
    sptIndex * out_item,
    sptSparseTensor *tsr,
    const sptIndex * in_item,
    const sptElementIndex bits)
{
    sptIndex nmodes = tsr->nmodes;
    
    for(sptIndex m=0; m<nmodes; ++m) {
        out_item[m] = in_item[m] >> bits;
    }

    return 0;
}


/**
 * Set scheduler for kernels.
 * @param[out] kschr  nmodes kernel schedulers
 * @param[out] nkiters  the number of columns for nmodes schedulers
 * @param[in] kptr  a vector of kernel pointers
 * @param[in] tsr    a pointer to a sparse tensor
 * @param[in] sk_bits    the bits of superblock size (sk)
 * @return      mode pointers
 */
int spt_SetKernelScheduler(
    sptIndexVector **kschr,
    sptIndex *nkiters,
    sptNnzIndexVector * const kptr,
    sptSparseTensor *tsr, 
    const sptElementIndex sk_bits)
{
    sptIndex nmodes = tsr->nmodes;
    sptIndex * ndims = tsr->ndims;
    int result = 0;

    sptIndex * coord = (sptIndex *)malloc(nmodes * sizeof(*coord));
    sptIndex * kernel_coord = (sptIndex *)malloc(nmodes * sizeof(*kernel_coord));

    for(sptNnzIndex k=0; k<kptr->len - 1; ++k) {
        sptNnzIndex z = kptr->data[k];
        for(sptIndex m=0; m<nmodes; ++m) 
            coord[m] = tsr->inds[m].data[z];
        result = spt_LocateBeginCoord(kernel_coord, tsr, coord, sk_bits);
        spt_CheckError(result, "HiSpTns Convert", NULL);

        for(sptIndex m=0; m<nmodes; ++m) {
            result = sptAppendIndexVector(&(kschr[m][kernel_coord[m]]), k);
            spt_CheckError(result, "HiSpTns Convert", NULL);
        }
    }

    free(coord);
    free(kernel_coord);

    sptIndex sk = (sptIndex)pow(2, sk_bits);
    sptIndex tmp;
    for(sptIndex m=0; m<nmodes; ++m) {
        tmp = 0;
        sptIndex kernel_ndim = (ndims[m] + sk - 1) / sk;
        for(sptIndex i=0; i<kernel_ndim; ++i) {
            if(tmp < kschr[m][i].len)
                tmp = kschr[m][i].len;
        }
        nkiters[m] = tmp;
    }

    return 0;
}



/**
 * Pre-process COO sparse tensor by permuting, sorting, and record pointers to blocked rows. Kernels in Row-major order, blocks and elements are in Z-Morton order.
 * @param[out] kptr  a vector of kernel pointers
 * @param[out] kschr  nmodes kernel schedulers
 * @param[out] nkiters  the number of columns for nmodes schedulers
 * @param[in] tsr    a pointer to a sparse tensor
 * @param[in] sk_bits    the bits of superblock size (sk)
 * @param[in] sb_bits    the bits of block size (sb)
 * @param[in] tk    the number of threads
 */
int spt_PreprocessSparseTensor(
    sptNnzIndexVector * kptr,
    sptIndexVector **kschr,
    sptIndex *nkiters,
    sptSparseTensor *tsr, 
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    int const tk)
{
    sptNnzIndex nnz = tsr->nnz;
    int result;

    // TODO: possible permute modes to improve parallelism

    /* Sort tsr in a Row-major Block order to get all kernels. Not use Morton-order for kernels: 1. better support for higher-order tensors by limiting kernel size, because Morton key bit <= 128; */
    sptTimer rowblock_sort_timer;
    sptNewTimer(&rowblock_sort_timer, 0);
    sptStartTimer(rowblock_sort_timer);

    sptSparseTensorSortIndexRowBlock(tsr, 1, 0, nnz, sk_bits, tk);  // Parallelized inside

    sptStopTimer(rowblock_sort_timer);
    sptPrintElapsedTime(rowblock_sort_timer, "rowblock sorting");
    sptFreeTimer(rowblock_sort_timer);
#if PARTI_DEBUG == 3
    printf("Sorted by sptSparseTensorSortIndexRowBlock.\n");
    sptAssert(sptDumpSparseTensor(tsr, 0, stdout) == 0);
#endif

    sptTimer set_kernel_timer;
    sptNewTimer(&set_kernel_timer, 0);
    sptStartTimer(set_kernel_timer);

    result = sptSetKernelPointers(kptr, tsr, sk_bits);
    spt_CheckError(result, "HiSpTns Preprocess", NULL);
    result = spt_SetKernelScheduler(kschr, nkiters, kptr, tsr, sk_bits);
    spt_CheckError(result, "HiSpTns Preprocess", NULL);

    sptStopTimer(set_kernel_timer);
    sptPrintElapsedTime(set_kernel_timer, "Set Kernel Ptrs");
    sptFreeTimer(set_kernel_timer);

    sptTimer morton_sort_timer;
    sptNewTimer(&morton_sort_timer, 0);
    sptStartTimer(morton_sort_timer);

    /* Sort blocks in each kernel in Morton-order */
    sptNnzIndex k_begin, k_end;
    /* Loop for all kernels, 0-kptr.len for OMP code */
    #pragma omp parallel for num_threads(tk) 
    for(sptNnzIndex k=0; k<kptr->len - 1; ++k) {
        k_begin = kptr->data[k];
        k_end = kptr->data[k+1];   // exclusive
        /* Sort blocks in each kernel in Morton-order */
        sptSparseTensorSortIndexMorton(tsr, 1, k_begin, k_end, sb_bits, tk);
#if PARTI_DEBUG == 3
    printf("Kernel %"PARTI_PRI_NNZ_INDEX ": Sorted by sptSparseTensorSortIndexMorton.\n", k);
    sptAssert(sptDumpSparseTensor(tsr, 0, stdout) == 0);
#endif
    }

    sptStopTimer(morton_sort_timer);
    sptPrintElapsedTime(morton_sort_timer, "Morton sorting");
    sptFreeTimer(morton_sort_timer);

    return 0;
}

/*************************************************
 * PUBLIC FUNCTIONS
 *************************************************/
/**
 * Record mode pointers for kernel rows, from a sorted tensor.
 * @param[out] kptr  a vector of kernel pointers
 * @param[in] tsr    a pointer to a sparse tensor
 * @param[in] sk_bits    the bits of superblock size (sk)
 */
int sptSetKernelPointers(
    sptNnzIndexVector *kptr,
    sptSparseTensor *tsr, 
    const sptElementIndex sk_bits)
{
    sptIndex nmodes = tsr->nmodes;
    sptNnzIndex nnz = tsr->nnz;
    sptNnzIndex k = 0;  // count kernels
    sptNnzIndex knnz = 0;   // #Nonzeros per kernel
    int result = 0;
    result = sptAppendNnzIndexVector(kptr, 0);
    spt_CheckError(result, "HiSpTns Convert", NULL);

    sptIndex * coord = (sptIndex *)malloc(nmodes * sizeof(*coord));
    sptIndex * kernel_coord = (sptIndex *)malloc(nmodes * sizeof(*kernel_coord));
    sptIndex * kernel_coord_prior = (sptIndex *)malloc(nmodes * sizeof(*kernel_coord_prior));

    /* Process first nnz to get the first kernel_coord_prior */
    for(sptIndex m=0; m<nmodes; ++m) 
        coord[m] = tsr->inds[m].data[0];    // first nonzero indices
    result = spt_LocateBeginCoord(kernel_coord_prior, tsr, coord, sk_bits);
    spt_CheckError(result, "HiSpTns Convert", NULL);

    for(sptNnzIndex z=0; z<nnz; ++z) {
        for(sptIndex m=0; m<nmodes; ++m) 
            coord[m] = tsr->inds[m].data[z];
        result = spt_LocateBeginCoord(kernel_coord, tsr, coord, sk_bits);
        spt_CheckError(result, "HiSpTns Convert", NULL);

        if(spt_EqualWithTwoCoordinates(kernel_coord, kernel_coord_prior, nmodes) == 1) {
            ++ knnz;
        } else {
            ++ k;
            result = sptAppendNnzIndexVector(kptr, knnz + kptr->data[k-1]);
            spt_CheckError(result, "HiSpTns Convert", NULL);
            for(sptIndex m=0; m<nmodes; ++m) 
                kernel_coord_prior[m] = kernel_coord[m];
            knnz = 1;
        }
    }
    sptAssert(k < kptr->len);
    sptAssert(kptr->data[kptr->len-1] + knnz == nnz);

    /* Set the last element for kptr */
    sptAppendNnzIndexVector(kptr, nnz); 

    free(coord);
    free(kernel_coord);
    free(kernel_coord_prior);

    return 0;
}


/**
 * Convert a COO tensor to a HiCOO tensor.
 * @param[out] hitsr  the sparse tensor in HiCOO format
 * @param[out] max_nnzb  the maximum number of nonzeros per tensor block
 * @param[in] tsr    a pointer to a sparse tensor
 * @param[in] sb_bits    the bits of block size (sb)
 * @param[in] sk_bits    the bits of superblock size (sk)
 * @param[in] tk    the number of threads
 */
int sptSparseTensorToHiCOO(
    sptSparseTensorHiCOO *hitsr,
    sptNnzIndex *max_nnzb,
    sptSparseTensor *tsr, 
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    int const tk)
{
    const sptElementIndex sc_bits = 14; // It is kept for the future use.
    sptAssert(sk_bits >= sb_bits);
    sptAssert(sc_bits >= sb_bits);

    sptIndex i;
    int result;
    sptIndex nmodes = tsr->nmodes;
    sptNnzIndex nnz = tsr->nnz;

    sptElementIndex sb = pow(2, sb_bits);
    sptIndex sc = pow(2, sc_bits);

    /* Set HiCOO parameters. ndims for type conversion, size_t -> sptIndex */
    sptIndex * ndims = malloc(nmodes * sizeof *ndims);
    spt_CheckOSError(!ndims, "HiSpTns Convert");
    for(i = 0; i < nmodes; ++i) {
        ndims[i] = (sptIndex)tsr->ndims[i];
    }

    result = sptNewSparseTensorHiCOO(hitsr, (sptIndex)tsr->nmodes, ndims, (sptNnzIndex)tsr->nnz, sb_bits, sk_bits, sc_bits);
    spt_CheckError(result, "HiSpTns Convert", NULL);

    /* Pre-process tensor to get hitsr->kptr, values are nonzero locations. */
    sptTimer sort_timer;
    sptNewTimer(&sort_timer, 0);
    sptStartTimer(sort_timer);

    spt_PreprocessSparseTensor(&hitsr->kptr, hitsr->kschr, hitsr->nkiters, tsr, sb_bits, sk_bits, tk);

    sptStopTimer(sort_timer);
    sptPrintElapsedTime(sort_timer, "HiCOO sorting (rowblock + morton)");
    sptFreeTimer(sort_timer);
#if PARTI_DEBUG >= 2
    printf("Kernels: Row-major, blocks: Morton-order sorted:\n");
    sptAssert(sptDumpSparseTensor(tsr, 0, stdout) == 0);
    printf("hitsr->kptr:\n");
    sptDumpNnzIndexVector(&hitsr->kptr, stdout);
#endif

    sptTimer gen_timer;
    sptNewTimer(&gen_timer, 0);
    sptStartTimer(gen_timer);

    /* Temporary storage */
    sptIndex * block_begin = (sptIndex *)malloc(nmodes * sizeof(*block_begin));
    sptIndex * block_end = (sptIndex *)malloc(nmodes * sizeof(*block_end));
    sptIndex * block_begin_prior = (sptIndex *)malloc(nmodes * sizeof(*block_begin_prior));
    sptIndex * block_coord = (sptIndex *)malloc(nmodes * sizeof(*block_coord));

    sptNnzIndex k_begin, k_end; // #Nonzeros locations
    sptNnzIndex nk = 0; // #Kernels  
    sptNnzIndex nc = 0; // #Chunks  
    sptNnzIndex nb = 1; // #Blocks  // counting from the first nnz
    sptNnzIndex nb_tmp = 0;
    sptNnzIndex ne = 0; // #Nonzeros per block
    sptIndex eindex = 0;
    sptBlockIndex chunk_size = 0;

    /* different appending methods:
     * elements: append every nonzero entry
     * blocks: append when seeing a new block.
     * chunks: appending when seeting a new chunk. Notice the boundary of kernels and the last chunk of the whole tensor may be larger than the sc.
     * kernels: append when seeing a new kernel. Not appending a vector, just write data into an allocated array.
     */
    /* Process first nnz */
    for(sptIndex m=0; m<nmodes; ++m) 
        block_coord[m] = tsr->inds[m].data[0];    // first nonzero indices
    result = spt_LocateBeginCoord(block_begin_prior, tsr, block_coord, sb_bits);
    spt_CheckError(result, "HiSpTns Convert", NULL);
    for(sptIndex m=0; m<nmodes; ++m)
        sptAppendBlockIndexVector(&hitsr->binds[m], (sptBlockIndex)block_begin_prior[m]);
    sptAppendNnzIndexVector(&hitsr->bptr, 0);


    /* Loop for all kernels, 0 - hitsr->kptr.len - 1 for OMP code */
    for(sptNnzIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        k_begin = hitsr->kptr.data[k];
        k_end = hitsr->kptr.data[k+1]; // exclusive
        nb_tmp = k == 0 ? 0: nb;
        /* Modify kptr pointing to block locations */
        hitsr->kptr.data[k] = nb_tmp;
        ++ nk;

        /* Only append a chunk for the new kernel, the last chunk in the old kernel may be larger than sc */
        sptAppendNnzIndexVector(&hitsr->cptr, nb_tmp);
        ++ nc;
        chunk_size = 0;

        /* Loop nonzeros in each kernel */
        for(sptNnzIndex z = k_begin; z < k_end; ++z) {
            #if PARTI_DEBUG == 5
                printf("z: %"PARTI_PRI_NNZ_INDEX "\n", z);
            #endif

            for(sptIndex m=0; m<nmodes; ++m) 
                block_coord[m] = tsr->inds[m].data[z];    // first nonzero indices
            #if PARTI_DEBUG == 5
                printf("block_coord:\n");
                sptAssert(sptDumpIndexArray(block_coord, nmodes, stdout) == 0);
            #endif

            result = spt_LocateBeginCoord(block_begin, tsr, block_coord, sb_bits);
            spt_CheckError(result, "HiSpTns Convert", NULL);
            #if PARTI_DEBUG == 5
                printf("block_begin_prior:\n");
                sptAssert(sptDumpIndexArray(block_begin_prior, nmodes, stdout) == 0);
                printf("block_begin:\n");
                sptAssert(sptDumpIndexArray(block_begin, nmodes, stdout) == 0);
            #endif

            result = spt_BlockEnd(block_end, tsr, block_begin, sb);  // exclusive
            spt_CheckError(result, "HiSpTns Convert", NULL);

            /* Append einds and values */
            for(sptIndex m=0; m<nmodes; ++m) {
                eindex = tsr->inds[m].data[z] < (block_begin[m] << sb_bits) ? tsr->inds[m].data[z] : tsr->inds[m].data[z] - (block_begin[m] << sb_bits);
                sptAssert(eindex < sb);
                sptAppendElementIndexVector(&hitsr->einds[m], (sptElementIndex)eindex);
            }
            sptAppendValueVector(&hitsr->values, tsr->values.data[z]);


            /* z in the same block with last z */
            if (spt_EqualWithTwoCoordinates(block_begin, block_begin_prior, nmodes) == 1)
            {
                /* ne: #Elements in current block */
                ++ ne;
            } else { /* New block */
                /* ne: #Elements in the last block */
                /* Append block bptr and bidx */
                sptAppendNnzIndexVector(&hitsr->bptr, (sptBlockIndex)z);
                for(sptIndex m=0; m<nmodes; ++m)
                    sptAppendBlockIndexVector(&hitsr->binds[m], (sptBlockIndex)block_begin[m]);
                for(sptIndex m=0; m<nmodes; ++m)
                    block_begin_prior[m] = block_begin[m];

                /* ne: old block's number of nonzeros */
                if(chunk_size + ne >= sc) {    // calculate the prior block
                    /* Append a chunk ending by the old block */
                    sptAppendNnzIndexVector(&hitsr->cptr, nb);
                    ++ nc;
                    chunk_size = 0;
                } else {
                    chunk_size += ne;
                }

                ++ nb;
                ne = 1;              
            } // End new block
            #if PARTI_DEBUG == 5
                printf("nk: %u, nc: %u, nb: %u, ne: %u, chunk_size: %lu\n\n", nk, nc, nb, ne, chunk_size);
            #endif

        }   // End z loop
        
    }   // End k loop
    sptAssert(nb <= nnz);
    sptAssert(nb == hitsr->binds[0].len); 
    // sptAssert(nc <= nb);
    sptAssert(nk == hitsr->kptr.len - 1);

    /* Last element for kptr, cptr, bptr */
    hitsr->kptr.data[hitsr->kptr.len - 1] = hitsr->bptr.len;
    sptAppendNnzIndexVector(&hitsr->cptr, hitsr->bptr.len);
    sptAppendNnzIndexVector(&hitsr->bptr, nnz);


    *max_nnzb = hitsr->bptr.data[1] - hitsr->bptr.data[0];
    sptNnzIndex sum_nnzb = 0;
    for(sptIndex i=0; i < hitsr->bptr.len - 1; ++i) {
        sptNnzIndex nnzb = hitsr->bptr.data[i+1] - hitsr->bptr.data[i];
        sum_nnzb += nnzb;
        if(*max_nnzb < nnzb) {
          *max_nnzb = nnzb;
        }
    }
    sptAssert(sum_nnzb == hitsr->nnz);

    sptStopTimer(gen_timer);
    sptPrintElapsedTime(gen_timer, "Generate HiCOO");
    sptFreeTimer(gen_timer);


    free(block_begin);
    free(block_end);
    free(block_begin_prior);
    free(block_coord);

	return 0;
}
