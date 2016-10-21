#ifndef SPTOL_SPTENSOR_H
#define SPTOL_SPTENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <SpTOL.h>
#include "../error/error.h"

int spt_SparseTensorCompareIndices(const sptSparseTensor *tsr1, size_t ind1, const sptSparseTensor *tsr2, size_t ind2);

void spt_SparseTensorCollectZeros(sptSparseTensor *tsr);

void sptDistSparseTensor(sptSparseTensor * tsr,
    int const nthreads,
    size_t * const dist_nnzs,
    size_t * dist_nrows);

void sptDistSparseTensorFixed(sptSparseTensor * tsr,
    int const nthreads,
    size_t * const dist_nnzs,
    size_t * dist_nrows);
    
#ifdef __cplusplus
}
#endif

#endif
