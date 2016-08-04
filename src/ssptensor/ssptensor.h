#ifndef SPTOL_SSPTENSOR_H
#define SPTOL_SSPTENSOR_H

#include <SpTOL.h>

int spt_SemiSparseTensorAppend(sptSemiSparseTensor *tsr, const size_t indices[], sptScalar value);
int spt_SemiSparseTensorCompareIndices(const sptSemiSparseTensor *tsr1, size_t ind1, const sptSemiSparseTensor *tsr2, size_t ind2);
int spt_SemiSparseTensorMergeValues(sptSemiSparseTensor *tsr);

#endif
