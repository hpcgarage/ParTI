#ifndef SPTOL_SPTENSOR_H
#define SPTOL_SPTENSOR_H

int spt_SparseTensorCompareIndices(const sptSparseTensor *tsr1, size_t ind1, const sptSparseTensor *tsr2, size_t ind2);
void spt_SparseTensorCollectZeros(sptSparseTensor *tsr);

#endif
