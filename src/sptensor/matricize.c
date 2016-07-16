#include <SpTOL.h>
#include "sptensor.h"

void sptMatricize(sptSparseTensor const * const X, 
	size_t const m,
	sptSparseMatrix * const A,
	int const transpose) {

	size_t const nmodes = X->nmodes;
	size_t const nnz = X->nnz;
	size_t const * const ndims = X->ndims;
	sptScalar const * const vals = X->values.data;

	/* Initialize sparse matrix A. */
	A->nnz = nnz;
	sptNewSizeVector(&(A->rowind), 0, nnz);
	sptNewSizeVector(&(A->colind), 0, nnz);
	sptNewVector(&(A->values), 0, nnz);

	/* Calculate the new_order of the rest modes except mode-m. */
	size_t * new_order = (size_t *)malloc((nmodes-1) * sizeof * new_order);
	for(size_t i=1; i<nmodes; ++i) {
		new_order[i-1] = (m+i) % nmodes;
	}
	/* Calculate strides for tensor X, from mode-(N-1) to mode-0. */
	size_t * strides = (size_t *)malloc((nmodes-1) * sizeof *strides);
	strides[nmodes-2] = 1;
	for(size_t i=nmodes-3; i>=0; --i) {
		size_t new_i = new_order[i];
		strides[i] = strides[i+1] * ndims[new_i+1];
	}


	if(transpose == 1) {	// mode-m as row
		A->nrows = ndims[m];
		A->ncols = 1;
		for(size_t i=0; i<nmodes-1; ++i) {
			size_t new_i = new_order[i];
			A->ncols *= ndims[new_i];
		}

		sptCopySizeVector(&(A->rowind), &(X->inds[m]));
		for(size_t x=0; x<nnz; ++x) {
			size_t col = 0;
			for(size_t i=0; i<nmodes-1; ++i) {
				size_t new_i = new_order[i];
				size_t index = X->inds[new_i].data[x];
				col += index * strides[i];
			}
			sptAppendSizeVector(&(A->colind), col);
			sptAppendVector(&(A->values), vals[x]);
		}

	} else if(transpose == 0) {	// mode-m as column
		A->ncols = ndims[m];
		A->nrows = 1;
		for(size_t i=0; i<nmodes-1; ++i) {
			size_t new_i = new_order[i];
			A->nrows *= ndims[new_i];
		}

		sptCopySizeVector(&(A->colind), &(X->inds[m]));
		for(size_t x=0; x<nnz; ++x) {
			size_t row = 0;
			for(size_t i=0; i<nmodes-1; ++i) {
				size_t new_i = new_order[i];
				size_t index = X->inds[new_i].data[x];
				row += index * strides[i];
			}
			sptAppendSizeVector(&(A->colind), row);
			sptAppendVector(&(A->values), vals[x]);
		}

	} else{
		fprintf(stderr, "SpTOL ERROR: incorrect transpose value.\n");
		exit(-1);
	}

	free(strides);
	free(new_order);
}