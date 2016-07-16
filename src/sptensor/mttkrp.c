#include <SpTOL.h>
#include "sptensor.h"

void sptMTTKRP(sptSparseTensor const * const X, 
	sptMatrix ** const mats, 	// mats[nmodes] as temporary space.
	size_t const * const mats_order,	// Correspond to the mode order of X.
	size_t const mode,
	sptScalar * const scratch) {

	size_t const nmodes = X->nmodes;
	size_t const nnz = X->nnz;
	size_t const * const ndims = X->ndims;
	sptScalar const * const vals = X->values.data;
	size_t const nmats = nmodes - 1;

	/* Check the mats. */
	for(size_t i=0; i<nmodes; ++i) {
		assert(mats[i]->ncols == mats[nmodes]->ncols);
		assert(mats[i]->nrows == ndims[i]);
	}

	size_t const I = mats[mode]->nrows;
	size_t const R = mats[mode]->ncols;
	size_t const * const mode_ind = X->inds[mode].data;
	sptMatrix * const M = mats[nmodes];
	sptScalar * const mvals = M->values;
	memset(mvals, 0, I*R*sizeof(sptScalar));

	// TODO: omp partition by mode
	for(size_t x=0; x<nnz; ++x) {

		size_t times_mat_index = mats_order[0];
		sptMatrix * times_mat = mats[times_mat_index];
		size_t * times_inds = X->inds[times_mat_index].data;
		size_t tmp_i = times_inds[x];
		for(size_t r=0; r<R; ++r) {
			scratch[r] = times_mat->values[tmp_i * R + r];
		}

		for(size_t i=1; i<nmats; ++i) {
			times_mat_index = mats_order[i];
			times_mat = mats[times_mat_index];
			times_inds = X->inds[times_mat_index].data;
			tmp_i = times_inds[x];

			for(size_t r=0; r<R; ++r) {
				scratch[r] += times_mat->values[tmp_i * R + r];
			}
		}

		sptScalar const entry = vals[x];
		size_t const mode_i = mode_ind[x];
		for(size_t r=0; r<R; ++r) {
			mvals[mode_i * R + r] += entry * scratch[r];
		}
	}

}