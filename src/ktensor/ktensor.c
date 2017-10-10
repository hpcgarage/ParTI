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
#include <stdlib.h>
#include <string.h>
#include "../error/error.h"

int sptNewKruskalTensor(sptKruskalTensor *ktsr, sptIndex nmodes, const size_t ndims[], sptIndex rank)
{
    ktsr->nmodes = nmodes;
    ktsr->rank = rank;
    ktsr->ndims = (sptIndex*)malloc(nmodes*sizeof(sptIndex));
    for(sptIndex i=0; i<nmodes; ++i)
        ktsr->ndims[i] = (sptIndex) ndims[i];
    ktsr->lambda = (sptValue*)malloc(rank*sizeof(sptValue));
    ktsr->fit = 0.0;
    
	return 0;
}

void sptFreeKruskalTensor(sptKruskalTensor *ktsr)
{
	ktsr->rank = 0;
	ktsr->fit = 0.0;
	free(ktsr->ndims);
	free(ktsr->lambda);
	for(size_t i=0; i<ktsr->nmodes; ++i)
		sptFreeMatrix(ktsr->factors[i]);
    free(ktsr->factors);
	ktsr->nmodes = 0;
}

double KruskalTensorFit(
  sptSparseTensor const * const spten,
  sptValue const * const __restrict lambda,
  sptMatrix ** mats,
  sptMatrix const * const tmp_mat,
  sptMatrix ** ata) 
{
  sptIndex const nmodes = spten->nmodes;

  double spten_normsq = SparseTensorFrobeniusNormSquared(spten);
  // printf("spten_normsq: %lf\n", spten_normsq);
  double const norm_mats = KruskalTensorFrobeniusNormSquared(nmodes, lambda, ata);
  // printf("norm_mats: %lf\n", norm_mats);
  double const inner = SparseKruskalTensorInnerProduct(nmodes, lambda, mats, tmp_mat);
  // printf("inner: %lf\n", inner);
  double const residual = sqrt(spten_normsq + norm_mats - (2 * inner));
  // printf("residual: %lf\n", residual);
  double fit = 1 - (residual / sqrt(spten_normsq));

  return fit;
}


double KruskalTensorFrobeniusNormSquared(
  sptIndex const nmodes,
  sptValue const * const __restrict lambda,
  sptMatrix ** ata) 
{
  sptIndex const rank = ata[0]->ncols;
  sptValue * const __restrict tmp_atavals = ata[nmodes]->values;
  double norm_mats = 0;

  for(sptIndex x=0; x < rank*rank; ++x) {
    tmp_atavals[x] = 1.;
  }

  for(sptIndex m=0; m < nmodes; ++m) {
    sptValue const * const __restrict atavals = ata[m]->values;
    for(sptIndex x=0; x < rank*rank; ++x) {
      tmp_atavals[x] *= atavals[x];
    }
  }

  for(sptIndex i=0; i < rank; ++i) {
    for(sptIndex j=0; j < rank; ++j) {
      norm_mats += tmp_atavals[j+(i*rank)] * lambda[i] * lambda[j];
    }
  }

  return fabs(norm_mats);
}

double SparseKruskalTensorInnerProduct(
  sptIndex const nmodes,
  sptValue const * const __restrict lambda,
  sptMatrix ** mats,
  sptMatrix const * const tmp_mat) 
{
  sptIndex const rank = mats[0]->ncols;
  sptIndex const last_mode = nmodes - 1;
  sptIndex const I = tmp_mat->nrows;

  sptValue const * const last_vals = mats[last_mode]->values;
  sptValue const * const tmp_vals = tmp_mat->values;

  double inner = 0;

  double * const __restrict accum = (double *) malloc(rank*sizeof(*accum));

  for(sptIndex r=0; r < rank; ++r) {
    accum[r] = 0.0;
  }

  for(sptIndex i=0; i < I; ++i) {
    for(sptIndex r=0; r < rank; ++r) {
      accum[r] += last_vals[r+(i*rank)] * tmp_vals[r+(i*rank)];
    }
  }

  for(sptIndex r=0; r < rank; ++r) {
    inner += accum[r] * lambda[r];
  }


  return inner;
}