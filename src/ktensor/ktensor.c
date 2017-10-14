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
  sptMatrix ** ata) 
{
  sptIndex const nmodes = spten->nmodes;

  double spten_normsq = SparseTensorFrobeniusNormSquared(spten);
  printf("spten_normsq: %lf\n", spten_normsq);
  double const norm_mats = KruskalTensorFrobeniusNormSquared(nmodes, lambda, ata);
  printf("norm_mats: %lf\n", norm_mats);
  double const inner = SparseKruskalTensorInnerProduct(nmodes, lambda, mats);
  printf("inner: %lf\n", inner);
  double residual = spten_normsq + norm_mats - 2 * inner;
  if (residual > 0.0) {
    residual = sqrt(residual);
  }
  printf("residual: %lf\n", residual);
  double fit = 1 - (residual / sqrt(spten_normsq));

  return fit;
}


// Column-major. 
/* Compute a Kruskal tensor's norm is compute on "ata"s. Check Tammy's sparse  */
double KruskalTensorFrobeniusNormSquared(
  sptIndex const nmodes,
  sptValue const * const __restrict lambda,
  sptMatrix ** ata) // ata: column-major
{
  sptIndex const rank = ata[0]->ncols;
  sptIndex const stride = ata[0]->stride;
  sptValue * const __restrict tmp_atavals = ata[nmodes]->values;    // Column-major
  double norm_mats = 0;

  for(sptIndex x=0; x < rank*stride; ++x) {
    tmp_atavals[x] = 1.;
  }

  /* Compute Hadamard product for all "ata"s */
  for(sptIndex m=0; m < nmodes; ++m) {
    sptValue const * const __restrict atavals = ata[m]->values;
    for(size_t i=0; i < rank; ++i) {
        for(size_t j=0; j < rank; ++j) {
            tmp_atavals[j * stride + i] *= atavals[j * stride + i];
        }
    }
  }

  /* compute lambda^T * aTa[MAX_NMODES] * lambda, only compute a half of them because of its symmetric */
  for(sptIndex i=0; i < rank; ++i) {
    norm_mats += tmp_atavals[i+(i*stride)] * lambda[i] * lambda[i];
    for(sptIndex j=i+1; j < rank; ++j) {
      norm_mats += tmp_atavals[i+(j*stride)] * lambda[i] * lambda[j] * 2;
    }
  }

  return fabs(norm_mats);
}

// Row-major, compute via MTTKRP result (mats[nmodes]) and mats[nmodes-1].
double SparseKruskalTensorInnerProduct(
  sptIndex const nmodes,
  sptValue const * const __restrict lambda,
  sptMatrix ** mats) 
{
  sptIndex const rank = mats[0]->ncols;
  sptIndex const stride = mats[0]->stride;
  sptIndex const last_mode = nmodes - 1;
  sptIndex const I = mats[last_mode]->nrows;

  printf("mats[nmodes-1]:\n");
  sptDumpMatrix(mats[nmodes-1], stdout);
  printf("mats[nmodes]:\n");
  sptDumpMatrix(mats[nmodes], stdout);
  
  sptValue const * const last_vals = mats[last_mode]->values;
  sptValue const * const tmp_vals = mats[nmodes]->values;

  double inner = 0;

  double * const __restrict accum = (double *) malloc(rank*sizeof(*accum));

  for(sptIndex r=0; r < rank; ++r) {
    accum[r] = 0.0;
  }

  for(sptIndex i=0; i < I; ++i) {
    for(sptIndex r=0; r < rank; ++r) {
      accum[r] += last_vals[r+(i*stride)] * tmp_vals[r+(i*stride)];
    }
  }

  for(sptIndex r=0; r < rank; ++r) {
    inner += accum[r] * lambda[r];
  }


  return inner;
}