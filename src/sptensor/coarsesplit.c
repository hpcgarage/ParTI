/*
    This file is part of SpTOL.

    SpTOL is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    SpTOL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with SpTOL.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <assert.h>
#include <SpTOL.h>
#include "sptensor.h"

int sptCoarseSplitSparseTensor(sptSparseTensor *tsr, const int num, sptSparseTensor *cstsr) {
    int result = 0;
    assert(num > 1);

    size_t const nmodes = tsr->nmodes;
    size_t const nnz = tsr->nnz;
    size_t const * ndims = tsr->ndims;
    sptSizeVector * inds = tsr->inds;
    sptVector values = tsr->values;

    sptSparseTensorSortIndex(tsr);

    size_t * csnnz = (size_t *)malloc(num * sizeof(size_t));
    memset(csnnz, 0, num * sizeof(size_t));
    size_t aver_nnz = nnz / num;


    size_t ** csndims = (size_t**)malloc(num* sizeof(size_t*));
    for(int n=0; n<num; ++n) {
      csndims[n] = (size_t*)malloc(nmodes * sizeof(size_t));
      memset(csndims[n], 0, nmodes * sizeof(size_t));
    }
    size_t * slice_nnzs = (size_t *)malloc(ndims[0] * sizeof(size_t));
    memset(slice_nnzs, 0, ndims[0] * sizeof(size_t));
    for(size_t i=0; i<nnz; ++i) {
        size_t tmp_ind = inds[0].data[i];
        ++ slice_nnzs[tmp_ind];
    }
    // printf("slice_nnzs:\n");
    // for(size_t n=0; n<ndims[0]; ++n) {
    //     printf("%lu ", slice_nnzs[n]);
    // }
    // printf("\n");

    int j = 0;
    for(size_t i=0; i<ndims[0]; ++i) {
        if(csnnz[j] < aver_nnz || j == num - 1) {
            csnnz[j] += slice_nnzs[i];
            ++ csndims[j][0];
        } else {
            ++ j;
            csnnz[j] = slice_nnzs[i];
            csndims[j][0] = 1;
        }
    }
    printf("csnnz:\n");
    for(int n=0; n<num; ++n) {
        printf("%ld ", csnnz[n]);
    }
    printf("\n");
    assert(j == num-1);

    for(int n=0; n<num; ++n) {
      for(size_t m=1; m<nmodes; ++m) {
        csndims[n][m] = ndims[m];
      }
    }

    size_t * nnz_loc = (size_t*)malloc(num*sizeof(size_t));
    memset(nnz_loc, 0, num*sizeof(size_t));
    for(int n=1; n<num; ++n) {
      nnz_loc[n] = nnz_loc[n-1] + csnnz[n-1];
    }

    for(int n=0; n<num; ++n) {
      sptNewSparseTensor(cstsr+n, nmodes, csndims[n]);
      cstsr[n].nnz = csnnz[n];
      for(size_t m=0; m<nmodes; ++m) {
        cstsr[n].inds[m].len = csnnz[n];
        cstsr[n].inds[m].cap = csnnz[n];
        cstsr[n].inds[m].data = inds[m].data + nnz_loc[n];
      }
      cstsr[n].values.len = csnnz[n];
      cstsr[n].values.cap = csnnz[n];
      cstsr[n].values.data = values.data + nnz_loc[n];
    }

    // sptDumpSparseTensor(tsr, 0, stdout);
    // printf("\n");
    // for(int n=0; n<num; ++n) {
    //     sptDumpSparseTensor(cstsr+n, 0, stdout);
    //     printf("\n");
    // }

    free(csnnz);
    free(nnz_loc);
    for(int n=0; n<num; ++n)
        free(csndims[n]);
    free(csndims);
    free(slice_nnzs);

    return 0;
}
