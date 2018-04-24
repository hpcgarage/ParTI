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
#include "hicoo.h"
#include <assert.h>

void sptSparseTensorStatusHiCOO(sptSparseTensorHiCOO *hitsr, FILE *fp)
{
  sptIndex nmodes = hitsr->nmodes;
  fprintf(fp, "HiCOO Sparse Tensor information ---------\n");
  fprintf(fp, "DIMS=%"PARTI_PRI_INDEX, hitsr->ndims[0]);
  for(sptIndex m=1; m < nmodes; ++m) {
    fprintf(fp, "x%"PARTI_PRI_INDEX, hitsr->ndims[m]);
  }
  fprintf(fp, " NNZ=%"PARTI_PRI_NNZ_INDEX, hitsr->nnz);
  fprintf(fp, "\n");
  fprintf(fp, "sb=%"PARTI_PRI_INDEX, (sptIndex)pow(2, hitsr->sb_bits));
  fprintf(fp, " sk=%"PARTI_PRI_INDEX, (sptIndex)pow(2, hitsr->sk_bits));
  fprintf(fp, " sc=%"PARTI_PRI_INDEX, (sptIndex)pow(2, hitsr->sc_bits));
  fprintf(fp, "\n");
  fprintf(fp, "nb=%"PARTI_PRI_NNZ_INDEX, hitsr->bptr.len - 1);
  fprintf(fp, " nk=%"PARTI_PRI_NNZ_INDEX, hitsr->kptr.len - 1);
  fprintf(fp, " nc=%"PARTI_PRI_NNZ_INDEX, hitsr->cptr.len - 1);
  fprintf(fp, "\n");

  sptNnzIndex bytes = hitsr->nnz * ( sizeof(sptValue) + nmodes * sizeof(sptElementIndex) );
  bytes += hitsr->binds[0].len * nmodes * sizeof(sptBlockIndex);
  bytes += hitsr->bptr.len * sizeof(sptNnzIndex);
  bytes += hitsr->kptr.len * sizeof(sptNnzIndex);
  bytes += hitsr->cptr.len * sizeof(sptNnzIndex);
  /* add kschr */
  sptIndex sk = (sptIndex)pow(2, hitsr->sk_bits);
  for(sptIndex m=0; m < nmodes; ++m) {
    sptIndex kernel_ndim = (hitsr->ndims[m] + sk - 1)/sk;
    for(sptIndex i=0; i < kernel_ndim; ++i) {
      bytes += hitsr->kschr[m][i].len * sizeof(sptIndex);
    }
    bytes += kernel_ndim * sizeof(sptIndexVector *);
  }
  bytes += nmodes * sizeof(sptIndexVector **);
  /* add nkiters  */
  bytes += nmodes * sizeof(sptIndex);

  char * bytestr = sptBytesString(bytes);
  fprintf(fp, "HiCOO-STORAGE=%s\n", bytestr);
  free(bytestr);

  fprintf(fp, "SCHEDULE INFO [KERNEL]: \n");
  for(sptIndex m=0; m < nmodes; ++m) {
    sptIndex kernel_ndim = (hitsr->ndims[m] + sk - 1)/sk;
    fprintf(fp, "SCHEDULE MODE %"PARTI_PRI_INDEX" : %"PARTI_PRI_INDEX" x %"PARTI_PRI_INDEX"\n", m, kernel_ndim, hitsr->nkiters[m]);
  }

  // fprintf(fp, "SCHEDULE DETAILS (kschr): \n");
  // for(sptIndex m=0; m < nmodes; ++m) {
  //   printf("Mode %u\n", m);
  //   sptIndex kernel_ndim = (hitsr->ndims[m] + sk - 1)/sk;
  //   for(sptIndex i=0; i < kernel_ndim; ++i) {
  //     sptDumpIndexVector(&hitsr->kschr[m][i], fp);
  //   }
  //   fprintf(fp, "\n");
  // }
  // fprintf(fp, "\n");

  sptIndex sb = (sptIndex)pow(2, hitsr->sb_bits);
  sptNnzIndex max_nnzb = hitsr->bptr.data[1] - hitsr->bptr.data[0];
  sptNnzIndex min_nnzb = hitsr->bptr.data[1] - hitsr->bptr.data[0];
  sptNnzIndex sum_nnzb = 0;
  double geo_mean_nnzb = 1;
  sptNnzIndex nb = hitsr->bptr.len - 1;
  sptNnzIndex * nnzb_array = (sptNnzIndex *)malloc(nb * sizeof(* nnzb_array));
  fprintf(fp, "block nnzs:\n");
  for(sptNnzIndex i=0; i < hitsr->bptr.len - 1; ++i) {
    sptNnzIndex nnzb = hitsr->bptr.data[i+1] - hitsr->bptr.data[i];
    // fprintf(fp, "%lu, ", nnzb);
    if(max_nnzb < nnzb) {
      max_nnzb = nnzb;
    }
    if(min_nnzb > nnzb) {
      min_nnzb = nnzb;
    }
    sum_nnzb += nnzb;
    geo_mean_nnzb *= pow( (double)nnzb / sb, 1.0/nb );
    nnzb_array[i] = nnzb;
  }
  assert(sum_nnzb == hitsr->nnz);
  sptNnzIndex aver_nnzb = (sptNnzIndex)sum_nnzb / (hitsr->bptr.len - 1);

  /* Compute median */
  sptQuickSortNnzIndexArray(nnzb_array, 0, nb);
  sptNnzIndex median_loc = (nb + 1) / 2 - 1;
  assert (median_loc >= 0);
  sptNnzIndex median_nnzb = nnzb_array[median_loc];
  free(nnzb_array);
  
  fprintf(fp, "Nnzb: Max=%lu, Min=%lu, Aver=%lu\n", max_nnzb, min_nnzb, aver_nnzb);
  fprintf(fp, "cb: Max=%.3lf, Min=%.3lf, Aver=%.3lf\n", (double)max_nnzb / sb, (double)min_nnzb / sb, (double)aver_nnzb / sb);
  fprintf(fp, "median cb: %.3lf, geometric mean cb: %.3lf\n", (double)median_nnzb / sb, geo_mean_nnzb);
  fprintf(fp, "alpha_b: %lf\n", (double)(hitsr->bptr.len - 1) / hitsr->nnz);

  fprintf(fp, "\nParameter configuration --------\n");
  fprintf(fp, "Suggest B (sb) <= %.2lf / R. For cache efficiency\n", (double)L1_SIZE / hitsr->nmodes / sizeof(sptValue));
  fprintf(fp, "Suggest alpha_b in (0,1], small is better. For tensor storage\n");
  fprintf(fp, "Suggest cb > 1, large is better. For MTTKRP performance\n");
  fprintf(fp, "Suggest num_tasks should in [%d, %d] (PAR_DEGREE: %d). For parallel efficiency\n", NUM_CORES, PAR_DEGREE * NUM_CORES, PAR_DEGREE);
  fprintf(fp, "\n\n");

}