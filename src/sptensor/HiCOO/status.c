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


void sptSparseTensorStatusHiCOO(sptSparseTensorHiCOO *hitsr, FILE *fp)
{
  sptIndex nmodes = hitsr->nmodes;
  fprintf(fp, "HiCOO Sparse Tensor information ---------\n");
  fprintf(fp, "DIMS=%"SPT_PF_INDEX, hitsr->ndims[0]);
  for(sptIndex m=1; m < nmodes; ++m) {
    fprintf(fp, "x%"SPT_PF_INDEX, hitsr->ndims[m]);
  }
  fprintf(fp, " NNZ=%"SPT_PF_NNZINDEX, hitsr->nnz);
  fprintf(fp, "\n");

  size_t bytes = hitsr->nnz * ( sizeof(sptValue) + nmodes * sizeof(sptElementIndex) );
  bytes += hitsr->binds[0].len * nmodes * sizeof(sptBlockIndex);
  bytes += hitsr->cptr.len * sizeof(sptBlockIndex);
  bytes += hitsr->kptr.len * sizeof(sptNnzIndex);

  char * bytestr = sptBytesString(bytes);
  fprintf(fp, "HiCOO-STORAGE=%s\n", bytestr);
  fprintf(fp, "\n");
  free(bytestr);
}