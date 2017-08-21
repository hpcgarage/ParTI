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
#include "sptensor.h"
#include <math.h>


double sptSparseTensorDensity(sptSparseTensor const * const tsr)
{
  double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
  double density = 1.0;
  for(sptIndex m=0; m < tsr->nmodes; ++m) {
    density *= root / (double)tsr->ndims[m];
  }

  return density;
}



void sptSparseTensorStatus(sptSparseTensor *tsr, FILE *fp)
{
  fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
  fprintf(fp, "DIMS=%zu", tsr->ndims[0]);
  for(sptIndex m=1; m < tsr->nmodes; ++m) {
    fprintf(fp, "x%zu", tsr->ndims[m]);
  }
  fprintf(fp, " NNZ=%zu", tsr->nnz);
  fprintf(fp, " DENSITY=%e\n" , sptSparseTensorDensity(tsr));

  char * bytestr = sptBytesString(tsr->nnz * (sizeof(sptIndex) * tsr->nmodes + sizeof(sptValue)));
  fprintf(fp, "COO-STORAGE=%s\n", bytestr);
  fprintf(fp, "\n");
  free(bytestr);
}
