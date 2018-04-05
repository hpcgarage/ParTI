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

#ifndef PARTI_MATRICES_H
#define PARTI_MATRICES_H

/* Dense matrix */
static inline sptNnzIndex sptGetMatrixLength(const sptMatrix *mtx) {
    return mtx->nrows * mtx->stride;
}
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols);
int sptRandomizeMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols);
int sptIdentityMatrix(sptMatrix *mtx);
int sptConstantMatrix(sptMatrix * const mtx, sptValue const val);
int sptCopyMatrix(sptMatrix *dest, const sptMatrix *src);
int sptAppendMatrix(sptMatrix *mtx, const sptValue values[]);
int sptResizeMatrix(sptMatrix *mtx, sptIndex const new_nrows);
void sptFreeMatrix(sptMatrix *mtx);
int sptDumpMatrix(sptMatrix *mtx, FILE *fp);

/* Dense matrix operations */
int sptMatrixDotMul(sptMatrix const * A, sptMatrix const * B, sptMatrix const * C);
int sptMatrixDotMulSeq(sptIndex const mode, sptIndex const nmodes, sptMatrix ** mats);
int sptMatrixDotMulSeqCol(sptIndex const mode, sptIndex const nmodes, sptMatrix ** mats);
int sptMatrix2Norm(sptMatrix * const A, sptValue * const lambda);
int sptMatrixMaxNorm(sptMatrix * const A, sptValue * const lambda);
void GetFinalLambda(
  sptIndex const rank,
  sptIndex const nmodes,
  sptMatrix ** mats,
  sptValue * const lambda);
int sptMatrixSolveNormals(
  sptIndex const mode,
  sptIndex const nmodes,
  sptMatrix ** aTa,
  sptMatrix * rhs);
int sptSparseTensorToMatrix(sptMatrix *dest, const sptSparseTensor *src);

/* Dense Rank matrix, ncols = small rank (<= 256) */
int sptNewRankMatrix(sptRankMatrix *mtx, sptIndex const nrows, sptElementIndex const ncols);
int sptRandomizeRankMatrix(sptRankMatrix *mtx, sptIndex const nrows, sptElementIndex const ncols);
int sptConstantRankMatrix(sptRankMatrix *mtx, sptValue const val);
void sptFreeRankMatrix(sptRankMatrix *mtx);
int sptDumpRankMatrix(sptRankMatrix *mtx, FILE *fp);

/* Dense rank matrix operations */
int sptRankMatrixDotMulSeqTriangle(sptIndex const mode, sptIndex const nmodes, sptRankMatrix ** mats);
int sptRankMatrix2Norm(sptRankMatrix * const A, sptValue * const lambda);
int sptRankMatrixMaxNorm(sptRankMatrix * const A, sptValue * const lambda);
void GetRankFinalLambda(
  sptElementIndex const rank,
  sptIndex const nmodes,
  sptRankMatrix ** mats,
  sptValue * const lambda);
int sptRankMatrixSolveNormals(
  sptIndex const mode,
  sptIndex const nmodes,
  sptRankMatrix ** aTa,
  sptRankMatrix * rhs);


#endif