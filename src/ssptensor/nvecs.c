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
#include "ssptensor.h"
#include <math.h>

int sptSemiSparseTensorNvecs(
    sptSparseTensor           *u,
    const sptSemiSparseTensor *t,
    size_t                    n,
    size_t                    r
) {
    /* TODO */
    sptMatrix tnt;
    /* NOTE:
    %   A = SPTENMAT(T, RDIMS) creates a sparse matrix representation of
    %   an sptensor T.  The dimensions (or modes) specified in RDIMS map
    %   to the rows of the matrix, and the remaining dimensions (in
    %   ascending order) map to the columns.
    */
    /* For modes > 3, concatenate remaining dimensions */
    /* sptenmat(&tnt, t, n, 't'); */
    sptMatrix y;
    /* y = tnt' * tnt; */
    /* [u,d] = eigs(y,r,'LM',eigsopts); */
    /* LM means Largest Magnitude (default) */
    /* Refer to OpenBLAS
    Naming convension: http://www.netlib.org/lapack/lug/node24.html
    We need S**EV?
    */

    return 0;
}
