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
#include "../ssptensor/ssptensor.h"
#include <stdlib.h>
#include <math.h>

/*
  (sb) TODO:
  - Get to know types of each variable
  - Norm & Eigen calculation from cuSOLVER
  - Implement one-copy version of sspTTM
*/

int sptTuckerDecomposition(
    sptSemiSparseTensor   *core,
    const sptSparseTensor *X,
    const size_t          R[],
    double                tol /* = 1.0e-4 */,
    unsigned              maxiters /* = 50 */,
    const size_t          dimorder[]
) {
    size_t N = X->nmodes;
    double normX = spt_SparseTensorNorm(X);
    sptMatrix *U = malloc(N * sizeof *U);

    memset(core, 0, sizeof *core);

    // Random init
    for(size_t ni = 1; ni < N; ++ni) {
        size_t n = dimorder[ni];
        sptRandomizeMatrix(&U[n], X->ndims[n], R[n]);
    }

    double fit = 0;

    for(unsigned iter = 0; iter < maxiters; ++iter) {
        double fitold = fit;

        sptSemiSparseTensor Utilde;
        sptSparseTensorToSemiSparseTensor(&Utilde, X, dimorder[0]);
        for(size_t ni = 0; ni < N; ++ni) {
            size_t n = dimorder[ni];
            size_t m;
            // TODO: zero-copy TTM?
            // TODO: transpose?
            for(m = 0; m < N; ++m) {
                if(m != n) {
                    sptSemiSparseTensor Utilde_next;
                    sptSemiSparseTensorMulMatrix(&Utilde_next, &Utilde, &U[m], m);
                    sptFreeSemiSparseTensor(&Utilde);
                    Utilde = Utilde_next;
                }
            }
            // U[n] = nvecs(Utilde, n, R[n]);
        }

        if(core->nmodes != 0) {
            sptFreeSemiSparseTensor(core);
        }

        sptSemiSparseTensorMulMatrix(core, &Utilde, &U[dimorder[N-1]], dimorder[N-1]);

        double normresidual = hypot(normX, spt_SemiSparseTensorNorm(core));
        fit = 1 - normresidual / normX;
        double fitchange = abs(fitold - fit);

        if(iter != 0 && fitchange < tol) {
            break;
        }
    }

    return 0;
}
