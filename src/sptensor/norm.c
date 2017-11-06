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

double spt_SparseTensorNorm(const sptSparseTensor *X) {
    double sqnorm = 0;
    sptNnzIndex i;
    for(i = 0; i < X->nnz; ++i) {
        double cell_value = X->values.data[i];
        sqnorm += cell_value * cell_value;
    }
    return sqrt(sqnorm);
}
