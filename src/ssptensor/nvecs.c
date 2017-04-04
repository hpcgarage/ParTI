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
    /*
    TODO:
    tnt = double(sptenmat(t,n,'t'));
    y = tnt' * tnt;
    opts.disp = 0;
    [u,d] = eigs(y,r,'LM',eigsopts);
    */

    return 0;
}
