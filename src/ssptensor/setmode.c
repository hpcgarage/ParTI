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
#include "../error/error.h"

int spt_SemiSparseTensorSetMode(
    sptSemiSparseTensor       *dest,
    const sptSemiSparseTensor *src,
    sptIndex                    newmode
) {
    int result = 0;
    /* Something like this, but better */
    sptSparseTensor tmp;
    result = sptSemiSparseTensorToSparseTensor(&tmp, src, 0);
    spt_CheckError(result, "ssp setmode", NULL);
    sptSparseTensorToSemiSparseTensor(dest, &tmp, newmode);
    spt_CheckError(result, "ssp setmode", NULL);
    sptFreeSparseTensor(&tmp);

    return 0;
}
