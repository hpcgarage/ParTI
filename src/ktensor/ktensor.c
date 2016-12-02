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
#include <stdlib.h>
#include <string.h>
#include "../error/error.h"

int sptNewKruskalTensor(sptKruskalTensor *ktsr, size_t nmodes, const size_t ndims[])
{
	// TODO
	return 0;
}

void sptFreeKruskalTensor(sptKruskalTensor *ktsr)
{
	ktsr->rank = 0;
	ktsr->fit = 0.0;
	free(ktsr->dims);
	free(ktsr->lambda);
	for(size_t i=0; i<ktsr->nmodes; ++i)
		sptFreeMatrix(ktsr->factors[i]);
    free(ktsr->factors);
	ktsr->nmodes = 0;
}
