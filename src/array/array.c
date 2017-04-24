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

size_t sptMaxSizeArray(
  size_t const * const indices,
  size_t const size)
{
  size_t max = indices[0];
  for(size_t i=1; i < size; ++i) {
    if(indices[i] > max) {
      max = indices[i];
    }
  }
  return max;
}

void spt_DumpArray(const size_t array[], size_t length, size_t start_index, FILE *fp) {
    if(length == 0) {
        return;
    }
    fprintf(fp, "%zu", array[0] + start_index);
    size_t i;
    for(i = 1; i < length; ++i) {
        fprintf(fp, ", %zu", array[i] + start_index);
    }
    fprintf(fp, "\n");
}