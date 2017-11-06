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
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
    

sptNnzIndex sptMaxNnzIndexArray(
  sptNnzIndex const * const indices,
  sptNnzIndex const size)
{
  sptNnzIndex max = indices[0];
  for(sptNnzIndex i=1; i < size; ++i) {
    if(indices[i] > max) {
      max = indices[i];
    }
  }
  return max;
}


sptIndex sptMaxIndexArray(
  sptIndex const * const indices,
  sptNnzIndex const size)
{
  sptIndex max = indices[0];
  for(sptNnzIndex i=1; i < size; ++i) {
    if(indices[i] > max) {
      max = indices[i];
    }
  }
  return max;
}


char * sptBytesString(uint64_t const bytes)
{
  double size = (double)bytes;
  int suff = 0;
  const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
  while(size > 1024 && suff < 5) {
    size /= 1024.;
    ++suff;
  }
  char * ret = NULL;
  if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    fprintf(stderr, "SPT: asprintf failed with%zu bytes.\n", bytes);
    ret = NULL;
  }
  return ret;
}


sptValue sptRandomValue(void)
{
  sptValue v =  3.0 * ((sptValue) rand() / (sptValue) RAND_MAX);
  if(rand() % 2 == 0) {
    v *= -1;
  }
  return v;
}
