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


static inline int spt_PairCompareIndices(sptKeyValuePair const * kvarray, sptIndex loc1, sptIndex loc2) {

    if(kvarray[loc1].value < kvarray[loc2].value) {
        return -1;
    } else if(kvarray[loc1].value > kvarray[loc2].value) {
        return 1;
    } else {
        return 0;
    }
}


static inline void spt_SwapPairs(sptKeyValuePair * kvarray, sptIndex const ind1, sptIndex const ind2) {
    
    sptIndex eleind1 = kvarray[ind1].key;
    kvarray[ind1].key = kvarray[ind2].key;
    kvarray[ind2].key = eleind1;

    sptIndex val1 = kvarray[ind1].value;
    kvarray[ind1].value = kvarray[ind2].value;
    kvarray[ind2].value = val1;
}

static void spt_QuickSortPairArray(sptKeyValuePair const * kvarray, sptIndex const l, sptIndex const r)
{
    sptIndex i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(spt_PairCompareIndices(kvarray, i, p) < 0) {
            ++i;
        }
        while(spt_PairCompareIndices(kvarray, p, j) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        spt_SwapPairs(kvarray, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }

    spt_QuickSortPairArray(kvarray, l, i);
    spt_QuickSortPairArray(kvarray, i, r);

}

/**
 * Increasingly sort an key-value pair array in type sptIndex.
 *
 * @param array a pointer to an array to be sorted,
 * @param length number of values 
 *
 */
void sptPairArraySort(sptKeyValuePair const * kvarray, sptIndex const length)
{
    spt_QuickSortPairArray(kvarray, 0, length);
}