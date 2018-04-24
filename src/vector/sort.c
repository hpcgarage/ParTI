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

void sptQuickSortNnzIndexArray(sptNnzIndex * array, sptNnzIndex l, sptNnzIndex r) {
    sptNnzIndex i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(array[i] < array[p]) {
            ++i;
        }
        while(array[p] < array[j]) {
            --j;
        }
        if(i >= j) {
            break;
        }
        sptNnzIndex tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    sptQuickSortNnzIndexArray(array, l, i);
    sptQuickSortNnzIndexArray(array, i, r);
}