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

#ifndef DCP_SORT_H
#define DCP_SORT_H


// TODO(change): the whole file

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sparsetensor.h"
#include "timer.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

/**
* @brief Sort a tensor using a permutation of its modes. Sorting uses dim_perm
*        to order modes by decreasing priority. If dim_perm = {1, 0, 2} then
*        nonzeros will be ordered by ind[1], with ties broken by ind[0], and
*        finally deferring to ind[2].
*
* @param tt The tensor to sort.
* @param mode The primary for sorting.
* @param dim_perm An permutation array that defines sorting priority. If NULL,
*                 a default ordering of {0, 1, ..., m} is used.
*/
void tt_sort(
  SparseTensorCOO * const tt,
  IndexType const mode,
  IndexType * dim_perm);


/**
* @brief Sort a tensor using tt_sort on only a range of the nonzero elements.
*        Nonzeros in the range [start, end) will be sorted.
*
* @param tt The tensor to sort.
* @param mode The primary for sorting.
* @param dim_perm An permutation array that defines sorting priority. If NULL,
*                 a default ordering of {0, 1, ..., m} is used.
* @param start The first nonzero to include in the sorting.
* @param end The end of the nonzeros to sort (exclusive).
*/
void tt_sort_range(
  SparseTensorCOO * const tt,
  IndexType const mode,
  IndexType * dim_perm,
  IndexType const start,
  IndexType const end);


/**
* @brief An in-place insertion sort implementation for IndexType's.
*
* @param a The array to sort.
* @param n The number of items to sort.
*/
void insertion_sort(
  IndexType * const a,
  IndexType const n);


/**
* @brief An in-place quicksort implementation for IndexType's.
*
* @param a The array to sort.
* @param n The number of items to sort.
*/
void quicksort(
  IndexType * const a,
  IndexType const n);

#endif
