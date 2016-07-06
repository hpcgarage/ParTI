
// TODO(change): the whole file

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sort.h"


/******************************************************************************
 * DEFINES
 *****************************************************************************/
/* switch to insertion sort past this point */
#define MIN_QUICKSORT_SIZE 8



/******************************************************************************
 * STATIC FUNCTIONS
 *****************************************************************************/

/**
* @brief Compares ind*[i] and ind*[j] for n-mode tensors.
*
* @param tt The tensor we are sorting.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param i The index into ind*.
* @param j The second index into ind*.
*
* @return Returns -1 if ind[i] < ind[j], 1 if ind[i] > ind[j], and 0 if they
*         are equal.
*/
static inline int p_ttcmp(
  SparseTensorCOO const * const tt,
  IndexType const * const cmplt,
  IndexType const i,
  IndexType const j)
{
  for(IndexType m=0; m < tt->nmodes; ++m) {
    if(tt->inds[cmplt[m]][i] < tt->inds[cmplt[m]][j]) {
      return -1;
    } else if(tt->inds[cmplt[m]][j] < tt->inds[cmplt[m]][i]) {
      return 1;
    }
  }
  return 0;
}


/**
* @brief Compares ind*[i] and ind*[j] for n-mode tensors.
*
* @param tt The tensor we are sorting.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param i The index into ind*.
* @param j The coordinate we are comparing against.
*
* @return Returns -1 if ind[i] < j, 1 if ind[i] > j, and 0 if they are equal.
*/
static inline int p_ttqcmp(
  SparseTensorCOO const * const tt,
  IndexType const * const cmplt,
  IndexType const i,
  IndexType const j[MAX_NMODES])
{
  for(IndexType m=0; m < tt->nmodes; ++m) {
    if(tt->inds[cmplt[m]][i] < j[cmplt[m]]) {
      return -1;
    } else if(j[cmplt[m]] < tt->inds[cmplt[m]][i]) {
      return 1;
    }
  }
  return 0;
}


/**
* @brief Swap nonzeros i and j.
*
* @param tt The tensor to operate on.
* @param i The first nonzero to swap.
* @param j The second nonzero to swap with.
*/
static inline void p_ttswap(
  SparseTensorCOO * const tt,
  IndexType const i,
  IndexType const j)
{
  ValueType vtmp = tt->vals[i];
  tt->vals[i] = tt->vals[j];
  tt->vals[j] = vtmp;

  IndexType itmp;
  for(IndexType m=0; m < tt->nmodes; ++m) {
    itmp = tt->inds[m][i];
    tt->inds[m][i] = tt->inds[m][j];
    tt->inds[m][j] = itmp;
  }
}



/**
* @brief Perform insertion sort on an n-mode tensor between start and end.
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param start The first nonzero to sort.
* @param end The last nonzero to sort.
*/
static void p_tt_insertionsort(
  SparseTensorCOO * const tt,
  IndexType const * const cmplt,
  IndexType const start,
  IndexType const end)
{
  IndexType * ind;
  ValueType * const vals = tt->vals;
  IndexType const nmodes = tt->nmodes;

  ValueType vbuf;
  IndexType ibuf;

  for(size_t i=start+1; i < end; ++i) {
    size_t j = i;
    while (j > 0 && p_ttcmp(tt, cmplt, i, j-1) < 0) {
      --j;
    }

    vbuf = vals[i];

    /* shift all data */
    memmove(vals+j+1, vals+j, (i-j)*sizeof(ValueType));
    vals[j] = vbuf;
    for(IndexType m=0; m < nmodes; ++m) {
      ind = tt->inds[m];
      ibuf = ind[i];
      memmove(ind+j+1, ind+j, (i-j)*sizeof(IndexType));
      ind[j] = ibuf;
    }
  }
}


/**
* @brief Perform quicksort on a n-mode tensor between start and end.
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param start The first nonzero to sort.
* @param end The last nonzero to sort.
*/
static void p_tt_quicksort(
  SparseTensorCOO * const tt,
  IndexType const * const cmplt,
  IndexType const start,
  IndexType const end)
{
  ValueType vmid;
  IndexType imid[MAX_NMODES];

  IndexType * ind;
  ValueType * const vals = tt->vals;
  IndexType const nmodes = tt->nmodes;

  if((end-start) <= MIN_QUICKSORT_SIZE) {
    p_tt_insertionsort(tt, cmplt, start, end);
  } else {
    size_t i = start+1;
    size_t j = end-1;
    size_t k = start + ((end - start) / 2);

    /* grab pivot */
    vmid = vals[k];
    vals[k] = vals[start];
    for(IndexType m=0; m < nmodes; ++m) {
      ind = tt->inds[m];
      imid[m] = ind[k];
      ind[k] = ind[start];
    }

    while(i < j) {
      /* if tt[i] > mid  -> tt[i] is on wrong side */
      if(p_ttqcmp(tt,cmplt,i,imid) == 1) {
        /* if tt[j] <= mid  -> swap tt[i] and tt[j] */
        if(p_ttqcmp(tt,cmplt,j,imid) < 1) {
          p_ttswap(tt,i,j);
          ++i;
        }
        --j;
      } else {
        /* if tt[j] > mid  -> tt[j] is on right side */
        if(p_ttqcmp(tt,cmplt,j,imid) == 1) {
          --j;
        }
        ++i;
      }
    }

    /* if tt[i] > mid */
    if(p_ttqcmp(tt,cmplt,i,imid) == 1) {
      --i;
    }
    vals[start] = vals[i];
    vals[i] = vmid;
    for(IndexType m=0; m < nmodes; ++m) {
      ind = tt->inds[m];
      ind[start] = ind[i];
      ind[i] = imid[m];
    }

    if(i > start + 1) {
      p_tt_quicksort(tt, cmplt, start, i);
    }
    ++i; /* skip the pivot element */
    if(end - i > 1) {
      p_tt_quicksort(tt, cmplt, i, end);
    }
  }
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void tt_sort(
  SparseTensorCOO * const tt,
  IndexType const mode,
  IndexType * dim_perm)
{
  tt_sort_range(tt, mode, dim_perm, 0, tt->nnz);
}


void tt_sort_range(
  SparseTensorCOO * const tt,
  IndexType const mode,
  IndexType * dim_perm,
  IndexType const start,
  IndexType const end)
{
  IndexType * cmplt;
  if(dim_perm == NULL) {
    cmplt = (IndexType*) AlignedMalloc(tt->nmodes * sizeof(IndexType));
    cmplt[0] = mode;
    for(IndexType m=1; m < tt->nmodes; ++m) {
      cmplt[m] = (mode + m) % tt->nmodes;
    }
  } else {
    cmplt = dim_perm;
  }

  timer_start(&g_timers[TIMER_SORT]);
  p_tt_quicksort(tt, cmplt, start, end);

  if(dim_perm == NULL) {
    free(cmplt);
  }
  timer_stop(&g_timers[TIMER_SORT]);
}


void insertion_sort(
  IndexType * const a,
  IndexType const n)
{
  timer_start(&g_timers[TIMER_SORT]);
  for(size_t i=1; i < n; ++i) {
    IndexType b = a[i];
    size_t j = i;
    while (j > 0 &&  a[j-1] > b) {
      --j;
    }
    memmove(a+(j+1), a+j, sizeof(IndexType)*(i-j));
    a[j] = b;
  }
  timer_stop(&g_timers[TIMER_SORT]);
}


void quicksort(
  IndexType * const a,
  IndexType const n)
{
  timer_start(&g_timers[TIMER_SORT]);
  if(n < MIN_QUICKSORT_SIZE) {
    insertion_sort(a, n);
  } else {
    size_t i = 1;
    size_t j = n-1;
    size_t k = n >> 1;
    IndexType mid = a[k];
    a[k] = a[0];
    while(i < j) {
      if(a[i] > mid) { /* a[i] is on the wrong side */
        if(a[j] <= mid) { /* swap a[i] and a[j] */
          IndexType tmp = a[i];
          a[i] = a[j];
          a[j] = tmp;
          ++i;
        }
        --j;
      } else {
        if(a[j] > mid) { /* a[j] is on the right side */
          --j;
        }
        ++i;
      }
    }

    if(a[i] > mid) {
      --i;
    }
    a[0] = a[i];
    a[i] = mid;

    if(i > 1) {
      quicksort(a,i);
    }
    ++i; /* skip the pivot element */
    if(n-i > 1) {
      quicksort(a+i, n-i);
    }
  }
  timer_stop(&g_timers[TIMER_SORT]);
}

