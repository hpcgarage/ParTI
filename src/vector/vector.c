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


/**
 * Initialize a new value vector
 *
 * @param vec a valid pointer to an uninitialized sptValueVector variable,
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptNewValueVector(sptValueVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    if(cap < len) {
        cap = len;
    }
    if(cap < 2) {
        cap = 2;
    }
    vec->len = len;
    vec->cap = cap;
    vec->data = malloc(cap * sizeof *vec->data);
    spt_CheckOSError(!vec->data, "ValVec New");
    memset(vec->data, 0, cap * sizeof *vec->data);
    return 0;
}

/**
 * Fill an existed dense value vector with a specified constant
 *
 * @param vec   a valid pointer to an existed sptVector variable,
 * @param val   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptConstantValueVector(sptValueVector * const vec, sptValue const val) {
    for(sptNnzIndex i=0; i<vec->len; ++i)
        vec->data[i] = val;
    return 0;
}

/**
 * Copy a value vector to an uninitialized value vector
 *
 * @param dest a pointer to an uninitialized value vector
 * @param src  a pointer to an existing valid value vector
 *
 * The contents of `src` will be copied to `dest`.
 */
int sptCopyValueVector(sptValueVector *dest, const sptValueVector *src) {
    int result = sptNewValueVector(dest, src->len, src->len);
    spt_CheckError(result, "ValVec Copy", NULL);
    memcpy(dest->data, src->data, src->len * sizeof *src->data);
    return 0;
}

/**
 * Add a value to the end of a value vector
 *
 * @param vec   a pointer to a valid value vector
 * @param value the value to be appended
 *
 * The length of the value vector will be changed to contain the new value.
 */
int sptAppendValueVector(sptValueVector *vec, sptValue const value) {
    if(vec->cap <= vec->len) {
#ifndef MEMCHECK_MODE
        sptNnzIndex newcap = vec->cap + vec->cap/2;
#else
        sptNnzIndex newcap = vec->len+1;
#endif
        sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        spt_CheckOSError(!newdata, "ValVec Append");
        vec->cap = newcap;
        vec->data = newdata;
    }
    vec->data[vec->len] = value;
    ++vec->len;
    return 0;
}

/**
 * Add a value to the end of a value vector
 *
 * @param vec        a pointer to a valid value vector
 * @param append_vec a pointer to another value vector, containing the values to be appended
 *
 * The values from `append_vec` will be appended to `vec`.
 */
int sptAppendValueVectorWithVector(sptValueVector *vec, const sptValueVector *append_vec) {
    sptNnzIndex newlen = vec->len + append_vec->len;
    if(vec->cap <= newlen) {
        sptNnzIndex newcap = vec->cap + append_vec->cap;
        sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        spt_CheckOSError(!newdata, "ValVec Append ValVec");
        vec->cap = newcap;
        vec->data = newdata;
    }
    for(sptNnzIndex i=0; i<append_vec->len; ++i) {
        vec->data[vec->len + i] = append_vec->data[i];
    }
    vec->len = newlen;

    return 0;
}

/**
 * Resize a value vector
 *
 * @param vec  the value vector to resize
 * @param size the new size of the value vector
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size) {
    sptNnzIndex newcap = size < 2 ? 2 : size;
    if(newcap != vec->cap) {
        sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        spt_CheckOSError(!newdata, "ValVec Resize");
        vec->len = size;
        vec->cap = newcap;
        vec->data = newdata;
    } else {
        vec->len = size;
    }
    return 0;
}

/**
 * Release the memory buffer a value vector is holding
 *
 * @param vec a pointer to a valid value vector
 *
 */
void sptFreeValueVector(sptValueVector *vec) {
    vec->len = 0;
    vec->cap = 0;
    free(vec->data);
}


/*
 * Initialize a new sptIndex vector
 *
 * @param vec a valid pointer to an uninitialized sptIndex variable,
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */

int sptNewIndexVector(sptIndexVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    if(cap < len) {
        cap = len;
    }
    if(cap < 2) {
        cap = 2;
    }
    vec->len = len;
    vec->cap = cap;
    vec->data = malloc(cap * sizeof *vec->data);
    spt_CheckOSError(!vec->data, "IdxVec New");
    memset(vec->data, 0, cap * sizeof *vec->data);
    return 0;
}

/**
 * Fill an existed dense index vector with a specified constant
 *
 * @param vec   a valid pointer to an existed sptIndexVector variable,
 * @param num   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptConstantIndexVector(sptIndexVector * const vec, sptIndex const num) {
    for(sptNnzIndex i=0; i<vec->len; ++i)
        vec->data[i] = num;
    return 0;
}

/**
 * Copy an index vector to an uninitialized index vector
 *
 * @param dest a pointer to an uninitialized index vector
 * @param src  a pointer to an existing valid index vector
 *
 * The contents of `src` will be copied to `dest`.
 */
int sptCopyIndexVector(sptIndexVector *dest, const sptIndexVector *src) {
    int result = sptNewIndexVector(dest, src->len, src->len);
    spt_CheckError(result, "IdxVec Copy", NULL);
    memcpy(dest->data, src->data, src->len * sizeof *src->data);
    return 0;
}


/**
 * Copy an index vector to an uninitialized index vector
 *
 * @param dest a pointer to an uninitialized index vector
 * @param src  a pointer to an existing valid index vector
 *
 * The contents of `src` will be copied to `dest`.
 */
int sptCopyIndexVectorOmp(sptIndexVector *dest, const sptIndexVector *src, int const nt) {
    int result = sptNewIndexVector(dest, src->len, src->len);
    spt_CheckError(result, "IdxVec Copy", NULL);
    #pragma omp parallel for num_threads(nt)
    for (sptNnzIndex i=0; i<src->len; ++i) {
        dest->data[i] = src->data[i];
    }
    return 0;
}

/**
 * Add a value to the end of a sptIndexVector
 *
 * @param vec   a pointer to a valid index vector
 * @param value the value to be appended
 *
 * The length of the size vector will be changed to contain the new value.
 */
int sptAppendIndexVector(sptIndexVector *vec, sptIndex const value) {
    if(vec->cap <= vec->len) {
#ifndef MEMCHECK_MODE
        sptNnzIndex newcap = vec->cap + vec->cap/2;
#else
        sptNnzIndex newcap = vec->len+1;
#endif
        sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        spt_CheckOSError(!newdata, "IdxVec Append");
        vec->cap = newcap;
        vec->data = newdata;
    }
    vec->data[vec->len] = value;
    ++vec->len;
    return 0;
}

/**
 * Add a value to the end of an index vector
 *
 * @param vec        a pointer to a valid index vector
 * @param append_vec a pointer to another index vector, containing the values to be appended
 *
 * The values from `append_vec` will be appended to `vec`.
 */
int sptAppendIndexVectorWithVector(sptIndexVector *vec, const sptIndexVector *append_vec) {
    sptNnzIndex newlen = vec->len + append_vec->len;
    if(vec->cap <= newlen) {
        sptNnzIndex newcap = vec->cap + append_vec->cap;
        sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        spt_CheckOSError(!newdata, "IdxVec Append IdxVec");
        vec->cap = newcap;
        vec->data = newdata;
    }
    for(sptNnzIndex i=0; i<append_vec->len; ++i) {
        vec->data[vec->len + i] = append_vec->data[i];
    }
    vec->len = newlen;

    return 0;
}

/**
 * Resize an index vector
 *
 * @param vec  the index vector to resize
 * @param size the new size of the index vector
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size) {
    sptNnzIndex newcap = size < 2 ? 2 : size;
    if(newcap != vec->cap) {
        sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        spt_CheckOSError(!newdata, "IdxVec Resize");
        vec->len = size;
        vec->cap = newcap;
        vec->data = newdata;
    } else {
        vec->len = size;
    }
    return 0;
}

/**
 * Release the memory buffer a sptIndexVector is holding
 *
 * @param vec a pointer to a valid size vector
 *
 */
void sptFreeIndexVector(sptIndexVector *vec) {
    free(vec->data);
    vec->len = 0;
    vec->cap = 0;
}


/*
 * Initialize a new sptElementIndexVector vector
 *
 * @param vec a valid pointer to an uninitialized sptElementIndex variable,
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */

int sptNewElementIndexVector(sptElementIndexVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    if(cap < len) {
        cap = len;
    }
    if(cap < 2) {
        cap = 2;
    }
    vec->len = len;
    vec->cap = cap;
    vec->data = malloc(cap * sizeof *vec->data);
    spt_CheckOSError(!vec->data, "EleIdxVec New");
    memset(vec->data, 0, cap * sizeof *vec->data);
    return 0;
}

/**
 * Fill an existed dense element index vector with a specified constant
 *
 * @param vec   a valid pointer to an existed sptElementIndexVector variable,
 * @param num   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptConstantElementIndexVector(sptElementIndexVector * const vec, sptElementIndex const num) {
    for(sptNnzIndex i=0; i<vec->len; ++i)
        vec->data[i] = num;
    return 0;
}

/**
 * Copy an element index vector to an uninitialized element index vector
 *
 * @param dest a pointer to an uninitialized element index vector
 * @param src  a pointer to an existing valid element index vector
 *
 * The contents of `src` will be copied to `dest`.
 */
int sptCopyElementIndexVector(sptElementIndexVector *dest, const sptElementIndexVector *src) {
    int result = sptNewElementIndexVector(dest, src->len, src->len);
    spt_CheckError(result, "EleIdxVec Copy", NULL);
    memcpy(dest->data, src->data, src->len * sizeof *src->data);
    return 0;
}

/**
 * Add a value to the end of a sptElementIndexVector
 *
 * @param vec   a pointer to a valid element index vector
 * @param value the value to be appended
 *
 * The length of the element index vector will be changed to contain the new value.
 */
int sptAppendElementIndexVector(sptElementIndexVector *vec, sptElementIndex const value) {
    if(vec->cap <= vec->len) {
#ifndef MEMCHECK_MODE
        sptNnzIndex newcap = vec->cap + vec->cap/2;
#else
        sptNnzIndex newcap = vec->len+1;
#endif
        sptElementIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        spt_CheckOSError(!newdata, "EleIdxVec Append");
        vec->cap = newcap;
        vec->data = newdata;
    }
    vec->data[vec->len] = value;
    ++vec->len;
    return 0;
}

/**
 * Add a value to the end of an element index vector
 *
 * @param vec        a pointer to a valid element index vector
 * @param append_vec a pointer to another element index vector, containing the values to be appended
 *
 * The values from `append_vec` will be appended to `vec`.
 */
int sptAppendElementIndexVectorWithVector(sptElementIndexVector *vec, const sptElementIndexVector *append_vec) {
    sptNnzIndex newlen = vec->len + append_vec->len;
    if(vec->cap <= newlen) {
        sptNnzIndex newcap = vec->cap + append_vec->cap;
        sptElementIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        spt_CheckOSError(!newdata, "EleIdxVec Append EleIdxVec");
        vec->cap = newcap;
        vec->data = newdata;
    }
    for(sptNnzIndex i=0; i<append_vec->len; ++i) {
        vec->data[vec->len + i] = append_vec->data[i];
    }
    vec->len = newlen;

    return 0;
}

/**
 * Resize a element index vector
 *
 * @param vec  the element index vector to resize
 * @param size the new size of the element index vector
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeElementIndexVector(sptElementIndexVector *vec, sptNnzIndex const size) {
    sptNnzIndex newcap = size < 2 ? 2 : size;
    if(newcap != vec->cap) {
        sptElementIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        spt_CheckOSError(!newdata, "EleIdxVec Resize");
        vec->len = size;
        vec->cap = newcap;
        vec->data = newdata;
    } else {
        vec->len = size;
    }
    return 0;
}

/**
 * Release the memory buffer a sptElementIndexVector is holding
 *
 * @param vec a pointer to a valid size vector
 *
 */
void sptFreeElementIndexVector(sptElementIndexVector *vec) {
    free(vec->data);
    vec->len = 0;
    vec->cap = 0;
}


/*
 * Initialize a new sptBlockIndexVector vector
 *
 * @param vec a valid pointer to an uninitialized sptBlockIndex variable,
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */

int sptNewBlockIndexVector(sptBlockIndexVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    if(cap < len) {
        cap = len;
    }
    if(cap < 2) {
        cap = 2;
    }
    vec->len = len;
    vec->cap = cap;
    vec->data = malloc(cap * sizeof *vec->data);
    spt_CheckOSError(!vec->data, "BlkIdxVec New");
    memset(vec->data, 0, cap * sizeof *vec->data);
    return 0;
}

/**
 * Fill an existed dense element index vector with a specified constant
 *
 * @param vec   a valid pointer to an existed sptBlockIndexVector variable,
 * @param num   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptConstantBlockIndexVector(sptBlockIndexVector * const vec, sptBlockIndex const num) {
    for(sptNnzIndex i=0; i<vec->len; ++i)
        vec->data[i] = num;
    return 0;
}

/**
 * Copy a block index vector to an uninitialized block index vector
 *
 * @param dest a pointer to an uninitialized block index vector
 * @param src  a pointer to an existing valid block index vector
 *
 * The contents of `src` will be copied to `dest`.
 */
int sptCopyBlockIndexVector(sptBlockIndexVector *dest, const sptBlockIndexVector *src) {
    int result = sptNewBlockIndexVector(dest, src->len, src->len);
    spt_CheckError(result, "BlkIdxVec Copy", NULL);
    memcpy(dest->data, src->data, src->len * sizeof *src->data);
    return 0;
}

/**
 * Add a value to the end of a sptBlockIndexVector
 *
 * @param vec   a pointer to a valid block index vector
 * @param value the value to be appended
 *
 * The length of the block index vector will be changed to contain the new value.
 */
int sptAppendBlockIndexVector(sptBlockIndexVector *vec, sptBlockIndex const value) {
    if(vec->cap <= vec->len) {
#ifndef MEMCHECK_MODE
        sptNnzIndex newcap = vec->cap + vec->cap/2;
#else
        sptNnzIndex newcap = vec->len+1;
#endif
        sptBlockIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        spt_CheckOSError(!newdata, "BlkIdxVec Append");
        vec->cap = newcap;
        vec->data = newdata;
    }
    vec->data[vec->len] = value;
    ++vec->len;
    return 0;
}

/**
 * Add a value to the end of a block index vector
 *
 * @param vec        a pointer to a valid block index vector
 * @param append_vec a pointer to another block index vector, containing the values to be appended
 *
 * The values from `append_vec` will be appended to `vec`.
 */
int sptAppendBlockIndexVectorWithVector(sptBlockIndexVector *vec, const sptBlockIndexVector *append_vec) {
    sptNnzIndex newlen = vec->len + append_vec->len;
    if(vec->cap <= newlen) {
        sptNnzIndex newcap = vec->cap + append_vec->cap;
        sptBlockIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        spt_CheckOSError(!newdata, "BlkIdxVec Append BlkIdxVec");
        vec->cap = newcap;
        vec->data = newdata;
    }
    for(sptNnzIndex i=0; i<append_vec->len; ++i) {
        vec->data[vec->len + i] = append_vec->data[i];
    }
    vec->len = newlen;

    return 0;
}

/**
 * Resize a block index vector
 *
 * @param vec  the block index vector to resize
 * @param size the new size of the block index vector
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeBlockIndexVector(sptBlockIndexVector *vec, sptNnzIndex const size) {
    sptNnzIndex newcap = size < 2 ? 2 : size;
    if(newcap != vec->cap) {
        sptBlockIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        spt_CheckOSError(!newdata, "BlkIdxVec Resize");
        vec->len = size;
        vec->cap = newcap;
        vec->data = newdata;
    } else {
        vec->len = size;
    }
    return 0;
}

/**
 * Release the memory buffer a sptBlockIndexVector is holding
 *
 * @param vec a pointer to a valid size vector
 *
 */
void sptFreeBlockIndexVector(sptBlockIndexVector *vec) {
    free(vec->data);
    vec->len = 0;
    vec->cap = 0;
}


/*
 * Initialize a new sptNnzIndexVector vector
 *
 * @param vec a valid pointer to an uninitialized sptNnzIndex variable,
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */

int sptNewNnzIndexVector(sptNnzIndexVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    if(cap < len) {
        cap = len;
    }
    if(cap < 2) {
        cap = 2;
    }
    vec->len = len;
    vec->cap = cap;
    vec->data = malloc(cap * sizeof *vec->data);
    spt_CheckOSError(!vec->data, "NnzIdxVec New");
    memset(vec->data, 0, cap * sizeof *vec->data);
    return 0;
}

/**
 * Fill an existed dense long nnz index vector with a specified constant
 *
 * @param vec   a valid pointer to an existed sptNnzIndexVector variable,
 * @param num   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptConstantNnzIndexVector(sptNnzIndexVector * const vec, sptNnzIndex const num) {
    for(sptNnzIndex i=0; i<vec->len; ++i)
        vec->data[i] = num;
    return 0;
}

/**
 * Copy a long nnz index vector to an uninitialized long nnz index vector
 *
 * @param dest a pointer to an uninitialized long nnz index vector
 * @param src  a pointer to an existing valid long nnz index vector
 *
 * The contents of `src` will be copied to `dest`.
 */
int sptCopyNnzIndexVector(sptNnzIndexVector *dest, const sptNnzIndexVector *src) {
    int result = sptNewNnzIndexVector(dest, src->len, src->len);
    spt_CheckError(result, "NnzIdxVec Copy", NULL);
    memcpy(dest->data, src->data, src->len * sizeof *src->data);
    return 0;
}

/**
 * Add a value to the end of a sptNnzIndexVector
 *
 * @param vec   a pointer to a valid long nnz index vector
 * @param value the value to be appended
 *
 * The length of the long nnz index vector will be changed to contain the new value.
 */
int sptAppendNnzIndexVector(sptNnzIndexVector *vec, sptNnzIndex const value) {
    if(vec->cap <= vec->len) {
#ifndef MEMCHECK_MODE
        sptNnzIndex newcap = vec->cap + vec->cap/2;
#else
        sptNnzIndex newcap = vec->len+1;
#endif
        sptNnzIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        spt_CheckOSError(!newdata, "NnzIdxVec Append");
        vec->cap = newcap;
        vec->data = newdata;
    }
    vec->data[vec->len] = value;
    ++vec->len;
    return 0;
}

/**
 * Add a value to the end of a long nnz index vector
 *
 * @param vec        a pointer to a valid long nnz index vector
 * @param append_vec a pointer to another long nnz index vector, containing the values to be appended
 *
 * The values from `append_vec` will be appended to `vec`.
 */
int sptAppendNnzIndexVectorWithVector(sptNnzIndexVector *vec, const sptNnzIndexVector *append_vec) {
    sptNnzIndex newlen = vec->len + append_vec->len;
    if(vec->cap <= newlen) {
        sptNnzIndex newcap = vec->cap + append_vec->cap;
        sptNnzIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        spt_CheckOSError(!newdata, "NnzIdxVec Append NnzIdxVec");
        vec->cap = newcap;
        vec->data = newdata;
    }
    for(sptNnzIndex i=0; i<append_vec->len; ++i) {
        vec->data[vec->len + i] = append_vec->data[i];
    }
    vec->len = newlen;

    return 0;
}

/**
 * Resize a long nnz index vector
 *
 * @param vec  the long nnz index vector to resize
 * @param size the new size of the long nnz index vector
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeNnzIndexVector(sptNnzIndexVector *vec, sptNnzIndex const size) {
    sptNnzIndex newcap = size < 2 ? 2 : size;
    if(newcap != vec->cap) {
        sptNnzIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        spt_CheckOSError(!newdata, "NnzIdxVec Resize");
        vec->len = size;
        vec->cap = newcap;
        vec->data = newdata;
    } else {
        vec->len = size;
    }
    return 0;
}

/**
 * Release the memory buffer a sptNnzIndexVector is holding
 *
 * @param vec a pointer to a valid long nnz vector
 *
 */
void sptFreeNnzIndexVector(sptNnzIndexVector *vec) {
    free(vec->data);
    vec->len = 0;
    vec->cap = 0;
}

