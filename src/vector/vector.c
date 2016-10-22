#include <SpTOL.h>
#include <stdlib.h>
#include <string.h>

/**
 * Initialize a new value vector
 *
 * @param mtx   a valid pointer to an uninitialized sptMatrix variable,
 * @param nrows the number of rows
 * @param ncols the number of columns
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptNewVector(sptVector *vec, size_t len, size_t cap) {
    if(cap < len) {
        cap = len;
    }
    if(cap < 2) {
        cap = 2;
    }
    vec->len = len;
    vec->cap = cap;
    vec->data = malloc(cap * sizeof *vec->data);
    if(!vec->data) {
        return -1;
    }
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
int sptCopyVector(sptVector *dest, const sptVector *src) {
    int result = sptNewVector(dest, src->len, src->len);
    if(result) {
        return result;
    }
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
int sptAppendVector(sptVector *vec, sptScalar value) {
    if(vec->cap <= vec->len) {
#ifndef MEMCHECK_MODE
        size_t newcap = vec->cap + vec->cap/2;
#else
        size_t newcap = vec->len+1;
#endif
        sptScalar *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        if(!newdata) {
            return -1;
        }
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
int sptAppendVectorWithVector(sptVector *vec, const sptVector *append_vec) {
    if(vec->cap <= vec->len) {
        size_t newcap = vec->cap + append_vec->cap;
        sptScalar *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        if(!newdata) {
            return -1;
        }
        vec->cap = newcap;
        vec->data = newdata;
    }
    for(size_t i=0; i<append_vec->len; ++i) {
        vec->data[vec->len + i] = append_vec->data[i];
        ++vec->len;
    }

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
int sptResizeVector(sptVector *vec, size_t size) {
    if(size != vec->cap) {
        sptScalar *newdata = realloc(vec->data, size * sizeof *vec->data);
        if(!newdata) {
            return -1;
        }
        vec->len = size;
        vec->cap = size;
        vec->data = newdata;
    }
    return 0;
}

/**
 * Release the memory buffer a value vector is holding
 *
 * @param mtx a pointer to a valid value vector
 *
 * By using `sptFreeVector`, a valid value vector would become uninitialized
 * and should not be used anymore prior to another initialization
 */
void sptFreeVector(sptVector *vec) {
    free(vec->data);
}


/* Size vector functions. */
int sptNewSizeVector(sptSizeVector *vec, size_t len, size_t cap) {
    if(cap < len) {
        cap = len;
    }
    if(cap < 2) {
        cap = 2;
    }
    vec->len = len;
    vec->cap = cap;
    vec->data = malloc(cap * sizeof *vec->data);
    if(!vec->data) {
        return -1;
    }
    return 0;
}

int sptCopySizeVector(sptSizeVector *dest, const sptSizeVector *src) {
    int result = sptNewSizeVector(dest, src->len, src->len);
    if(result) {
        return result;
    }
    memcpy(dest->data, src->data, src->len * sizeof *src->data);
    return 0;
}

int sptAppendSizeVector(sptSizeVector *vec, size_t value) {
    if(vec->cap <= vec->len) {
#ifndef MEMCHECK_MODE
        size_t newcap = vec->cap + vec->cap/2;
#else
        size_t newcap = vec->len+1;
#endif
        size_t *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        if(!newdata) {
            return -1;
        }
        vec->cap = newcap;
        vec->data = newdata;
    }
    vec->data[vec->len] = value;
    ++vec->len;
    return 0;
}

int sptAppendSizeVectorWithVector(sptSizeVector *vec, const sptSizeVector *append_vec) {
    if(vec->cap <= vec->len) {
        size_t newcap = vec->cap + append_vec->cap;
        size_t *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        if(!newdata) {
            return -1;
        }
        vec->cap = newcap;
        vec->data = newdata;
    }
    for(size_t i=0; i<append_vec->len; ++i) {
        vec->data[vec->len + i] = append_vec->data[i];
        ++vec->len;
    }

    return 0;
}

int sptResizeSizeVector(sptSizeVector *vec, size_t size) {
    if(size != vec->cap) {
        size_t *newdata = realloc(vec->data, size * sizeof *vec->data);
        if(!newdata) {
            return -1;
        }
        vec->len = size;
        vec->cap = size;
        vec->data = newdata;
    }
    return 0;
}

void sptFreeSizeVector(sptSizeVector *vec) {
    free(vec->data);
}
