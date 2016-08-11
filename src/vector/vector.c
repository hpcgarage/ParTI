#include <SpTOL.h>
#include <stdlib.h>
#include <string.h>

/* Value vector functions. */
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

int sptCopyVector(sptVector *dest, const sptVector *src) {
    int result = sptNewVector(dest, src->len, src->len);
    if(result) {
        return result;
    }
    memcpy(dest->data, src->data, src->len * sizeof *src->data);
    return 0;
}

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

int sptAppendVectorWithVector(sptVector *vec, sptVector *append_vec) {
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

int sptAppendSizeVectorWithVector(sptSizeVector *vec, sptSizeVector *append_vec) {
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
