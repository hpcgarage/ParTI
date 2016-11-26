#include <stdlib.h>
#include "matrix.h"

#define spt_DefineCastArray(funcname, T)                                    \
T *funcname(const mxArray *pm) {                                            \
    mxClassID clsid = mxGetClassID(pm);                                     \
    void *data = mxGetData(pm);                                             \
    size_t n = mxGetNumberOfElements(pm);                                   \
    size_t i;                                                               \
    T *result = malloc(n * sizeof (T));                                     \
    switch(clsid) {                                                         \
    case mxINT8_CLASS:                                                      \
        for(i = 0; i < n; ++i) {                                            \
            result[i] = (T) ((signed char *) data)[i];                      \
        }                                                                   \
        break;                                                              \
    case mxUINT8_CLASS:                                                     \
        for(i = 0; i < n; ++i) {                                            \
            result[i] = (T) ((unsigned char *) data)[i];                    \
        }                                                                   \
        break;                                                              \
    case mxINT16_CLASS:                                                     \
        for(i = 0; i < n; ++i) {                                            \
            result[i] = (T) ((short *) data)[i];                            \
        }                                                                   \
        break;                                                              \
    case mxUINT16_CLASS:                                                    \
        for(i = 0; i < n; ++i) {                                            \
            result[i] = (T) ((unsigned short *) data)[i];                   \
        }                                                                   \
        break;                                                              \
    case mxINT32_CLASS:                                                     \
        for(i = 0; i < n; ++i) {                                            \
            result[i] = (T) ((int *) data)[i];                              \
        }                                                                   \
        break;                                                              \
    case mxUINT32_CLASS:                                                    \
        for(i = 0; i < n; ++i) {                                            \
            result[i] = (T) ((unsigned int *) data)[i];                     \
        }                                                                   \
        break;                                                              \
    case mxINT64_CLASS:                                                     \
        for(i = 0; i < n; ++i) {                                            \
            result[i] = (T) ((long long *) data)[i];                        \
        }                                                                   \
        break;                                                              \
    case mxUINT64_CLASS:                                                    \
        for(i = 0; i < n; ++i) {                                            \
            result[i] = (T) ((unsigned long long *) data)[i];               \
        }                                                                   \
        break;                                                              \
    case mxSINGLE_CLASS:                                                    \
        for(i = 0; i < n; ++i) {                                            \
            result[i] = (T) ((float *) data)[i];                            \
        }                                                                   \
        break;                                                              \
    case mxDOUBLE_CLASS:                                                    \
        for(i = 0; i < n; ++i) {                                            \
            result[i] = (T) ((double *) data)[i];                           \
        }                                                                   \
        break;                                                              \
    default:                                                                \
        free(result);                                                       \
        return NULL;                                                        \
    }                                                                       \
    return result;                                                          \
}

#define spt_DefineSetScalar(funcname, T)                                    \
static int funcname(const mxArray *pm, size_t idx, T value) {               \
    if(idx >= mxGetNumberOfElements(pm)) {                                  \
        return -1;                                                          \
    }                                                                       \
    void *data = mxGetData(pm);                                             \
    if(!data) {                                                             \
        return -1;                                                          \
    }                                                                       \
    mxClassID clsid = mxGetClassID(pm);                                     \
    switch(clsid) {                                                         \
    case mxINT8_CLASS:                                                      \
        ((signed char *) data)[idx] = (signed char) value;                  \
        break;                                                              \
    case mxUINT8_CLASS:                                                     \
        ((unsigned char *) data)[idx] = (unsigned char) value;              \
        break;                                                              \
    case mxINT16_CLASS:                                                     \
        ((short *) data)[idx] = (short) value;                              \
        break;                                                              \
    case mxUINT16_CLASS:                                                    \
        ((unsigned short *) data)[idx] = (unsigned short) value;            \
        break;                                                              \
    case mxINT32_CLASS:                                                     \
        ((int *) data)[idx] = (int) value;                                  \
        break;                                                              \
    case mxUINT32_CLASS:                                                    \
        ((unsigned int *) data)[idx] = (unsigned int) value;                \
        break;                                                              \
    case mxINT64_CLASS:                                                     \
        ((long long *) data)[idx] = (long long) value;                      \
        break;                                                              \
    case mxUINT64_CLASS:                                                    \
        ((unsigned long long *) data)[idx] = (unsigned long long) value;    \
        break;                                                              \
    case mxSINGLE_CLASS:                                                    \
        ((float *) data)[idx] = (float) value;                              \
        break;                                                              \
    case mxDOUBLE_CLASS:                                                    \
        ((double *) data)[idx] = (double) value;                            \
        break;                                                              \
    default:                                                                \
        return -1;                                                          \
    }                                                                       \
    return 0;                                                               \
}

