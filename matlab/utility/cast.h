#include "matrix.h"

#define DefineCastArray(funcname, T)                            \
T *funcname(const mxArray *pm) {                                \
    mxClassID clsid = mxGetClassID(pm);                         \
    void *data = mxGetData(pm);                                 \
    size_t n = mxGetNumberOfElements(pm);                       \
    size_t i;                                                   \
    T *result = malloc(n * sizeof (T));                         \
    switch(clsid) {                                             \
    case mxINT8_CLASS:                                          \
        for(i = 0; i < n; ++i) {                                \
            result[i] = (T) ((signed char *) data)[i];          \
        }                                                       \
        break;                                                  \
    case mxUINT8_CLASS:                                         \
        for(i = 0; i < n; ++i) {                                \
            result[i] = (T) ((unsigned char *) data)[i];        \
        }                                                       \
        break;                                                  \
    case mxINT16_CLASS:                                         \
        for(i = 0; i < n; ++i) {                                \
            result[i] = (T) ((short *) data)[i];                \
        }                                                       \
        break;                                                  \
    case mxUINT16_CLASS:                                        \
        for(i = 0; i < n; ++i) {                                \
            result[i] = (T) ((unsigned short *) data)[i];       \
        }                                                       \
        break;                                                  \
    case mxINT32_CLASS:                                         \
        for(i = 0; i < n; ++i) {                                \
            result[i] = (T) ((int *) data)[i];                  \
        }                                                       \
        break;                                                  \
    case mxUINT32_CLASS:                                        \
        for(i = 0; i < n; ++i) {                                \
            result[i] = (T) ((unsigned int *) data)[i];         \
        }                                                       \
        break;                                                  \
    case mxINT64_CLASS:                                         \
        for(i = 0; i < n; ++i) {                                \
            result[i] = (T) ((long long *) data)[i];            \
        }                                                       \
        break;                                                  \
    case mxUINT64_CLASS:                                        \
        for(i = 0; i < n; ++i) {                                \
            result[i] = (T) ((unsigned long long *) data)[i];   \
        }                                                       \
        break;                                                  \
    case mxSINGLE_CLASS:                                        \
        for(i = 0; i < n; ++i) {                                \
            result[i] = (T) ((float *) data)[i];                \
        }                                                       \
        break;                                                  \
    case mxDOUBLE_CLASS:                                        \
        for(i = 0; i < n; ++i) {                                \
            result[i] = (T) ((double *) data)[i];               \
        }                                                       \
        break;                                                  \
    default:                                                    \
        free(result);                                           \
        return NULL;                                            \
    }                                                           \
    return result;                                              \
}

