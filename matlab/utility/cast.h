#include "matrix.h"

#define DefineCastArray(funcname, T)                    \
T *funcname(const mxArray *pm) {	                \
    mxClassID clsid = mxGetClassID(pm);	                \
    void *data = mxGetData(pm);	                        \
    size_t n = mxGetNumberOfElements(pm);	        \
    size_t i;	                                        \
    T *result = malloc(n * sizeof (T));	                \
    switch(clsid) {	                                \
    case mxINT8_CLASS:	                                \
        for(i = 0; i < n; ++i) {	                \
            result[i] = i[(signed char *) data];	\
        }	                                        \
        break;	                                        \
    case mxUINT8_CLASS:	                                \
        for(i = 0; i < n; ++i) {	                \
            result[i] = i[(unsigned char *) data];	\
        }	                                        \
        break;	                                        \
    case mxINT16_CLASS:	                                \
        for(i = 0; i < n; ++i) {	                \
            result[i] = i[(short *) data];	        \
        }	                                        \
        break;	                                        \
    case mxUINT16_CLASS:	                        \
        for(i = 0; i < n; ++i) {	                \
            result[i] = i[(unsigned short *) data];	\
        }	                                        \
        break;	                                        \
    case mxINT32_CLASS:	                                \
        for(i = 0; i < n; ++i) {	                \
            result[i] = i[(int *) data];	        \
        }	                                        \
        break;	                                        \
    case mxUINT32_CLASS:	                        \
        for(i = 0; i < n; ++i) {	                \
            result[i] = i[(unsigned int *) data];	\
        }	                                        \
        break;	                                        \
    case mxINT64_CLASS:	                                \
        for(i = 0; i < n; ++i) {	                \
            result[i] = i[(long long *) data];	        \
        }	                                        \
        break;	                                        \
    case mxUINT64_CLASS:	                        \
        for(i = 0; i < n; ++i) {	                \
            result[i] = i[(unsigned long long *) data];	\
        }	                                        \
        break;	                                        \
    case mxSINGLE_CLASS:	                        \
        for(i = 0; i < n; ++i) {	                \
            result[i] = i[(float *) data];	        \
        }	                                        \
        break;	                                        \
    case mxDOUBLE_CLASS:	                        \
        for(i = 0; i < n; ++i) {	                \
            result[i] = i[(double *) data];	        \
        }	                                        \
        break;	                                        \
    default:	                                        \
        free(result);	                                \
        return NULL;	                                \
    }	                                                \
    return result;                                      
}

