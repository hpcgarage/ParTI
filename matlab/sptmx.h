/*
    This file is part of SpTOL.

    SpTOL is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    SpTOL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with SpTOL.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdlib.h>
#include "matrix.h"

#define spt_mxCheckArgs(funcname, lnum, ltxt, rnum, rtxt)                   \
if(nrhs != rnum) {                                                          \
    if(rnum != 1) {                                                         \
        mexErrMsgIdAndTxt("SpTOL:" #funcname, #rtxt "inputs required.");    \
    } else {                                                                \
        mexErrMsgIdAndTxt("SpTOL:" #funcname, #rtxt "input required.");     \
    }                                                                       \
}                                                                           \
if(nlhs != lnum) {                                                          \
    if(lnum != 1) {                                                         \
        mexErrMsgIdAndTxt("SpTOL:" #funcname, #ltxt "outputs required.");   \
    } else {                                                                \
        mexErrMsgIdAndTxt("SpTOL:" #funcname, #ltxt "output required.");    \
    }                                                                       \
}

#define spt_DefineCastArray(funcname, T)                                    \
T *funcname(const mxArray *pm) {                                            \
    mxClassID clsid = mxGetClassID(pm);                                     \
    void *data = mxGetData(pm);                                             \
    size_t n = mxGetNumberOfElements(pm);                                   \
    size_t i;                                                               \
    T *result = malloc(n * sizeof (T));                                     \
    switch(clsid) {                                                         \
    case mxLOGICAL_CLASS:                                                   \
        for(i = 0; i < n; ++i) {                                            \
            result[i] = (T) ((mxLogical *) data)[i];                        \
        }                                                                   \
        break;                                                              \
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
    case mxCHAR_CLASS:                                                      \
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
    case mxCHAR_CLASS:                                                      \
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

static inline void *spt_mxGetPointer(const mxArray *pa, mwIndex idx) {
    mxArray *pm = mxGetProperty(pa, idx, "ptr");
    if(mxGetClassID(pm) != mxUINT64_CLASS || mxGetNumberOfElements(pm) < 1) {
        return NULL;
    }
    size_t *pptr = mxGetData(pm);
    void *ptr = (void *) pptr[0];
    mxDestroyArray(pm);

    return ptr;
}

static inline void spt_mxSetPointer(mxArray *pa, mwIndex idx, void *ptr) {
    mxArray *mxptr = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    size_t *pptr = mxGetData(mxptr);
    pptr[0] = (size_t) ptr;
    mxSetProperty(pa, idx, "ptr", mxptr);
    mxDestroyArray(mxptr);
}
