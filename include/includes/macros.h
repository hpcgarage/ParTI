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

#ifndef PARTI_MACROS_H
#define PARTI_MACROS_H

/* User defined system configuration */
#define NUM_CORES 56    // for L size and determine privatilization or not
#define L1_SIZE 32000    // for B size
#define LLC_SIZE 40960000
#define PAR_DEGREE_REDUCE 20     // for determine privatilization or not  
#define PAR_MAX_DEGREE 100 // for L size
#define PAR_MIN_DEGREE 4 // for L size

#ifndef PARTI_DEFAULT_NLOCKS
#define PARTI_DEFAULT_NLOCKS 1024
#endif

#ifndef PARTI_DEFAULT_LOCK_PAD_SIZE
#define PARTI_DEFAULT_LOCK_PAD_SIZE 16
#endif

/**
 * An opaque data type to store a specific time point, using either CPU or GPU clock.
 */
typedef struct sptTagTimer *sptTimer;

typedef enum {
    SPTERR_NO_ERROR       = 0,
    SPTERR_UNKNOWN        = 1,
    SPTERR_SHAPE_MISMATCH = 2,
    SPTERR_VALUE_ERROR    = 3,
    SPTERR_ZERO_DIVISION  = 4,
    SPTERR_NO_MORE        = 99,
    SPTERR_OS_ERROR       = 0x10000,
    SPTERR_CUDA_ERROR     = 0x20000,
} SptError;

#endif
