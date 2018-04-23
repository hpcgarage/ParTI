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

#ifndef PARTI_TYPES_H
#define PARTI_TYPES_H

#include <stdint.h>

/**
 * Define types, TODO: check the bit size of them, add branch for different settings
 */
#define PARTI_INDEX_TYPEWIDTH 32
#define PARTI_VALUE_TYPEWIDTH 32
#define PARTI_ELEMENT_INDEX_TYPEWIDTH 8

#if PARTI_INDEX_TYPEWIDTH == 32
  typedef uint32_t sptIndex;
  typedef uint32_t sptBlockIndex;
  #define PARTI_INDEX_MAX UINT32_MAX
  #define PARTI_PRI_INDEX PRIu32
  #define PARTI_SCN_INDEX SCNu32
  #define PARTI_PRI_BLOCK_INDEX PRIu32
  #define PARTI_SCN_BLOCK_INDEX SCNu32
#elif PARTI_INDEX_TYPEWIDTH == 64
  typedef uint64_t sptIndex;
  typedef uint64_t sptBlockIndex;
  #define PARTI_INDEX_MAX UINT64_MAX
  #define PARTI_PFI_INDEX PRIu64
  #define PARTI_SCN_INDEX SCNu64
  #define PARTI_PRI_BLOCK_INDEX PRIu64
  #define PARTI_SCN_BLOCK_INDEX SCNu64
#else
  #error "Unrecognized PARTI_INDEX_TYPEWIDTH."
#endif

#if PARTI_VALUE_TYPEWIDTH == 32
  typedef float sptValue;
  #define PARTI_PRI_VALUE "f"
  #define PARTI_SCN_VALUE "f"
#elif PARTI_VALUE_TYPEWIDTH == 64
  typedef double sptValue;
  #define PARTI_PRI_VALUE "lf"
  #define PARTI_SCN_VALUE "lf"
#else
  #error "Unrecognized PARTI_VALUE_TYPEWIDTH."
#endif

#if PARTI_ELEMENT_INDEX_TYPEWIDTH == 8
  // typedef uint_fast8_t sptElementIndex;
  // typedef uint_fast16_t sptBlockMatrixIndex;  // R < 256
  // #define PARTI_PRI_ELEMENT_INDEX PRIuFAST8
  // #define PARTI_SCN_ELEMENT_INDEX SCNuFAST8
  // #define PARTI_PRI_BLOCKMATRIX_INDEX PRIuFAST16
  // #define PARTI_SCN_BLOCKMATRIX_INDEX SCNuFAST16
  typedef uint8_t sptElementIndex;
  typedef uint16_t sptBlockMatrixIndex;  // R < 256
  #define PARTI_PRI_ELEMENT_INDEX PRIu8
  #define PARTI_SCN_ELEMENT_INDEX SCNu8
  #define PARTI_PRI_BLOCKMATRIX_INDEX PRIu16
  #define PARTI_SCN_BLOCKMATRIX_INDEX SCNu16
#elif PARTI_ELEMENT_INDEX_TYPEWIDTH == 16
  typedef uint16_t sptElementIndex;
  typedef uint32_t sptBlockMatrixIndex;
  #define PARTI_PFI_ELEMENT_INDEX PRIu16
  #define PARTI_SCN_ELEMENT_INDEX SCNu16
  #define PARTI_PRI_BLOCKMATRIX_INDEX PRIu32
  #define PARTI_SCN_BLOCKMATRIX_INDEX SCNu32
#elif PARTI_ELEMENT_INDEX_TYPEWIDTH == 32
  typedef uint32_t sptElementIndex;
  typedef uint32_t sptBlockMatrixIndex;
  #define PARTI_PFI_ELEMENT_INDEX PRIu32
  #define PARTI_SCN_ELEMENT_INDEX SCNu32
  #define PARTI_PRI_BLOCKMATRIX_INDEX PRIu32
  #define PARTI_SCN_BLOCKMATRIX_INDEX SCNu32
#else
  #error "Unrecognized PARTI_ELEMENT_INDEX_TYPEWIDTH."
#endif

typedef sptBlockIndex sptBlockNnzIndex;
#define PARTI_PRI_BLOCKNNZ_INDEX PARTI_PRI_BLOCK_INDEX
#define PARTI_SCN_BLOCKNNZ_INDEX PARTI_SCN_BLOCK_INDEX

typedef uint64_t sptNnzIndex;
#define PARTI_PRI_NNZ_INDEX PRIu64
#define PARTI_SCN_NNZ_INDEX PRIu64

typedef unsigned __int128 sptMortonIndex;
// typedef __uint128_t sptMortonIndex;


#endif