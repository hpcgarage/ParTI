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

#ifndef PARTI_VECTORS_H
#define PARTI_VECTORS_H


/* Dense Array functions */
sptNnzIndex sptMaxNnzIndexArray(sptNnzIndex const * const indices, sptNnzIndex const size);
sptIndex sptMaxIndexArray(sptIndex const * const indices, sptNnzIndex const size);
void sptPairArraySort(sptKeyValuePair const * kvarray, sptIndex const length);
int sptDumpIndexArray(sptIndex *array, sptNnzIndex const n, FILE *fp);
int sptDumpNnzIndexArray(sptNnzIndex *array, sptNnzIndex const n, FILE *fp);

/* Dense vector, with sptValueVector type */
int sptNewValueVector(sptValueVector *vec, uint64_t len, uint64_t cap);
int sptConstantValueVector(sptValueVector * const vec, sptValue const val);
int sptCopyValueVector(sptValueVector *dest, const sptValueVector *src);
int sptAppendValueVector(sptValueVector *vec, sptValue const value);
int sptAppendValueVectorWithVector(sptValueVector *vec, const sptValueVector *append_vec);
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size);
void sptFreeValueVector(sptValueVector *vec);
int sptDumpValueIndexVector(sptValueVector *vec, FILE *fp);

/* Dense vector, with sptIndexVector type */
int sptNewIndexVector(sptIndexVector *vec, uint64_t len, uint64_t cap);
int sptConstantIndexVector(sptIndexVector * const vec, sptIndex const num);
int sptCopyIndexVector(sptIndexVector *dest, const sptIndexVector *src);
int sptAppendIndexVector(sptIndexVector *vec, sptIndex const value);
int sptAppendIndexVectorWithVector(sptIndexVector *vec, const sptIndexVector *append_vec);
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size);
void sptFreeIndexVector(sptIndexVector *vec);
int sptDumpIndexVector(sptIndexVector *vec, FILE *fp);

/* Dense vector, with sptElementIndexVector type */
int sptNewElementIndexVector(sptElementIndexVector *vec, uint64_t len, uint64_t cap);
int sptConstantElementIndexVector(sptElementIndexVector * const vec, sptElementIndex const num);
int sptCopyElementIndexVector(sptElementIndexVector *dest, const sptElementIndexVector *src);
int sptAppendElementIndexVector(sptElementIndexVector *vec, sptElementIndex const value);
int sptAppendElementIndexVectorWithVector(sptElementIndexVector *vec, const sptElementIndexVector *append_vec);
int sptResizeElementIndexVector(sptElementIndexVector *vec, sptNnzIndex const size);
void sptFreeElementIndexVector(sptElementIndexVector *vec);
int sptDumpElementIndexVector(sptElementIndexVector *vec, FILE *fp);

/* Dense vector, with sptBlockIndexVector type */
int sptNewBlockIndexVector(sptBlockIndexVector *vec, uint64_t len, uint64_t cap);
int sptConstantBlockIndexVector(sptBlockIndexVector * const vec, sptBlockIndex const num);
int sptCopyBlockIndexVector(sptBlockIndexVector *dest, const sptBlockIndexVector *src);
int sptAppendBlockIndexVector(sptBlockIndexVector *vec, sptBlockIndex const value);
int sptAppendBlockIndexVectorWithVector(sptBlockIndexVector *vec, const sptBlockIndexVector *append_vec);
int sptResizeBlockIndexVector(sptBlockIndexVector *vec, sptNnzIndex const size);
void sptFreeBlockIndexVector(sptBlockIndexVector *vec);
int sptDumpBlockIndexVector(sptBlockIndexVector *vec, FILE *fp);

/* Dense vector, with sptNnzIndexVector type */
int sptNewNnzIndexVector(sptNnzIndexVector *vec, uint64_t len, uint64_t cap);
int sptConstantNnzIndexVector(sptNnzIndexVector * const vec, sptNnzIndex const num);
int sptCopyNnzIndexVector(sptNnzIndexVector *dest, const sptNnzIndexVector *src);
int sptAppendNnzIndexVector(sptNnzIndexVector *vec, sptNnzIndex const value);
int sptAppendNnzIndexVectorWithVector(sptNnzIndexVector *vec, const sptNnzIndexVector *append_vec);
int sptResizeNnzIndexVector(sptNnzIndexVector *vec, sptNnzIndex const size);
void sptFreeNnzIndexVector(sptNnzIndexVector *vec);
int sptDumpNnzIndexVector(sptNnzIndexVector *vec, FILE *fp);

#endif