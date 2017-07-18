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
 * Dump a dense value vector to file
 *
 * @param vec a pointer to a valid value vector
 * @param fp a file pointer
 *
 */
int sptDumpVector(sptVector *vec, FILE *fp) {
    int iores;
    size_t len = vec->len;
    iores = fprintf(fp, "Value vector length: %zu\n", len);
    spt_CheckOSError(iores < 0, "Vec Dump");
    for(size_t i=0; i < len; ++i) {
        iores = fprintf(fp, "%.2lf\t", vec->data[i]);
        spt_CheckOSError(iores < 0, "Vec Dump");
    }
    iores = fprintf(fp, "\n");

    return 0;
}


/**
 * Dump a dense size vector to file
 *
 * @param vec a pointer to a valid size vector
 * @param fp a file pointer
 *
 */
int sptDumpSizeVector(sptSizeVector *vec, FILE *fp) {
    int iores;
    size_t len = vec->len;
    iores = fprintf(fp, "Size vector length: %zu\n", len);
    spt_CheckOSError(iores < 0, "SzVec Dump");
    for(size_t i=0; i < len; ++i) {
        iores = fprintf(fp, "%zu\t", vec->data[i]);
        spt_CheckOSError(iores < 0, "SzVec Dump");
    }
    iores = fprintf(fp, "\n");

    return 0;
}


/**
 * Dump a dense sptElementIndexVector to file
 *
 * @param vec a pointer to a valid sptElementIndexVector
 * @param fp a file pointer
 *
 */
int sptDumpElementIndexVector(sptElementIndexVector *vec, FILE *fp) {
    int iores;
    uint64_t len = vec->len;
    iores = fprintf(fp, "sptElementIndexVector length: %lu\n", len);
    spt_CheckOSError(iores < 0, "EleIdxVec Dump");
    for(uint64_t i=0; i < len; ++i) {
        iores = fprintf(fp, "%u\t", vec->data[i]);
        spt_CheckOSError(iores < 0, "EleIdxVec Dump");
    }
    iores = fprintf(fp, "\n");

    return 0;
}


/**
 * Dump a dense sptBlockIndexVector to file
 *
 * @param vec a pointer to a valid sptBlockIndexVector
 * @param fp a file pointer
 *
 */
int sptDumpBlockIndexVector(sptBlockIndexVector *vec, FILE *fp) {
    int iores;
    uint64_t len = vec->len;
    iores = fprintf(fp, "sptBlockIndexVector length: %lu\n", len);
    spt_CheckOSError(iores < 0, "BlkIdxVec Dump");
    for(uint64_t i=0; i < len; ++i) {
        iores = fprintf(fp, "%u\t", vec->data[i]);
        spt_CheckOSError(iores < 0, "BlkIdxVec Dump");
    }
    iores = fprintf(fp, "\n");

    return 0;
}


/**
 * Dump a dense sptNnzIndexVector to file
 *
 * @param vec a pointer to a valid sptNnzIndexVector
 * @param fp a file pointer
 *
 */
int sptDumpNnzIndexVector(sptNnzIndexVector *vec, FILE *fp) {
    int iores;
    uint64_t len = vec->len;
    iores = fprintf(fp, "sptNnzIndexVector length: %lu\n", len);
    spt_CheckOSError(iores < 0, "NnzIdxVec Dump");
    for(uint64_t i=0; i < len; ++i) {
        iores = fprintf(fp, "%lu\t", vec->data[i]);
        spt_CheckOSError(iores < 0, "NnzIdxVec Dump");
    }
    iores = fprintf(fp, "\n");

    return 0;
}


/**
 * Dump a dense sptIndexVector to file
 *
 * @param vec a pointer to a valid sptIndexVector
 * @param fp a file pointer
 *
 */
int sptDumpIndexVector(sptIndexVector *vec, FILE *fp) {
    int iores;
    uint64_t len = vec->len;
    iores = fprintf(fp, "sptIndexVector length: %lu\n", len);
    spt_CheckOSError(iores < 0, "IdxVec Dump");
    for(uint64_t i=0; i < len; ++i) {
        iores = fprintf(fp, "%u\t", vec->data[i]);
        spt_CheckOSError(iores < 0, "IdxVec Dump");
    }
    iores = fprintf(fp, "\n");

    return 0;
}


/**
 * Dump a dense sptValueVector to file
 *
 * @param vec a pointer to a valid sptValueVector
 * @param fp a file pointer
 *
 */
int sptDumpValueIndexVector(sptValueVector *vec, FILE *fp) {
    int iores;
    uint64_t len = vec->len;
    iores = fprintf(fp, "sptValueVector length: %lu\n", len);
    spt_CheckOSError(iores < 0, "ValVec Dump");
    for(uint64_t i=0; i < len; ++i) {
        iores = fprintf(fp, "%f\t", vec->data[i]);
        spt_CheckOSError(iores < 0, "ValVec Dump");
    }
    iores = fprintf(fp, "\n");

    return 0;
}


/**
 * Dump a dense sptIndex array to file
 *
 * @param array a pointer to a valid sptIndex array
 * @param size of the array
 * @param fp a file pointer
 *
 */
int sptDumpIndexArray(sptIndex *array, uint64_t n, FILE *fp) {
    int iores;
    iores = fprintf(fp, "sptIndex array length: %lu\n", n);
    spt_CheckOSError(iores < 0, "IdxArray Dump");
    for(uint64_t i=0; i < n; ++i) {
        iores = fprintf(fp, "%u\t", array[i]);
        spt_CheckOSError(iores < 0, "IdxArray Dump");
    }
    iores = fprintf(fp, "\n");

    return 0;
}