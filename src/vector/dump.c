#include <SpTOL.h>
#include <stdlib.h>
#include <string.h>
#include "../error/error.h"


/**
 * Dum a dense value vector to file
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
 * Dum a dense size vector to file
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