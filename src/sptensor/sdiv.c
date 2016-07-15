#include <SpTOL.h>

int sptSparseTensorDivScalar(sptSparseTensor *X, sptScalar a) {
	if(a != 0) {
	    size_t i;
	    #pragma omp parallel for schedule(static)
	    for(i = 0; i < X->nnz; ++i) {
	        X->values.data[i] /= a;
	    }
	    return 0;
	} else {
		fprintf(stderr, "SpTOL ERROR: dividing zero.\n");
		return -1;
	}
    
}
