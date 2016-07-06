#include <SpTOL.h>
#include "sptensor.h"
#include <assert.h>

static int spt_CompareIndices(const sptSparseTensor *tsr1, size_t ind1, const sptSparseTensor *tsr2, size_t ind2) {
    size_t i;
    assert(tsr1->nmodes == tsr2->nmodes);
    for(i = 0; i < tsr1->nmodes; ++i) {
        size_t eleind1 = tsr1->inds[i].data[ind1];
        size_t eleind2 = tsr2->inds[i].data[ind2];
        if(eleind1 < eleind2) {
            return -1;
        } else if(eleind1 > eleind2) {
            return 1;
        }
    }
    return 0;
}

