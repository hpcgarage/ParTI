#include <SpTOL.h>
#include "ssptensor.h"
#include <assert.h>

int spt_SemiSparseTensorCompareIndices(const sptSemiSparseTensor *tsr1, size_t ind1, const sptSemiSparseTensor *tsr2, size_t ind2) {
    size_t i;
    assert(tsr1->nmodes == tsr2->nmodes);
    assert(tsr1->mode == tsr2->mode);
    for(i = 0; i < tsr1->nmodes; ++i) {
        if(i != tsr1->mode) {
            size_t eleind1 = tsr1->inds[i].data[ind1];
            size_t eleind2 = tsr2->inds[i].data[ind2];
            if(eleind1 < eleind2) {
                return -1;
            } else if(eleind1 > eleind2) {
                return 1;
            }
        }
    }
    return 0;
}
