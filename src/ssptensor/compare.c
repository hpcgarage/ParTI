#include <SpTOL.h>
#include "ssptensor.h"
#include <assert.h>

/**
 * compare two indices from two identical or distinct semi sparse tensors lexicographically
 * @param tsr1 the first semi sparse tensor
 * @param ind1 the order of the element in the first semi sparse tensor whose index is to be compared
 * @param tsr2 the second semi sparse tensor
 * @param ind2 the order of the element in the second semi sparse tensor whose index is to be compared
 * @return -1 for less, 0 for equal, 1 for greater
 */
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
