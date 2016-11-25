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

#include <assert.h>
#include <SpTOL.h>
#include "sptensor.h"

struct spt_SplitStatus {
    const sptSparseTensor *tsr;
    sptSizeVector cuts_by_mode;
    sptSizeVector partial_low;
    sptSizeVector partial_high;
    sptSizeVector index_step;
};

int spt_StartSplitSparseTensor(struct spt_SplitStatus *status, const sptSparseTensor *tsr, const size_t cuts_by_mode[]) {
    int result = 0;

    if(status->tsr->nnz == 0) {
        spt_CheckError(SPTERR_NO_MORE, "SpTns Star Split", "no splits");
    }

    status->tsr = tsr;
    result = sptNewSizeVector(&status->cuts_by_mode, tsr->nmodes, tsr->nmodes);
    spt_CheckError(result, "SpTns Start Split", NULL);
    memcpy(&status->cuts_by_mode.data, cuts_by_mode, tsr->nmodes * sizeof (size_t));

    result = sptNewSizeVector(&status->partial_low, 1, tsr->nmodes+1);
    spt_CheckError(result, "SpTns Start Split", NULL);
    result = sptNewSizeVector(&status->partial_high, 1, tsr->nmodes+1);
    spt_CheckError(result, "SpTns Start Split", NULL);
    result = sptNewSizeVector(&status->index_step, 0, tsr->nmodes);
    spt_CheckError(result, "SpTns Start Split", NULL);

    status->partial_low.data[0] = 0;
    status->partial_high.data[0] = tsr->nnz;

    return result;
}

int spt_SplitSparseTensor(sptSparseTensor *dest, struct spt_SplitStatus *status) {
    int result = 0;

    size_t mode = status->partial_low.len;

    while(mode <= status->tsr->nmodes) {
        size_t low = status->partial_low.data[mode-1];
        size_t high = status->partial_high.data[mode-1];
        assert(low < high);

        // Count distinct index values on this mode
        size_t last_index = status->tsr->inds[mode].data[low];
        size_t index_counts = 1;
        size_t i;
        for(i = low; i < high; ++i) {
            if(status->tsr->inds[mode].data[i] != last_index) {
                ++index_counts;
                last_index = status->tsr->inds[mode].data[i];
            }
        }

        // Calculate index step for this mode
        size_t index_step = index_counts / status->cuts_by_mode.data[mode];
        if(index_step == 0) {
            index_step = 1;
        }
        result = sptAppendSizeVector(&status->index_step, index_step);
        spt_CheckError(result, "SpTns Split", NULL);

        // Set initial cut for this mode
        last_index = status->tsr->inds[mode].data[low];
        index_counts = 0;
        for(i = low; i < high; ++i) {
            if(status->tsr->inds[mode].data[i] != last_index) {
                ++index_counts;
                last_index = status->tsr->inds[mode].data[i];
                if(index_counts == status->index_step.data[mode]) {
                    break;
                }
            }
        }
        result = sptAppendSizeVector(&status->partial_low, low);
        spt_CheckError(result, "SpTns Split", NULL);
        result = sptAppendSizeVector(&status->partial_high, i);
        spt_CheckError(result, "SpTns Split", NULL);

        ++mode;
    }

    // Now we have gone through the initial cutting for all modes
    // status->partial_low through status->partial_high should be the next cut
    // TODO: Do the cut

    // Find the next chunk and return current function
    mode = status->tsr->nmodes;
    while(mode-- > 0) {
        // Starting from the rest of this mode, to the end of previous mode
        size_t low = status->partial_high.data[mode];
        size_t high = status->partial_high.data[mode-1];
        if(low >= high) {
            --status->partial_low.len;
            --status->partial_high.len;
            --status->index_step.len;
            --mode;
            continue;
        }

        size_t last_index = status->tsr->inds[mode].data[low];
        size_t index_counts = 0;
        size_t i;
        for(i = low; i < high; ++i) {
            if(status->tsr->inds[mode].data[i] != last_index) {
                ++index_counts;
                last_index = status->tsr->inds[mode].data[i];
                if(index_counts == status->index_step.data[mode]) {
                    break;
                }
            }
        }
        status->partial_low.data[mode] = low;
        status->partial_high.data[mode] = i;
        return 0;
    }

    // Mode should be 0 now, which means unable to find the next chunk
    spt_CheckError(SPTERR_NO_MORE, "SpTns Star Split", "no more splits");
}

void spt_FinishSplitSparseTensor(struct spt_SplitStatus *status) {
    status->tsr = NULL;
    sptFreeSizeVector(&status->cuts_by_mode);
    sptFreeSizeVector(&status->partial_low);
    sptFreeSizeVector(&status->partial_high);
    sptFreeSizeVector(&status->index_step);
}
