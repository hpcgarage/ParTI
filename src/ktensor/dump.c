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

int sptDumpKruskalTensor(sptKruskalTensor *ktsr, sptIndex start_index, FILE *fp)
{
    int iores;
    sptIndex mode;

    iores = fprintf(fp, "nmodes: %u, rank: %u\n", ktsr->nmodes, ktsr->rank);
    spt_CheckOSError(iores < 0, "KruskalTns Dump");
    for(mode = 0; mode < ktsr->nmodes; ++mode) {
        if(mode != 0) {
            iores = fputs(" ", fp);
            spt_CheckOSError(iores < 0, "KruskalTns Dump");
        }
        iores = fprintf(fp, "%u", ktsr->ndims[mode]);
        spt_CheckOSError(iores < 0, "KruskalTns Dump");
    }
    fputs("\n", fp);
    iores = fprintf(fp, "nmodes: %u, rank: %u\n", ktsr->nmodes, ktsr->rank);
    spt_CheckOSError(iores < 0, "KruskalTns Dump");

    iores = fprintf(fp, "fit: %lf\n", ktsr->fit);
    fprintf(fp, "lambda:\n");    
    for(mode = 0; mode < ktsr->nmodes; ++mode) {
        iores = fprintf(fp, "%lf ", ktsr->lambda[mode]);
        spt_CheckOSError(iores != 0, "KruskalTns Dump");
    }

    fprintf(fp, "Factor matrices:\n");
    for(mode=0; mode < ktsr->nmodes+1; ++mode) {
        iores = sptDumpMatrix(ktsr->factors[mode], fp);
        spt_CheckOSError(iores != 0, "KruskalTns Dump");
    }
    return 0;
}


int sptDumpRankKruskalTensor(sptRankKruskalTensor *ktsr, sptIndex start_index, FILE *fp)
{
    int iores;
    sptIndex mode;

    iores = fprintf(fp, "nmodes: %u, rank: %u\n", ktsr->nmodes, ktsr->rank);
    spt_CheckOSError(iores < 0, "KruskalTns Dump");
    for(mode = 0; mode < ktsr->nmodes; ++mode) {
        if(mode != 0) {
            iores = fputs(" ", fp);
            spt_CheckOSError(iores < 0, "KruskalTns Dump");
        }
        iores = fprintf(fp, "%u", ktsr->ndims[mode]);
        spt_CheckOSError(iores < 0, "KruskalTns Dump");
    }
    fputs("\n", fp);
    iores = fprintf(fp, "nmodes: %u, rank: %u\n", ktsr->nmodes, ktsr->rank);
    spt_CheckOSError(iores < 0, "KruskalTns Dump");

    iores = fprintf(fp, "fit: %lf\n", ktsr->fit);
    fprintf(fp, "lambda:\n");    
    for(mode = 0; mode < ktsr->nmodes; ++mode) {
        iores = fprintf(fp, "%lf ", ktsr->lambda[mode]);
        spt_CheckOSError(iores != 0, "KruskalTns Dump");
    }

    fprintf(fp, "Factor matrices:\n");
    for(mode=0; mode < ktsr->nmodes+1; ++mode) {
        iores = sptDumpRankMatrix(ktsr->factors[mode], fp);
        spt_CheckOSError(iores != 0, "KruskalTns Dump");
    }
    return 0;
}