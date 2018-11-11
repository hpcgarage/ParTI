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
#include <stdio.h>
#include <stdlib.h>
#include "../src/error/error.h"

int main(void) {
    {
        static char bufX[] = "3\n"
            "3 3 3\n"
            "0 0 0 1\n"
            "0 0 1 2\n"
            "0 0 2 3\n"
            "0 1 0 4\n"
            "0 1 1 5\n"
            "0 1 2 6\n"
            "0 2 0 7\n"
            "0 2 1 8\n"
            "0 2 2 9\n"
            "1 0 0 1\n"
            "1 0 1 2\n"
            "1 0 2 3\n"
            "1 1 0 4\n"
            "1 1 1 5\n"
            "1 1 2 6\n"
            "1 2 0 7\n"
            "1 2 1 8\n"
            "1 2 2 9\n"
            "2 0 0 1\n"
            "2 0 1 2\n"
            "2 0 2 3\n"
            "2 1 0 4\n"
            "2 1 1 5\n"
            "2 1 2 6\n"
            "2 2 0 7\n"
            "2 2 1 8\n"
            "2 2 2 9\n";
        FILE *stream = fmemopen(bufX, sizeof bufX - 1, "r");
        sptSparseTensor X;
        int result = sptLoadSparseTensor(&X, 0, stream);
        spt_CheckError(result, "load", NULL);
        fclose(stream);

        sptKruskalTensor ktensor;
        result = sptNewKruskalTensor(&ktensor, 3, X.ndims, 2);
        spt_CheckError(result, "new ktensor", NULL);
        result = sptCpdAls(&X, 2, 5, 1e-9, &ktensor);
        spt_CheckError(result, "cpd als", NULL);

        char *buf = calloc(1, 1024);
        stream = fmemopen(buf, 1023, "w");
        result = sptDumpKruskalTensor(&ktensor, stream);
        spt_CheckError(result, "dump", NULL);
        fclose(stream);

        // TODO: allow reproducible tests by allowing specifying random seed
        printf("%s\n", buf);
        free(buf);

        sptFreeKruskalTensor(&ktensor);
        sptFreeSparseTensor(&X);
    }
    return 0;
}
