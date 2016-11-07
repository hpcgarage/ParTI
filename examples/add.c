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

#include <stdio.h>
#include <stdlib.h>
#include <SpTOL.h>

int main(int argc, char *argv[]) {
    FILE *fa, *fb, *fo;
    sptSparseTensor a, b, c;

    if(argc != 4 && argc != 5) {
        printf("Usage: %s a b out [num_iters]\n\n", argv[0]);
        exit(1);
    }

    int niters = 1;
    if(argc == 5) {
        niters = atoi(argv[4]);
        if(niters < 1) {
            fprintf(stderr, "Error: invalid num_iters value \"%s\"\n", argv[4]);
            exit(1);
        }
    }

    fa = fopen(argv[1], "r");
    if(!fa) {
        fprintf(stderr, "Error: failed to open file \"%s\"\n", argv[1]);
        exit(1);
    }
    if(sptLoadSparseTensor(&a, 1, fa)) {
        fprintf(stderr, "Error: failed to load tensor A\n");
        exit(1);
    }
    fclose(fa);

    fb = fopen(argv[2], "r");
    if(!fb) {
        fprintf(stderr, "Error: failed to open file \"%s\"\n", argv[2]);
        exit(1);
    }
    if(sptLoadSparseTensor(&b, 1, fb)) {
        fprintf(stderr, "Error: failed to load tensor B\n");
        exit(1);
    }
    fclose(fb);

    sptTimer timer;
    if(sptNewTimer(&timer, 0)) {
        fprintf(stderr, "Error: failed to create timer\n");
        exit(1);
    }
    if(sptStartTimer(timer)) {
        fprintf(stderr, "Error: failed to start timer\n");
        exit(1);
    }

    for(int i = 0; i<niters; ++i) {
        if(sptSparseTensorAdd(&c, &a, &b)) {
            fprintf(stderr, "Error: failed to calculate A+B\n");
            exit(1);
        }
    }

    if(sptStopTimer(timer)) {
        fprintf(stderr, "Error: failed to stop timer\n");
        exit(1);
    }
    sptPrintElapsedTime(timer, "Add");

    fo = fopen(argv[3], "w");
    if(!fo) {
        fprintf(stderr, "Error: failed to open file \"%s\"\n", argv[3]);
        exit(1);
    }
    if(sptDumpSparseTensor(&c, 1, fo)) {
        fprintf(stderr, "Error: failed to dump tensor\n");
        exit(1);
    }
    fclose(fo);

    return 0;
}
