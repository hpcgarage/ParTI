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

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <omp.h>
#include <ParTI.h>
#include "../src/sptensor/sptensor.h"

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    printf("         -m MODE, --mode=MODE (default -1: loop all modes, or specify a mode, e.g., 0 or 1 or 2 for third-order tensors.)\n");
    printf("         -s sortcase, --sortcase=SORTCASE (0:default,1,2,3,4. Different tensor sorting.)\n");
    printf("         -b BLOCKSIZE, --blocksize=BLOCKSIZE (in bits) (only for sortcase=3)\n");
    printf("         -k KERNELSIZE, --kernelsize=KERNELSIZE (in bits) (only for sortcase=3)\n");
    printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    printf("         -r RANK (the number of matrix columns, 16:default)\n");
    printf("         Reordering options: \n");
    printf("         -e RELABEL, --relabel=RELABEL (0:no-relabeling,default; 1:relabel with Lexi-order; 2:relabel with BFS-MCS; 3:randomly relabel)\n");
    printf("         -n NITERS_RENUM (default: 3, required when -e 1)\n");
    printf("         OpenMP options: \n");
    printf("         -t NTHREADS, --nt=NT (1:default)\n");
    printf("         -u use_reduce, --ur=use_reduce (use privatization or not)\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    FILE *fi = NULL, *fo = NULL;
    sptSparseTensor X;
    sptMatrix ** U;
    sptMatrix ** copy_U;

    sptIndex mode = PARTI_INDEX_MAX;
    sptIndex R = 16;
    int cuda_dev_id = -2;
    int niters = 5;
    int relabel = 0;
    int niters_renum = 3;
    /* relabel:
     * = 0 : no relabel.
     * = 1 : relabel with Lexi-order
     * = 2 : relabel with BFS-like
     * = 3 : randomly relabel, specify niters_renum.
     */
    int use_reduce = 0; // Need to choose from two omp parallel approaches
    int nt = 1;
    /* sortcase:
     * = 0 : the same with the old COO code.
     * = 1 : best case. Sort order: [mode, (ordered by increasing dimension sizes)]
     * = 2 : worse case. Sort order: [(ordered by decreasing dimension sizes)]
     * = 3 : Z-Morton ordering (same with HiCOO format order)
     * = 4 : random shuffling for elements.
     * = 5 : blocking only not mode-n indices.
     */
    int sortcase = 0;
    sptElementIndex sb_bits;
    sptElementIndex sk_bits;
    printf("niters: %d\n", niters);

    if(argc <= 3) { // #Required arguments
        print_usage(argv);
        exit(1);
    }

    int c;
    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"mode", required_argument, 0, 'm'},
            {"output", optional_argument, 0, 'o'},
            {"bs", required_argument, 0, 'b'},
            {"ks", required_argument, 0, 'k'},
            {"sortcase", optional_argument, 0, 's'},
            {"impl-num", optional_argument, 0, 'p'},
            {"relabel", optional_argument, 0, 'e'},
            {"niters-renum", optional_argument, 0, 'n'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"nt", optional_argument, 0, 't'},
            {"use-reduce", optional_argument, 0, 'u'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "i:m:o:b:k:s:e:d:r:t:u:n:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            fi = fopen(optarg, "r");
            sptAssert(fi != NULL);
            printf("input file: %s\n", optarg); fflush(stdout);
            break;
        case 'o':
            fo = fopen(optarg, "aw");
            sptAssert(fo != NULL);
            printf("output file: %s\n", optarg); fflush(stdout);
            break;
        case 'm':
            sscanf(optarg, "%"PARTI_SCN_INDEX, &mode);
            break;
        case 'b':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'k':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sk_bits);
            break;
        case 's':
            sscanf(optarg, "%d", &sortcase);
            break;
        case 'e':
            sscanf(optarg, "%d", &relabel);
            break;
        case 'n':
            sscanf(optarg, "%d", &niters_renum);
            break;
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case 'r':
            sscanf(optarg, "%u"PARTI_SCN_INDEX, &R);
            break;
        case 'u':
            sscanf(optarg, "%d", &use_reduce);
            break;
        case 't':
            sscanf(optarg, "%d", &nt);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }

    if(mode == PARTI_INDEX_MAX) printf("mode: all\n");
    else printf("mode: %"PARTI_PRI_INDEX "\n", mode);
    printf("cuda_dev_id: %d\n", cuda_dev_id);
    printf("sortcase: %d\n", sortcase);
    printf("relabel: %d\n", relabel);
    if (relabel == 1)
        printf("niters_renum: %d\n\n", niters_renum);

    /* Load a sparse tensor from file as it is */
    sptAssert(sptLoadSparseTensor(&X, 1, fi) == 0);
    fclose(fi);

    /* relabel the input tensor */
    sptIndex ** map_inds;
    if (relabel > 0) {
        map_inds = (sptIndex **)malloc(X.nmodes * sizeof *map_inds);
        spt_CheckOSError(!map_inds, "MTTKRP HiCOO");
        for(sptIndex m = 0; m < X.nmodes; ++m) {
            map_inds[m] = (sptIndex *)malloc(X.ndims[m] * sizeof (sptIndex));
            spt_CheckError(!map_inds[m], "MTTKRP HiCOO", NULL);
            for(sptIndex i = 0; i < X.ndims[m]; ++i) 
                map_inds[m][i] = i;
        }

        sptTimer renumber_timer;
        sptNewTimer(&renumber_timer, 0);
        sptStartTimer(renumber_timer);

        if ( relabel == 1 || relabel == 2) { /* Set the Lexi-order or BFS-like renumbering */
            sptIndexRenumber(&X, map_inds, relabel, niters_renum, nt);
        }
        if ( relabel == 3) { /* Set randomly renumbering */
            sptGetRandomShuffledIndices(&X, map_inds);
        }

        sptStopTimer(renumber_timer);
        sptPrintElapsedTime(renumber_timer, "Renumbering");
        sptFreeTimer(renumber_timer);

        sptTimer shuffle_timer;
        sptNewTimer(&shuffle_timer, 0);
        sptStartTimer(shuffle_timer);

        sptSparseTensorShuffleIndices(&X, map_inds);

        sptStopTimer(shuffle_timer);
        sptPrintElapsedTime(shuffle_timer, "Shuffling time");
        sptFreeTimer(shuffle_timer);
        printf("\n");
    }

    sptIndex nmodes = X.nmodes;
    U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    for(sptIndex m=0; m<nmodes+1; ++m) {
      U[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    }
    sptIndex max_ndims = 0;
    for(sptIndex m=0; m<nmodes; ++m) {
      sptAssert(sptNewMatrix(U[m], X.ndims[m], R) == 0);
      sptAssert(sptRandomizeMatrix(U[m], X.ndims[m], R) == 0);
      // sptAssert(sptConstantMatrix(U[m], 1) == 0);
      if(X.ndims[m] > max_ndims)
        max_ndims = X.ndims[m];
    }
    sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    sptIndex stride = U[0]->stride;

    sptIndex * mode_order = (sptIndex*) malloc(nmodes * sizeof(*mode_order));
    sptIndex * mats_order = (sptIndex*) malloc(nmodes * sizeof(sptIndex));

    /* Initialize locks */
    sptMutexPool * lock_pool = NULL;
    if(cuda_dev_id == -1 && use_reduce == 0) {
        lock_pool = sptMutexAlloc();
    }


    if (mode == PARTI_INDEX_MAX) {

        for(sptIndex mode=0; mode<nmodes; ++mode) {

            /* Reset U[nmodes] */
            U[nmodes]->nrows = X.ndims[mode];
            sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);

            /* Sort sparse tensor */
            memset(mode_order, 0, X.nmodes * sizeof(*mode_order));
            switch (sortcase) {
                case 0:
                    sptSparseTensorSortIndex(&X, 1, nt);
                    break;
                case 1:
                    sptGetBestModeOrder(mode_order, mode, X.ndims, X.nmodes);
                    sptSparseTensorSortIndexCustomOrder(&X, mode_order, 1, nt);
                    break;
                case 2:
                    sptGetWorstModeOrder(mode_order, mode, X.ndims, X.nmodes);
                    sptSparseTensorSortIndexCustomOrder(&X, mode_order, 1, nt);
                    break;
                case 3:
                    /* Pre-process tensor, the same with the one used in HiCOO.
                     * Only difference is not setting kptr and kschr in this function.
                     */
                    sptSparseTensorMixedOrder(&X, sb_bits, sk_bits, nt);
                    break;
                case 4:
                    sptGetRandomShuffleElements(&X);
                    break;
                case 5:
                    sptGetBestModeOrder(mode_order, mode, X.ndims, X.nmodes);
                    sptSparseTensorSortPartialIndex(&X, mode_order, sb_bits, nt);
                    break;
                default:
                    printf("Wrong sortcase number, reset by -s. \n");
            }
            if(sortcase != 0) {
                printf("mode_order:\n");
                sptDumpIndexArray(mode_order, X.nmodes, stdout);
            }

            sptSparseTensorStatus(&X, stdout);

            /* Set zeros for temporary copy_U, for mode-"mode" */
            char * bytestr;
            if(cuda_dev_id == -1 && use_reduce == 1) {
                copy_U = (sptMatrix **)malloc(nt * sizeof(sptMatrix*));
                for(int t=0; t<nt; ++t) {
                    copy_U[t] = (sptMatrix *)malloc(sizeof(sptMatrix));
                    sptAssert(sptNewMatrix(copy_U[t], X.ndims[mode], R) == 0);
                    sptAssert(sptConstantMatrix(copy_U[t], 0) == 0);
                }
                sptNnzIndex bytes = nt * X.ndims[mode] * R * sizeof(sptValue);
                bytestr = sptBytesString(bytes);
                printf("MODE MATRIX COPY=%s\n", bytestr);
                free(bytestr);
            }

            switch (sortcase) {
            case 0:
            case 3:
            case 4:
            case 5:
                mats_order[0] = mode;
                for(sptIndex i=1; i<nmodes; ++i)
                    mats_order[i] = (mode+i) % nmodes;
                break;
            case 1: // Reverse of mode_order except the 1st one
                mats_order[0] = mode;
                for(sptIndex i=1; i<nmodes; ++i)
                    mats_order[i] = mode_order[nmodes - i];
                break;
            case 2: // Totally reverse of mode_order
                for(sptIndex i=0; i<nmodes; ++i)
                    mats_order[i] = mode_order[nmodes - i];
                break;
            }

            /* For warm-up caches, timing not included */
            if(cuda_dev_id == -2) {
                sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
            } else if(cuda_dev_id == -1) {
                printf("nt: %d\n", nt);
                if(use_reduce == 1) {
                    printf("sptOmpMTTKRP_Reduce:\n");
                    sptAssert(sptOmpMTTKRP_Reduce(&X, U, copy_U, mats_order, mode, nt) == 0);
                } else {
                    printf("sptOmpMTTKRP:\n");
                    sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nt) == 0);
                    // printf("sptOmpMTTKRP_Lock:\n");
                    // sptAssert(sptOmpMTTKRP_Lock(&X, U, mats_order, mode, nt, lock_pool) == 0);
                }
            }

            
            sptTimer timer;
            sptNewTimer(&timer, 0);
            sptStartTimer(timer);

            for(int it=0; it<niters; ++it) {
                // sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
                if(cuda_dev_id == -2) {
                    sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
                } else if(cuda_dev_id == -1) {
                    if(use_reduce == 1) {
                        sptAssert(sptOmpMTTKRP_Reduce(&X, U, copy_U, mats_order, mode, nt) == 0);
                    } else {
                        sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nt) == 0);
                        // printf("sptOmpMTTKRP_Lock:\n");
                        // sptAssert(sptOmpMTTKRP_Lock(&X, U, mats_order, mode, nt, lock_pool) == 0);
                    }
                }
            }

            sptStopTimer(timer);

            if(cuda_dev_id == -2 || cuda_dev_id == -1) {
                char * prg_name;
                int ret = asprintf(&prg_name, "CPU  SpTns MTTKRP MODE %"PARTI_PRI_INDEX, mode);
                if(ret < 0) {
                    perror("asprintf");
                    abort();
                }
                double aver_time = sptPrintAverageElapsedTime(timer, niters, prg_name);

                double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
                uint64_t bytes = ( nmodes * sizeof(sptIndex) + sizeof(sptValue) ) * X.nnz; 
                for (sptIndex m=0; m<nmodes; ++m) {
                    bytes += X.ndims[m] * R * sizeof(sptValue);
                }
                double gbw = (double)bytes / aver_time / 1e9;
                // printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
            }
            sptFreeTimer(timer);

            if(fo != NULL) {
                if (relabel > 0) {
                    sptMatrixInverseShuffleIndices(U[nmodes], map_inds[mode]);
                }
                    sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
            }

        } // End nmodes

    } else {
        /* Sort sparse tensor */
        memset(mode_order, 0, X.nmodes * sizeof(*mode_order));
        switch (sortcase) {
            case 0:
                sptSparseTensorSortIndex(&X, 1, nt);
                break;
            case 1:
                sptGetBestModeOrder(mode_order, mode, X.ndims, X.nmodes);
                sptSparseTensorSortIndexCustomOrder(&X, mode_order, 1, nt);
                break;
            case 2:
                sptGetWorstModeOrder(mode_order, mode, X.ndims, X.nmodes);
                sptSparseTensorSortIndexCustomOrder(&X, mode_order, 1, nt);
                break;
            case 3:
                /* Pre-process tensor, the same with the one used in HiCOO.
                 * Only difference is not setting kptr and kschr in this function.
                 */
                sptSparseTensorMixedOrder(&X, sb_bits, sk_bits, nt);
                break;
            case 4:
                sptGetRandomShuffleElements(&X);
                break;
            case 5:
                sptGetBestModeOrder(mode_order, mode, X.ndims, X.nmodes);
                sptSparseTensorSortPartialIndex(&X, mode_order, sb_bits, nt);
                break;
            default:
                printf("Wrong sortcase number, reset by -s. \n");
        }
        if(sortcase != 0) {
            printf("mode_order:\n");
            sptDumpIndexArray(mode_order, X.nmodes, stdout);
        }

        /* Set zeros for temporary copy_U, for mode-"mode" */
        char * bytestr;
        if(cuda_dev_id == -1 && use_reduce == 1) {
            copy_U = (sptMatrix **)malloc(nt * sizeof(sptMatrix*));
            for(int t=0; t<nt; ++t) {
                copy_U[t] = (sptMatrix *)malloc(sizeof(sptMatrix));
                sptAssert(sptNewMatrix(copy_U[t], X.ndims[mode], R) == 0);
                sptAssert(sptConstantMatrix(copy_U[t], 0) == 0);
            }
            sptNnzIndex bytes = nt * X.ndims[mode] * R * sizeof(sptValue);
            bytestr = sptBytesString(bytes);
            printf("MODE MATRIX COPY=%s\n", bytestr);
            free(bytestr);
        }

        sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(sptIndex));
        switch (sortcase) {
        case 0:
        case 3:
        case 4:
        case 5:
            mats_order[0] = mode;
            for(sptIndex i=1; i<nmodes; ++i)
                mats_order[i] = (mode+i) % nmodes;
            break;
        case 1: // Reverse of mode_order except the 1st one
            mats_order[0] = mode;
            for(sptIndex i=1; i<nmodes; ++i)
                mats_order[i] = mode_order[nmodes - i];
            break;
        case 2: // Totally reverse of mode_order
            for(sptIndex i=0; i<nmodes; ++i)
                mats_order[i] = mode_order[nmodes - i];
            break;
        }

        /* For warm-up caches, timing not included */
        if(cuda_dev_id == -2) {
            sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
        } else if(cuda_dev_id == -1) {
            printf("nt: %d\n", nt);
            if(use_reduce == 1) {
                printf("sptOmpMTTKRP_Reduce:\n");
                sptAssert(sptOmpMTTKRP_Reduce(&X, U, copy_U, mats_order, mode, nt) == 0);
            } else {
                printf("sptOmpMTTKRP:\n");
                sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nt) == 0);
                // printf("sptOmpMTTKRP_Lock:\n");
                // sptAssert(sptOmpMTTKRP_Lock(&X, U, mats_order, mode, nt, lock_pool) == 0);
            }
        }

        
        sptTimer timer;
        sptNewTimer(&timer, 0);
        sptStartTimer(timer);

        for(int it=0; it<niters; ++it) {
            // sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
            if(cuda_dev_id == -2) {
                sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
            } else if(cuda_dev_id == -1) {
                if(use_reduce == 1) {
                    sptAssert(sptOmpMTTKRP_Reduce(&X, U, copy_U, mats_order, mode, nt) == 0);
                } else {
                    sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nt) == 0);
                    // printf("sptOmpMTTKRP_Lock:\n");
                    // sptAssert(sptOmpMTTKRP_Lock(&X, U, mats_order, mode, nt, lock_pool) == 0);
                }
            }
        }

        sptStopTimer(timer);

        if(cuda_dev_id == -2 || cuda_dev_id == -1) {
            double aver_time = sptPrintAverageElapsedTime(timer, niters, "CPU SpTns MTTKRP");

            double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
            uint64_t bytes = ( nmodes * sizeof(sptIndex) + sizeof(sptValue) ) * X.nnz; 
            for (sptIndex m=0; m<nmodes; ++m) {
                bytes += X.ndims[m] * R * sizeof(sptValue);
            }
            double gbw = (double)bytes / aver_time / 1e9;
            printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
        }
        sptFreeTimer(timer);

        if(fo != NULL) {
            if (relabel > 0) {
                sptMatrixInverseShuffleIndices(U[nmodes], map_inds[mode]);
            }
            sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
        }

    } // End execute a specified mode

    if(fo != NULL) {
        fclose(fo);
    }
    if (relabel > 0) {
        for(sptIndex m = 0; m < X.nmodes; ++m) {
            free(map_inds[m]);
        }
        free(map_inds);
    }
    if(cuda_dev_id == -1) {
        if (use_reduce == 1) {
            for(int t=0; t<nt; ++t) {
                sptFreeMatrix(copy_U[t]);
            }
            free(copy_U);
        }
        if(lock_pool != NULL) {
            sptMutexFree(lock_pool);
        }
    }
    for(sptIndex m=0; m<nmodes; ++m) {
        sptFreeMatrix(U[m]);
    }
    sptFreeSparseTensor(&X);
    free(mats_order);
    free(mode_order);
    sptFreeMatrix(U[nmodes]);
    free(U);

    return 0;
}
