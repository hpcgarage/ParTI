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
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -m MODE, --mode=MODE (default -1: loop all modes)\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)\n");
    printf("         -s sortcase, --sortcase=SORTCASE (1,2,3,4)\n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
    printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
    printf("         -r RANK\n");
    printf("         -t NTHREADS, --nt=NT\n");
    printf("         -u use_reduce, --ur=use_reduce\n");
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
    int nthreads;
    int impl_num = 0;
    int use_reduce = 1; // Need to choose from two omp parallel approaches
    int nt = 1;
    /* sortcase:
     * = 0 : the same with the old COO code.
     * = 1 : best case. Sort order: [mode, (ordered by increasing dimension sizes)]
     * = 2 : worse case. Sort order: [(ordered by decreasing dimension sizes)]
     * = 3 : Z-Morton ordering (same with HiCOO format order)
     * = 4 : random shuffling.
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
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"nt", optional_argument, 0, 't'},
            {"use-reduce", optional_argument, 0, 'u'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "i:m:o:b:k:s:p:d:r:t:u:", long_options, &option_index);
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
            fo = fopen(optarg, "w");
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
        case 'p':
            sscanf(optarg, "%d", &impl_num);
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

    printf("mode: %"PARTI_PRI_INDEX "\n", mode);
    printf("cuda_dev_id: %d\n", cuda_dev_id);
    printf("sortcase: %d\n", sortcase);

    /* Load a sparse tensor from file as it is */
    sptAssert(sptLoadSparseTensor(&X, 1, fi) == 0);
    fclose(fi);
    // sptAssert(sptDumpSparseTensor(&X, 0, stdout) == 0);

    sptIndex nmodes = X.nmodes;
    U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    for(sptIndex m=0; m<nmodes+1; ++m) {
      U[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    }
    sptIndex max_ndims = 0;
    for(sptIndex m=0; m<nmodes; ++m) {
      // sptAssert(sptRandomizeMatrix(U[m], X.ndims[m], R) == 0);
      sptAssert(sptNewMatrix(U[m], X.ndims[m], R) == 0);
      sptAssert(sptConstantMatrix(U[m], 1) == 0);
      if(X.ndims[m] > max_ndims)
        max_ndims = X.ndims[m];
    }
    sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    sptIndex stride = U[0]->stride;

    sptIndex * mode_order = (sptIndex*) malloc(X.nmodes * sizeof(*mode_order));
    sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(sptIndex));

    /* Initialize locks */
    sptMutexPool * lock_pool = NULL;
    if(cuda_dev_id == -1 && use_reduce == 0) {
        lock_pool = sptMutexAlloc();
    }


    if (mode == PARTI_INDEX_MAX) {

        for(sptIndex mode=0; mode<nmodes; ++mode) {

            /* Sort sparse tensor */
            memset(mode_order, 0, X.nmodes * sizeof(*mode_order));
            switch (sortcase) {
                case 0:
                    sptSparseTensorSortIndex(&X, 1);
                    break;
                case 1:
                    sptGetBestModeOrder(mode_order, mode, X.ndims, X.nmodes);
                    sptSparseTensorSortIndexCustomOrder(&X, mode_order, 1);
                    break;
                case 2:
                    sptGetWorstModeOrder(mode_order, mode, X.ndims, X.nmodes);
                    sptSparseTensorSortIndexCustomOrder(&X, mode_order, 1);
                    break;
                case 3:
                    /* Pre-process tensor, the same with the one used in HiCOO.
                     * Only difference is not setting kptr and kschr in this function.
                     */
                    sptSparseTensorMixedOrder(&X, sb_bits, sk_bits, nt);
                    break;
                case 4:
                    // sptGetBestModeOrder(mode_order, 0, X.ndims, X.nmodes);
                    sptGetRandomShuffleElements(&X);
                    break;
                default:
                    printf("Wrong sortcase number, reset by -s. \n");
            }
            if(sortcase != 0) {
                printf("mode_order:\n");
                sptDumpIndexArray(mode_order, X.nmodes, stdout);
            }

            sptSparseTensorStatus(&X, stdout);
            // sptAssert(sptDumpSparseTensor(&X, 0, stdout) == 0);


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
            // printf("mats_order:\n");
            // sptDumpIndexArray(mats_order, nmodes, stdout);


            /* For warm-up caches, timing not included */
            if(cuda_dev_id == -2) {
                nthreads = 1;
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
                    nthreads = 1;
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
                asprintf(&prg_name, "CPU  SpTns MTTKRP MODE %"PARTI_PRI_INDEX, mode);
                double aver_time = sptPrintAverageElapsedTime(timer, niters, prg_name);

                double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
                uint64_t bytes = ( nmodes * sizeof(sptIndex) + sizeof(sptValue) ) * X.nnz; 
                for (sptIndex m=0; m<nmodes; ++m) {
                    bytes += X.ndims[m] * R * sizeof(sptValue);
                }
                double gbw = (double)bytes / aver_time / 1e9;
                printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
            }
            sptFreeTimer(timer);

        } // End nmodes

    } else {
        /* Sort sparse tensor */
        memset(mode_order, 0, X.nmodes * sizeof(*mode_order));
        switch (sortcase) {
            case 0:
                sptSparseTensorSortIndex(&X, 1);
                break;
            case 1:
                sptGetBestModeOrder(mode_order, mode, X.ndims, X.nmodes);
                sptSparseTensorSortIndexCustomOrder(&X, mode_order, 1);
                break;
            case 2:
                sptGetWorstModeOrder(mode_order, mode, X.ndims, X.nmodes);
                sptSparseTensorSortIndexCustomOrder(&X, mode_order, 1);
                break;
            case 3:
                /* Pre-process tensor, the same with the one used in HiCOO.
                 * Only difference is not setting kptr and kschr in this function.
                 */
                sptSparseTensorMixedOrder(&X, sb_bits, sk_bits, nt);
                break;
            case 4:
                // sptGetBestModeOrder(mode_order, 0, X.ndims, X.nmodes);
                sptGetRandomShuffleElements(&X);
                break;
            default:
                printf("Wrong sortcase number, reset by -s. \n");
        }
        if(sortcase != 0) {
            printf("mode_order:\n");
            sptDumpIndexArray(mode_order, X.nmodes, stdout);
        }

        sptSparseTensorStatus(&X, stdout);
        // sptAssert(sptDumpSparseTensor(&X, 0, stdout) == 0);

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
        // printf("mats_order:\n");
        // sptDumpIndexArray(mats_order, nmodes, stdout);

        /* For warm-up caches, timing not included */
        if(cuda_dev_id == -2) {
            nthreads = 1;
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
                nthreads = 1;
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

    } // End execute a specified mode

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

    if(fo != NULL) {
        sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
        fclose(fo);
    }

    sptFreeMatrix(U[nmodes]);
    free(U);

    return 0;
}
