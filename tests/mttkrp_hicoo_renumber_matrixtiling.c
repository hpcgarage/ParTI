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
#include "../src/sptensor/hicoo/hicoo.h"

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)\n");
    printf("         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)\n");
    printf("         -e RENUMBER, --renumber=RENUMBER\n");
    printf("         -m MODE, --mode=MODE (default -1: loop all modes)\n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
    printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
    printf("         -r RANK\n");
    printf("         -t TK, --tk=TK\n");
    printf("         -l TB, --tb=TB\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    printf("mttkrp_hicoo_renumber_matrixtiling: \n");

    FILE *fi = NULL, *fo = NULL;
    sptSparseTensor tsr;
    sptRankMatrix ** U;
    sptRankMatrix ** copy_U;
    sptSparseTensorHiCOO hitsr;
    sptElementIndex sb_bits;
    sptElementIndex sk_bits;
    sptElementIndex sc_bits;

    sptIndex mode = PARTI_INDEX_MAX;
    sptElementIndex R = 16;
    int cuda_dev_id = -2;
    int niters = 5;
    int nthreads;
    int impl_num = 0;
    int renumber = 0;
    int niters_renum = 3;
    /* renumber:
     * = 0 : no renumbering.
     * = 1 : renumber with Lexi-order
     * = 2 : renumber with BFS-like
     * = 3 : randomly renumbering, specify niters_renum.
     */
    int tk = 1;
    int tb = 1;
    int par_iters = 0;
    printf("niters: %d\n", niters);

    if(argc <= 6) { // #Required arguments
        print_usage(argv);
        exit(1);
    }

    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"bs", required_argument, 0, 'b'},
            {"ks", required_argument, 0, 'k'},
            {"cs", required_argument, 0, 'c'},
            {"mode", required_argument, 0, 'm'},
            {"output", optional_argument, 0, 'o'},
            {"impl-num", optional_argument, 0, 'p'},
            {"renumber", optional_argument, 0, 'e'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"tk", optional_argument, 0, 't'},
            {"tb", optional_argument, 0, 'l'},
            {"niters-renum", optional_argument, 0, 'n'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 0;
        c = getopt_long(argc, argv, "i:o:b:k:c:m:p:e:d:r:t:l:n:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            fi = fopen(optarg, "r");
            sptAssert(fi != NULL);
            break;
        case 'o':
            fo = fopen(optarg, "w");
            sptAssert(fo != NULL);
            break;
        case 'b':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'k':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sk_bits);
            break;
        case 'c':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sc_bits);
            break;
        case 'm':
            sscanf(optarg, "%"PARTI_SCN_INDEX, &mode);
            break;
        case 'p':
            sscanf(optarg, "%d", &impl_num);
            break;
        case 'e':
            sscanf(optarg, "%d", &renumber);
            break;
        case 'n':
            sscanf(optarg, "%d", &niters_renum);
            break;
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case 'r':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &R);
            break;
        case 't':
            sscanf(optarg, "%d", &tk);
            break;
        case 'l':
            sscanf(optarg, "%d", &tb);
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
    printf("renumber: %d\n", renumber);
    if (renumber == 1)
        printf("niters_renum: %d\n\n", niters_renum);

    /* A sorting included in load tensor */
    sptAssert(sptLoadSparseTensor(&tsr, 1, fi) == 0);
    fclose(fi);
    sptSparseTensorStatus(&tsr, stdout);
    // sptAssert(sptDumpSparseTensor(&tsr, 0, stdout) == 0);

    /* Renumber the input tensor */
    sptIndex ** map_inds;
    if (renumber > 0) {
        map_inds = (sptIndex **)malloc(tsr.nmodes * sizeof *map_inds);
        spt_CheckOSError(!map_inds, "MTTKRP HiCOO");
        for(sptIndex m = 0; m < tsr.nmodes; ++m) {
            map_inds[m] = (sptIndex *)malloc(tsr.ndims[m] * sizeof (sptIndex));
            spt_CheckError(!map_inds[m], "MTTKRP HiCOO", NULL);
            for(sptIndex i = 0; i < tsr.ndims[m]; ++i) 
                map_inds[m][i] = i;
        }

        sptTimer renumber_timer;
        sptNewTimer(&renumber_timer, 0);
        sptStartTimer(renumber_timer);

        if ( renumber == 1 || renumber == 2) { /* Set the Lexi-order or BFS-like renumbering */
            orderit(&tsr, map_inds, renumber, niters_renum);
            // sptIndexRenumber(&tsr, map_inds, renumber, niters_renum);
        }
        if ( renumber == 3) { /* Set randomly renumbering */
            sptGetRandomShuffledIndices(&tsr, map_inds);
        }
        fflush(stdout);

        sptStopTimer(renumber_timer);
        sptPrintElapsedTime(renumber_timer, "Renumbering");
        sptFreeTimer(renumber_timer);

        sptTimer shuffle_timer;
        sptNewTimer(&shuffle_timer, 0);
        sptStartTimer(shuffle_timer);

        sptSparseTensorShuffleIndices(&tsr, map_inds);

        sptStopTimer(shuffle_timer);
        sptPrintElapsedTime(shuffle_timer, "Shuffling time");
        sptFreeTimer(shuffle_timer);
        printf("\n");


        // sptSparseTensorSortIndex(&tsr, 1);   // debug purpose only
        // FILE * debug_fp = fopen("new.txt", "w");
        // sptAssert(sptDumpSparseTensor(&tsr, 0, debug_fp) == 0);
        // fprintf(debug_fp, "\nmap_inds:\n");
        // for(sptIndex m = 0; m < tsr.nmodes; ++m) {
        //     sptDumpIndexArray(map_inds[m], tsr.ndims[m], debug_fp);
        // }
        // sptAssert(sptDumpSparseTensor(&tsr, 0, debug_fp) == 0);
        // fclose(debug_fp);
    }

    /* Convert to HiCOO tensor */
    sptNnzIndex max_nnzb = 0;
    sptTimer convert_timer;
    sptNewTimer(&convert_timer, 0);
    sptStartTimer(convert_timer);

    sptAssert(sptSparseTensorToHiCOO(&hitsr, &max_nnzb, &tsr, sb_bits, sk_bits, sc_bits) == 0);

    sptStopTimer(convert_timer);
    sptPrintElapsedTime(convert_timer, "Convert HiCOO");
    sptFreeTimer(convert_timer);

    sptFreeSparseTensor(&tsr);
    sptSparseTensorStatusHiCOO(&hitsr, stdout);
    // sptAssert(sptDumpSparseTensorHiCOO(&hitsr, stdout) == 0);

    /* Initialize factor matrices */
    sptIndex nmodes = hitsr.nmodes;
    sptNnzIndex factor_bytes = 0;
    U = (sptRankMatrix **)malloc((nmodes+1) * sizeof(sptRankMatrix*));
    for(sptIndex m=0; m<nmodes+1; ++m) {
      U[m] = (sptRankMatrix *)malloc(sizeof(sptRankMatrix));
    }
    sptIndex max_ndims = 0;
    for(sptIndex m=0; m<nmodes; ++m) {
      // sptAssert(sptRandomizeMatrix(U[m], tsr.ndims[m], R) == 0);
      sptAssert(sptNewRankMatrix(U[m], hitsr.ndims[m], R) == 0);
      sptAssert(sptConstantRankMatrix(U[m], 1) == 0);
      if(hitsr.ndims[m] > max_ndims)
        max_ndims = hitsr.ndims[m];
      factor_bytes += hitsr.ndims[m] * R * sizeof(sptValue);
      // sptAssert(sptDumpMatrix(U[m], stdout) == 0);
    }
    sptAssert(sptNewRankMatrix(U[nmodes], max_ndims, R) == 0);
    sptAssert(sptConstantRankMatrix(U[nmodes], 0) == 0);
    // sptAssert(sptDumpMatrix(U[nmodes], stdout) == 0);

    /* output factor size */
    char * bytestr;
    bytestr = sptBytesString(factor_bytes);
    printf("FACTORS-STORAGE=%s\n", bytestr);
    printf("\n");
    free(bytestr);

    sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(*mats_order));

    if (mode == PARTI_INDEX_MAX) {
        for(sptIndex mode=0; mode<nmodes; ++mode) {
            par_iters = 0;
            /* Reset U[nmodes] */
            sptAssert(sptConstantRankMatrix(U[nmodes], 0) == 0);

            /* determine niters or num_kernel_dim to be parallelized */
            sptIndex sk = (sptIndex)pow(2, hitsr.sk_bits);
            sptIndex num_kernel_dim = (hitsr.ndims[mode] + sk - 1) / sk;
            printf("hitsr.nkiters[mode] / num_kernel_dim: %u (threshold: %u)\n", hitsr.nkiters[mode]/num_kernel_dim, PAR_DEGREE_REDUCE);
            if(num_kernel_dim <= PAR_MIN_DEGREE * NUM_CORES && hitsr.nkiters[mode] / num_kernel_dim >= PAR_DEGREE_REDUCE) {
                par_iters = 1;
            }
            sptIndex num_tasks = (par_iters == 1) ? hitsr.nkiters[mode] : num_kernel_dim;
            printf("par_iters: %d, num_tasks: %u\n", par_iters, num_tasks);

            /* Set zeros for temporary copy_U, for mode-"mode" */
            if(cuda_dev_id == -1 && par_iters == 1) {
                copy_U = (sptRankMatrix **)malloc(tk * sizeof(sptRankMatrix*));
                for(int t=0; t<tk; ++t) {
                    copy_U[t] = (sptRankMatrix *)malloc(sizeof(sptRankMatrix));
                    sptAssert(sptNewRankMatrix(copy_U[t], hitsr.ndims[mode], R) == 0);
                    sptAssert(sptConstantRankMatrix(copy_U[t], 0) == 0);
                }
                sptNnzIndex bytes = tk * hitsr.ndims[mode] * R * sizeof(sptValue);
                bytestr = sptBytesString(bytes);
                printf("MODE MATRIX COPY=%s\n", bytestr);
            }

            mats_order[0] = mode;
            for(sptIndex i=1; i<nmodes; ++i)
                mats_order[i] = (mode+i) % nmodes;
            // printf("mats_order:\n");
            // sptDumpIndexArray(mats_order, nmodes, stdout);

            /* For warm-up caches, timing not included */
            if(cuda_dev_id == -2) {
                nthreads = 1;
                sptAssert(sptMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode) == 0);
            } else if(cuda_dev_id == -1) {
                printf("tk: %d, tb: %d\n", tk, tb);
                // printf("sptOmpMTTKRPHiCOO_MatrixTiling:\n");
                // sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode, tk, tb) == 0);
                if(par_iters == 0) {
                    printf("sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled:\n");
                    sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled(&hitsr, U, mats_order, mode, tk, tb) == 0);
                } else {
                    printf("sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce:\n");
                    sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(&hitsr, U, copy_U, mats_order, mode, tk, tb) == 0);
                }
            }

            sptTimer timer;
            sptNewTimer(&timer, 0);
            sptStartTimer(timer);

            for(int it=0; it<niters; ++it) {
                if(cuda_dev_id == -2) {
                    nthreads = 1;
                    sptAssert(sptMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode) == 0);
                } else if(cuda_dev_id == -1) {
                    // sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode, tk, tb) == 0);
                    if(par_iters == 0) {
                        sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled(&hitsr, U, mats_order, mode, tk, tb) == 0);
                    } else {
                        sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(&hitsr, U, copy_U, mats_order, mode, tk, tb) == 0);
                    }
                }
            }

            sptStopTimer(timer);
            char * prg_name;
            asprintf(&prg_name, "CPU  SpTns MTTKRP MODE %"PARTI_PRI_INDEX, mode);
            sptPrintAverageElapsedTime(timer, niters, prg_name);
            printf("\n");
            sptFreeTimer(timer);
        }   // End nmodes

    } else {
    
        /* determine niters or num_kernel_dim to be parallelized */
        sptIndex sk = (sptIndex)pow(2, hitsr.sk_bits);
        sptIndex num_kernel_dim = (hitsr.ndims[mode] + sk - 1) / sk;
        printf("num_kernel_dim: %u, hitsr.nkiters[mode] / num_kernel_dim: %u\n", num_kernel_dim, hitsr.nkiters[mode]/num_kernel_dim);
        if(num_kernel_dim <= PAR_MIN_DEGREE * NUM_CORES && hitsr.nkiters[mode] / num_kernel_dim >= PAR_DEGREE_REDUCE) {
            par_iters = 1;
        }

        /* Set zeros for temporary copy_U, for mode-"mode" */
        if(cuda_dev_id == -1 && par_iters == 1) {
            copy_U = (sptRankMatrix **)malloc(tk * sizeof(sptRankMatrix*));
            for(int t=0; t<tk; ++t) {
                copy_U[t] = (sptRankMatrix *)malloc(sizeof(sptRankMatrix));
                sptAssert(sptNewRankMatrix(copy_U[t], hitsr.ndims[mode], R) == 0);
                sptAssert(sptConstantRankMatrix(copy_U[t], 0) == 0);
            }
            sptNnzIndex bytes = tk * hitsr.ndims[mode] * R * sizeof(sptValue);
            bytestr = sptBytesString(bytes);
            printf("MODE MATRIX COPY=%s\n", bytestr);
        }

        sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(*mats_order));
        mats_order[0] = mode;
        for(sptIndex i=1; i<nmodes; ++i)
            mats_order[i] = (mode+i) % nmodes;
        // printf("mats_order:\n");
        // sptDumpIndexArray(mats_order, nmodes, stdout);

        /* For warm-up caches, timing not included */
        if(cuda_dev_id == -2) {
            nthreads = 1;
            sptAssert(sptMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode) == 0);
        } else if(cuda_dev_id == -1) {
            printf("tk: %d, tb: %d\n", tk, tb);
            // printf("sptOmpMTTKRPHiCOO_MatrixTiling:\n");
            // sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode, tk, tb) == 0);
            if(par_iters == 0) {
                printf("sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled:\n");
                sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled(&hitsr, U, mats_order, mode, tk, tb) == 0);
            } else {
                printf("sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce:\n");
                sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(&hitsr, U, copy_U, mats_order, mode, tk, tb) == 0);
            }
        }

        sptTimer timer;
        sptNewTimer(&timer, 0);
        sptStartTimer(timer);

        for(int it=0; it<niters; ++it) {
            if(cuda_dev_id == -2) {
                nthreads = 1;
                sptAssert(sptMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode) == 0);
            } else if(cuda_dev_id == -1) {
                // sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode, tk, tb) == 0);
                if(par_iters == 0) {
                    sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled(&hitsr, U, mats_order, mode, tk, tb) == 0);
                } else {
                    sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(&hitsr, U, copy_U, mats_order, mode, tk, tb) == 0);
                }
            }
        }

        sptStopTimer(timer);
        sptPrintAverageElapsedTime(timer, niters, "CPU  SpTns MTTKRP");
        printf("\n");
        sptFreeTimer(timer);

        if(fo != NULL) {
            sptAssert(sptDumpRankMatrix(U[nmodes], fo) == 0);
            fclose(fo);
        }
    }   // End execute a specified mode

    if (renumber > 0) {
        for(sptIndex m = 0; m < tsr.nmodes; ++m) {
            free(map_inds[m]);
        }
        free(map_inds);
    }

    if(cuda_dev_id == -1 && par_iters == 1) {
        for(int t=0; t<tk; ++t) {
            sptFreeRankMatrix(copy_U[t]);
        }
        free(copy_U);
        free(bytestr);
    }
    for(sptIndex m=0; m<nmodes; ++m) {
        sptFreeRankMatrix(U[m]);
    }
    sptFreeRankMatrix(U[nmodes]);
    free(U);
    free(mats_order);
    sptFreeSparseTensorHiCOO(&hitsr);

    return 0;
}
