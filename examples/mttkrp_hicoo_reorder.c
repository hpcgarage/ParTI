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
#include <ParTI.h>
#include "../src/sptensor/sptensor.h"
#include "../src/sptensor/hicoo/hicoo.h"
#ifdef PARTI_USE_OPENMP
    #include <omp.h>
#endif


void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    printf("         -m MODE, --mode=MODE (default -1: loop all modes, or specify a mode, e.g., 0 or 1 or 2 for third-order tensors.)\n");
    printf("         -b BLOCKSIZE, --blocksize=BLOCKSIZE (in bits) (required)\n");
    printf("         -k KERNELSIZE, --kernelsize=KERNELSIZE (in bits) (required)\n");
    printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    printf("         -r RANK (the number of matrix columns, 16:default)\n");
    printf("         Reordering options: \n");
    printf("         -e RELABEL, --relabel=RELABEL (0:no-relabeling,default; 1:relabel with Lexi-order; 2:relabel with BFS-MCS; 3:randomly relabel; 4:relabel according to shuffle map in -s)\n");
    printf("         -n NITERS_RENUM (default: 3, required when -e 1)\n");
    printf("         -s SHUFFLE FILE, --shuffle=SHUFFLE FILE (required when -e 3)\n");
    printf("         OpenMP options: \n");
    printf("         -t NTHREADS, --nt=NT (1:default)\n");
    printf("         -a balanced\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char ** argv) 
{
    FILE *fi = NULL, *fo = NULL, *fs = NULL;
    sptSparseTensor tsr;
    sptRankMatrix ** U;
    sptRankMatrix ** copy_U;
    sptSparseTensorHiCOO hitsr;
    sptElementIndex sb_bits;
    sptElementIndex sk_bits;

    sptIndex mode = PARTI_INDEX_MAX;
    sptElementIndex R = 16;
    int dev_id = -2;
    int niters = 5;
    int relabel = 0;
    int niters_renum = 3;
    /* relabel:
     * = 0 : no relabeling.
     * = 1 : relabel with Lexi-order, specify niters_renum.
     * = 2 : relabel with BFS-like
     * = 3 : randomly relabel.
     * = 4 : relabel according to shuffle map in fs.
     */
    int tk = 1;
    int par_iters = 0;
    int balanced = -1, input_balanced = -1;
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
            {"mode", required_argument, 0, 'm'},
            {"shuffle", optional_argument, 0, 's'},
            {"output", optional_argument, 0, 'o'},
            {"impl-num", optional_argument, 0, 'p'},
            {"relabel", optional_argument, 0, 'e'},
            {"niters-renum", optional_argument, 0, 'n'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"tk", optional_argument, 0, 't'},
            {"balanced", optional_argument, 0, 'a'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 0;
        c = getopt_long(argc, argv, "i:o:b:k:m:e:d:r:t:n:a:s:", long_options, &option_index);
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
        case 's':
            fs = fopen(optarg, "r");
            sptAssert(fs != NULL);
            break;
        case 'b':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'k':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sk_bits);
            break;
        case 'm':
            sscanf(optarg, "%"PARTI_SCN_INDEX, &mode);
            break;
        case 'e':
            sscanf(optarg, "%d", &relabel);
            break;
        case 'n':
            sscanf(optarg, "%d", &niters_renum);
            break;
        case 'd':
            sscanf(optarg, "%d", &dev_id);
            break;
        case 'r':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &R);
            break;
        case 't':
            sscanf(optarg, "%d", &tk);
            break;
        case 'a':
            sscanf(optarg, "%d", &input_balanced);
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
    printf("dev_id: %d\n", dev_id);
    printf("relabel: %d\n", relabel);
    if (relabel == 1)
        printf("niters_renum: %d\n\n", niters_renum);

    /* A sorting included in load tensor */
    sptAssert(sptLoadSparseTensor(&tsr, 1, fi) == 0);
    fclose(fi);
    sptSparseTensorStatus(&tsr, stdout);
    // sptAssert(sptDumpSparseTensor(&tsr, 0, stdout) == 0);

    /* Relabel the input tensor */
    sptIndex ** map_inds;
    if (relabel > 0) {
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

        if ( relabel == 1 || relabel == 2) { /* Set the Lexi-order or BFS-like renumbering */
            sptIndexRenumber(&tsr, map_inds, relabel, niters_renum, tk);

        } else if ( relabel == 3) { /* Set randomly renumbering */
            sptGetRandomShuffledIndices(&tsr, map_inds);
        } else if ( relabel == 4) { /* read shuffle map from fs */
            if (fs == NULL) {
                printf("[Error]: Input shuffle file.\n");
                return -1;
            }
            sptLoadShuffleFile(&tsr, fs, map_inds);
            sptSparseTensorInvMap(&tsr, map_inds);
            fclose(fs);
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
    }

    /* Convert to HiCOO tensor */
    sptNnzIndex max_nnzb = 0;
    sptTimer convert_timer;
    sptNewTimer(&convert_timer, 0);
    sptStartTimer(convert_timer);

    sptAssert(sptSparseTensorToHiCOO(&hitsr, &max_nnzb, &tsr, sb_bits, sk_bits, tk) == 0);

    sptStopTimer(convert_timer);
    sptPrintElapsedTime(convert_timer, "Convert HiCOO");
    sptFreeTimer(convert_timer);

    sptFreeSparseTensor(&tsr);
    sptSparseTensorStatusHiCOO(&hitsr, stdout);

    /* Initialize factor matrices */
    sptIndex nmodes = hitsr.nmodes;
    sptNnzIndex factor_bytes = 0;
    U = (sptRankMatrix **)malloc((nmodes+1) * sizeof(sptRankMatrix*));
    for(sptIndex m=0; m<nmodes+1; ++m) {
      U[m] = (sptRankMatrix *)malloc(sizeof(sptRankMatrix));
    }
    sptIndex max_ndims = 0;
    for(sptIndex m=0; m<nmodes; ++m) {
      sptAssert(sptNewRankMatrix(U[m], hitsr.ndims[m], R) == 0);
      sptAssert(sptRandomizeRankMatrix(U[m], tsr.ndims[m], R) == 0);
      // sptAssert(sptConstantRankMatrix(U[m], 1) == 0);
      if(hitsr.ndims[m] > max_ndims)
        max_ndims = hitsr.ndims[m];
      factor_bytes += hitsr.ndims[m] * R * sizeof(sptValue);
    }
    sptAssert(sptNewRankMatrix(U[nmodes], max_ndims, R) == 0);
    sptAssert(sptConstantRankMatrix(U[nmodes], 0) == 0);

    /* output factor size */
    char * bytestr;
    bytestr = sptBytesString(factor_bytes);
    printf("FACTORS-STORAGE=%s\n", bytestr);
    printf("\n");
    free(bytestr);

    sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(*mats_order));
    sptIndex sk = (sptIndex)pow(2, hitsr.sk_bits);

    if (mode == PARTI_INDEX_MAX) {
        for(sptIndex mode=0; mode < nmodes; ++mode) {
            par_iters = 0;
            /* Reset U[nmodes] */
            U[nmodes]->nrows = hitsr.ndims[mode];
            sptAssert(sptConstantRankMatrix(U[nmodes], 0) == 0);

            /* Determine use balanced or not */
            double rest_nnz_portion_th = 0.10;
            printf("rest_nnz_portion_th: %.2lf\n", rest_nnz_portion_th);
            sptIndex num_kernel_dim = (hitsr.ndims[mode] + sk - 1) / sk;
            sptIndex npars = hitsr.kschr_balanced_pos[mode][0].len - 1;
            sptNnzIndex sum_balanced_nnzk = 0;
            sptIndex kernel_ndim = (hitsr.ndims[mode] + sk - 1)/sk;
            for(sptIndex i=0; i < kernel_ndim; ++i) {
              for(sptIndex j=0; j < hitsr.kschr_balanced[mode][i].len; ++j) {
                sptIndex kernel_num = hitsr.kschr_balanced[mode][i].data[j];
                sum_balanced_nnzk += hitsr.knnzs.data[kernel_num];
              }
            }
            double rest_nnz_portion = 1.0 - (double)sum_balanced_nnzk / hitsr.nnz;
            if (input_balanced == -1) {
                if (rest_nnz_portion < rest_nnz_portion_th && max(num_kernel_dim, npars) > PAR_MIN_DEGREE * NUM_CORES ) {
                    balanced = 1;
                } else {
                    balanced = 0;
                }
            } else {
                balanced = input_balanced;
            }
            printf("balanced: %d\n", balanced);
            fflush(stdout);

            /* determine niters or num_kernel_dim to be parallelized */
            sptIndex ratio_schr_ncols, num_tasks;
            if(balanced == 0) {
                ratio_schr_ncols = hitsr.nkiters[mode] / num_kernel_dim;
            } else if (balanced == 1) {
                ratio_schr_ncols = npars / num_kernel_dim;
            }
            printf("Schr ncols / nrows: %u (threshold: %u)\n", ratio_schr_ncols, PAR_DEGREE_REDUCE);
            if(num_kernel_dim <= PAR_MIN_DEGREE * NUM_CORES && ratio_schr_ncols >= PAR_DEGREE_REDUCE) {
                par_iters = 1;
            }
            if(balanced == 0) {
                num_tasks = (par_iters == 1) ? hitsr.nkiters[mode] : num_kernel_dim;
            } else if (balanced == 1) {
                num_tasks = (par_iters == 1) ? npars : num_kernel_dim;
            }
            printf("par_iters: %d, num_tasks: %u\n", par_iters, num_tasks);

            /* Set zeros for temporary copy_U, for mode-"mode" */
            if(dev_id == -1 && par_iters == 1) {
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

            /* For warm-up caches, timing not included */
            if(dev_id == -2) {
                sptAssert(sptMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode) == 0);
            } else if(dev_id == -1) {
                printf("tk: %d\n", tk);
                // printf("sptOmpMTTKRPHiCOO_MatrixTiling:\n");
                // sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode, tk) == 0);
                if(par_iters == 0) {
                    printf("sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled:\n");
                    sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled(&hitsr, U, mats_order, mode, tk, balanced) == 0);
                } else {
                    printf("sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce:\n");
                    sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(&hitsr, U, copy_U, mats_order, mode, tk, balanced) == 0);
                }
            }

            sptTimer timer;
            sptNewTimer(&timer, 0);
            sptStartTimer(timer);

            for(int it=0; it<niters; ++it) {
                if(dev_id == -2) {
                    sptAssert(sptMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode) == 0);
                } else if(dev_id == -1) {
                    // sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode, tk) == 0);
                    if(par_iters == 0) {
                        sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled(&hitsr, U, mats_order, mode, tk, balanced) == 0);
                    } else {
                        sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(&hitsr, U, copy_U, mats_order, mode, tk, balanced) == 0);
                    }
                }
            }

            sptStopTimer(timer);
            char * prg_name;
            int ret = asprintf(&prg_name, "CPU  SpTns MTTKRP MODE %"PARTI_PRI_INDEX, mode);
            if(ret < 0) {
                perror("asprintf");
                abort();
            }
            sptPrintAverageElapsedTime(timer, niters, prg_name);
            printf("\n");
            sptFreeTimer(timer);


            if (relabel > 0) {
                sptTimer back_shuffle_timer;
                sptNewTimer(&back_shuffle_timer, 0);
                sptStartTimer(back_shuffle_timer);
            
                sptRankMatrixInverseShuffleIndices(U[nmodes], map_inds[mode]);

                sptStopTimer(back_shuffle_timer);
                sptPrintElapsedTime(back_shuffle_timer, "Inverse-Shuffling time");
                sptFreeTimer(back_shuffle_timer);
                printf("\n");
            }
            if(fo != NULL) {
                sptAssert(sptDumpRankMatrix(U[nmodes], fo) == 0);
            }

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
        if(dev_id == -1 && par_iters == 1) {
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

        /* For warm-up caches, timing not included */
        if(dev_id == -2) {
            sptAssert(sptMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode) == 0);
        } else if(dev_id == -1) {
            printf("tk: %d\n", tk);
            // printf("sptOmpMTTKRPHiCOO_MatrixTiling:\n");
            // sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode, tk) == 0);
            if(par_iters == 0) {
                printf("sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled:\n");
                sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled(&hitsr, U, mats_order, mode, tk, balanced) == 0);
            } else {
                printf("sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce:\n");
                sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(&hitsr, U, copy_U, mats_order, mode, tk, balanced) == 0);
            }
        }

        sptTimer timer;
        sptNewTimer(&timer, 0);
        sptStartTimer(timer);

        for(int it=0; it<niters; ++it) {
            if(dev_id == -2) {
                sptAssert(sptMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode) == 0);
            } else if(dev_id == -1) {
                // sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode, tk) == 0);
                if(par_iters == 0) {
                    sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled(&hitsr, U, mats_order, mode, tk, balanced) == 0);
                } else {
                    sptAssert(sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(&hitsr, U, copy_U, mats_order, mode, tk, balanced) == 0);
                }
            }
        }

        sptStopTimer(timer);
        sptPrintAverageElapsedTime(timer, niters, "CPU  SpTns MTTKRP");
        printf("\n");
        sptFreeTimer(timer);

        sptTimer shuffle_timer;
        sptNewTimer(&shuffle_timer, 0);
        sptStartTimer(shuffle_timer);
        if (relabel > 0) {
            sptRankMatrixInverseShuffleIndices(U[nmodes], map_inds[mode]);
        }
        sptStopTimer(shuffle_timer);
        sptPrintElapsedTime(shuffle_timer, "Inverse-Shuffling time");
        sptFreeTimer(shuffle_timer);

        if(fo != NULL) {
            sptAssert(sptDumpRankMatrix(U[nmodes], fo) == 0);
        }
    }   // End execute a specified mode

    if (relabel > 0) {
        for(sptIndex m = 0; m < tsr.nmodes; ++m) {
            free(map_inds[m]);
        }
        free(map_inds);
    }

    if(fo != NULL) {
        fclose(fo);
    }
    if(dev_id == -1 && par_iters == 1) {
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
