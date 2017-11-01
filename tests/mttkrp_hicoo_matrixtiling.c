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

int main(int argc, char * const argv[]) {
    FILE *fi = NULL, *fo = NULL;
    sptSparseTensor tsr;
    sptRankMatrix ** U;
    sptRankMatrix ** copy_U;
    sptSparseTensorHiCOO hitsr;
    sptElementIndex sb_bits;
    sptElementIndex sk_bits;
    sptElementIndex sc_bits;

    sptIndex mode = 0;
    sptIndex R = 16;
    int cuda_dev_id = -2;
    int niters = 5;
    int nthreads;
    int impl_num = 0;
    int tk = 1;
    int tb = 1;
    int par_iters = 0;
    printf("niters: %d\n", niters);

    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"output", required_argument, 0, 'o'},
            {"bs", required_argument, 0, 'b'},
            {"ks", required_argument, 0, 'k'},
            {"cs", required_argument, 0, 'c'},
            {"mode", required_argument, 0, 'm'},
            {"impl-num", optional_argument, 0, 'p'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"tk", optional_argument, 0, 't'},
            {"tb", optional_argument, 0, 'h'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 0;
        // c = getopt_long(argc, argv, "i:o:b:k:c:m:", long_options, &option_index);
        c = getopt_long(argc, argv, "i:o:b:k:c:m:p:d:r:t:h:", long_options, &option_index);
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
            sscanf(optarg, "%"SPT_PF_ELEMENTINDEX, &sb_bits);
            break;
        case 'k':
            sscanf(optarg, "%"SPT_PF_ELEMENTINDEX, &sk_bits);
            break;
        case 'c':
            sscanf(optarg, "%"SPT_PF_ELEMENTINDEX, &sc_bits);
            break;
        case 'm':
            sscanf(optarg, "%"SPT_PF_INDEX, &mode);
            break;
        case 'p':
            sscanf(optarg, "%d", &impl_num);
            break;
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case 'r':
            sscanf(optarg, "%"SPT_PF_INDEX, &R);
            break;
        case 't':
            sscanf(optarg, "%d", &tk);
            break;
        case 'h':
            sscanf(optarg, "%d", &tb);
            break;
        default:
            abort();
        }
    }

    // if(optind > argc) {
    if(argc < 2) {
        printf("Usage: %s [options] \n", argv[0]);
        printf("Options: -i INPUT, --input=INPUT\n");
        printf("         -o OUTPUT, --output=OUTPUT\n");
        printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
        printf("         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)\n");
        printf("         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)\n");
        printf("         -m MODE, --mode=MODE\n");
        printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
        printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
        printf("         -r RANK\n");
        printf("         -t TK, --tk=TK\n");
        printf("         -h TB, --tb=TB\n");
        printf("\n");
        return 1;
    }


    sptAssert(sptLoadSparseTensor(&tsr, 1, fi) == 0);
    fclose(fi);
    sptSparseTensorStatus(&tsr, stdout);
    // sptAssert(sptDumpSparseTensor(&tsr, 0, stdout) == 0);

    /* Convert to HiCOO tensor */
    sptNnzIndex max_nnzb = 0;
    sptAssert(sptSparseTensorToHiCOO(&hitsr, &max_nnzb, &tsr, sb_bits, sk_bits, sc_bits) == 0);
    sptFreeSparseTensor(&tsr);
    sptSparseTensorStatusHiCOO(&hitsr, stdout);
    // sptAssert(sptDumpSparseTensorHiCOO(&hitsr, stdout) == 0);

    sptIndex nmodes = hitsr.nmodes;
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
      // sptAssert(sptDumpMatrix(U[m], stdout) == 0);
    }
    sptAssert(sptNewRankMatrix(U[nmodes], max_ndims, R) == 0);
    sptAssert(sptConstantRankMatrix(U[nmodes], 0) == 0);
    // sptAssert(sptDumpMatrix(U[nmodes], stdout) == 0);

    /* determine niters or num_kernel_dim to be parallelized */
    sptIndex sk = (sptIndex)pow(2, hitsr.sk_bits);
    sptIndex num_kernel_dim = (hitsr.ndims[mode] + sk - 1) / sk;
    printf("num_kernel_dim: %u, hitsr.nkiters[mode] / num_kernel_dim: %u\n", num_kernel_dim, hitsr.nkiters[mode]/num_kernel_dim);
    if(num_kernel_dim <= 24 && hitsr.nkiters[mode] / num_kernel_dim >= 20) {
        par_iters = 1;
    }

    /* Set zeros for temporary copy_U, for mode-"mode" */
    char * bytestr;
    if(cuda_dev_id == -1 && par_iters == 1) {
        copy_U = (sptRankMatrix **)malloc(tk * sizeof(sptRankMatrix*));
        for(int t=0; t<tk; ++t) {
            copy_U[t] = (sptRankMatrix *)malloc(sizeof(sptRankMatrix));
            sptAssert(sptNewRankMatrix(copy_U[t], hitsr.ndims[mode], R) == 0);
            sptAssert(sptConstantRankMatrix(copy_U[t], 0) == 0);
        }
        sptNnzIndex bytes = tk * hitsr.ndims[mode] * R * sizeof(sptValue);
        bytestr = sptBytesString(bytes);
        printf("MODE MATRIX COPY=%s\n\n", bytestr);
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
    } /* else {
        sptCudaSetDevice(cuda_dev_id);
        sptAssert(sptCudaMTTKRPHiCOO(&hitsr, U, mats_order, mode, impl_num) == 0);
    } */

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
        } /* else {
            sptCudaSetDevice(cuda_dev_id);
            sptAssert(sptCudaMTTKRPHiCOO(&hitsr, U, mats_order, mode, impl_num) == 0);
        } */
    }

    sptStopTimer(timer);
    sptPrintAverageElapsedTime(timer, niters, "CPU  SpTns MTTKRP");
    sptFreeTimer(timer);

    if(fo != NULL) {
        sptAssert(sptDumpRankMatrix(U[nmodes], fo) == 0);
        fclose(fo);
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
