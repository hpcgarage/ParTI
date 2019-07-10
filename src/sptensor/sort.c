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

#include <assert.h>
#include <math.h>
#include <time.h>
#include <ParTI.h>
#include "sptensor.h"


/*************************************************
 * PRIVATE FUNCTIONS
 *************************************************/
static void spt_QuickSortIndex(sptSparseTensor *tsr, sptNnzIndex l, sptNnzIndex r);
static void spt_QuickSortAtMode(sptSparseTensor *tsr, sptNnzIndex const l, sptNnzIndex const r, sptIndex const mode);
static void spt_QuickSortIndexRowBlock(sptSparseTensor *tsr, sptNnzIndex l, sptNnzIndex r, const sptElementIndex sk_bits);
static void spt_QuickSortIndexExceptSingleMode(sptSparseTensor *tsr, sptNnzIndex l, sptNnzIndex r, sptIndex * mode_order);
static void spt_QuickSortIndexMorton3D(sptSparseTensor *tsr, sptNnzIndex l, sptNnzIndex r, const sptElementIndex sb_bits);
static void spt_QuickSortIndexMorton4D(sptSparseTensor *tsr, sptNnzIndex l, sptNnzIndex r, const sptElementIndex sb_bits);

static int spt_SparseTensorCompareAtMode(const sptSparseTensor *tsr1, sptNnzIndex const ind1, const sptSparseTensor *tsr2, sptNnzIndex const ind2, sptIndex const mode);
static int spt_SparseTensorCompareIndicesRowBlock(
    const sptSparseTensor *tsr1, 
    sptNnzIndex loc1, 
    const sptSparseTensor *tsr2, 
    sptNnzIndex loc2,
    const sptElementIndex sk_bits);
static int spt_SparseTensorCompareIndicesMorton3D(
    const sptSparseTensor *tsr1, 
    uint64_t loc1, 
    const sptSparseTensor *tsr2, 
    uint64_t loc2);
static int spt_SparseTensorCompareIndicesMorton4D(
    const sptSparseTensor *tsr1, 
    uint64_t loc1, 
    const sptSparseTensor *tsr2, 
    uint64_t loc2);

/* Mode order: X -> Y -> Z, x indices are sorted, y and z are Morton order sorted. */
static const uint32_t morton256_z[256] =
{
    0x00000000,
    0x00000001, 0x00000008, 0x00000009, 0x00000040, 0x00000041, 0x00000048, 0x00000049, 0x00000200,
    0x00000201, 0x00000208, 0x00000209, 0x00000240, 0x00000241, 0x00000248, 0x00000249, 0x00001000,
    0x00001001, 0x00001008, 0x00001009, 0x00001040, 0x00001041, 0x00001048, 0x00001049, 0x00001200,
    0x00001201, 0x00001208, 0x00001209, 0x00001240, 0x00001241, 0x00001248, 0x00001249, 0x00008000,
    0x00008001, 0x00008008, 0x00008009, 0x00008040, 0x00008041, 0x00008048, 0x00008049, 0x00008200,
    0x00008201, 0x00008208, 0x00008209, 0x00008240, 0x00008241, 0x00008248, 0x00008249, 0x00009000,
    0x00009001, 0x00009008, 0x00009009, 0x00009040, 0x00009041, 0x00009048, 0x00009049, 0x00009200,
    0x00009201, 0x00009208, 0x00009209, 0x00009240, 0x00009241, 0x00009248, 0x00009249, 0x00040000,
    0x00040001, 0x00040008, 0x00040009, 0x00040040, 0x00040041, 0x00040048, 0x00040049, 0x00040200,
    0x00040201, 0x00040208, 0x00040209, 0x00040240, 0x00040241, 0x00040248, 0x00040249, 0x00041000,
    0x00041001, 0x00041008, 0x00041009, 0x00041040, 0x00041041, 0x00041048, 0x00041049, 0x00041200,
    0x00041201, 0x00041208, 0x00041209, 0x00041240, 0x00041241, 0x00041248, 0x00041249, 0x00048000,
    0x00048001, 0x00048008, 0x00048009, 0x00048040, 0x00048041, 0x00048048, 0x00048049, 0x00048200,
    0x00048201, 0x00048208, 0x00048209, 0x00048240, 0x00048241, 0x00048248, 0x00048249, 0x00049000,
    0x00049001, 0x00049008, 0x00049009, 0x00049040, 0x00049041, 0x00049048, 0x00049049, 0x00049200,
    0x00049201, 0x00049208, 0x00049209, 0x00049240, 0x00049241, 0x00049248, 0x00049249, 0x00200000,
    0x00200001, 0x00200008, 0x00200009, 0x00200040, 0x00200041, 0x00200048, 0x00200049, 0x00200200,
    0x00200201, 0x00200208, 0x00200209, 0x00200240, 0x00200241, 0x00200248, 0x00200249, 0x00201000,
    0x00201001, 0x00201008, 0x00201009, 0x00201040, 0x00201041, 0x00201048, 0x00201049, 0x00201200,
    0x00201201, 0x00201208, 0x00201209, 0x00201240, 0x00201241, 0x00201248, 0x00201249, 0x00208000,
    0x00208001, 0x00208008, 0x00208009, 0x00208040, 0x00208041, 0x00208048, 0x00208049, 0x00208200,
    0x00208201, 0x00208208, 0x00208209, 0x00208240, 0x00208241, 0x00208248, 0x00208249, 0x00209000,
    0x00209001, 0x00209008, 0x00209009, 0x00209040, 0x00209041, 0x00209048, 0x00209049, 0x00209200,
    0x00209201, 0x00209208, 0x00209209, 0x00209240, 0x00209241, 0x00209248, 0x00209249, 0x00240000,
    0x00240001, 0x00240008, 0x00240009, 0x00240040, 0x00240041, 0x00240048, 0x00240049, 0x00240200,
    0x00240201, 0x00240208, 0x00240209, 0x00240240, 0x00240241, 0x00240248, 0x00240249, 0x00241000,
    0x00241001, 0x00241008, 0x00241009, 0x00241040, 0x00241041, 0x00241048, 0x00241049, 0x00241200,
    0x00241201, 0x00241208, 0x00241209, 0x00241240, 0x00241241, 0x00241248, 0x00241249, 0x00248000,
    0x00248001, 0x00248008, 0x00248009, 0x00248040, 0x00248041, 0x00248048, 0x00248049, 0x00248200,
    0x00248201, 0x00248208, 0x00248209, 0x00248240, 0x00248241, 0x00248248, 0x00248249, 0x00249000,
    0x00249001, 0x00249008, 0x00249009, 0x00249040, 0x00249041, 0x00249048, 0x00249049, 0x00249200,
    0x00249201, 0x00249208, 0x00249209, 0x00249240, 0x00249241, 0x00249248, 0x00249249
};

// pre-shifted table for Y coordinates (1 bit to the left)
static const uint32_t morton256_y[256] = {
    0x00000000,
    0x00000002, 0x00000010, 0x00000012, 0x00000080, 0x00000082, 0x00000090, 0x00000092, 0x00000400,
    0x00000402, 0x00000410, 0x00000412, 0x00000480, 0x00000482, 0x00000490, 0x00000492, 0x00002000,
    0x00002002, 0x00002010, 0x00002012, 0x00002080, 0x00002082, 0x00002090, 0x00002092, 0x00002400,
    0x00002402, 0x00002410, 0x00002412, 0x00002480, 0x00002482, 0x00002490, 0x00002492, 0x00010000,
    0x00010002, 0x00010010, 0x00010012, 0x00010080, 0x00010082, 0x00010090, 0x00010092, 0x00010400,
    0x00010402, 0x00010410, 0x00010412, 0x00010480, 0x00010482, 0x00010490, 0x00010492, 0x00012000,
    0x00012002, 0x00012010, 0x00012012, 0x00012080, 0x00012082, 0x00012090, 0x00012092, 0x00012400,
    0x00012402, 0x00012410, 0x00012412, 0x00012480, 0x00012482, 0x00012490, 0x00012492, 0x00080000,
    0x00080002, 0x00080010, 0x00080012, 0x00080080, 0x00080082, 0x00080090, 0x00080092, 0x00080400,
    0x00080402, 0x00080410, 0x00080412, 0x00080480, 0x00080482, 0x00080490, 0x00080492, 0x00082000,
    0x00082002, 0x00082010, 0x00082012, 0x00082080, 0x00082082, 0x00082090, 0x00082092, 0x00082400,
    0x00082402, 0x00082410, 0x00082412, 0x00082480, 0x00082482, 0x00082490, 0x00082492, 0x00090000,
    0x00090002, 0x00090010, 0x00090012, 0x00090080, 0x00090082, 0x00090090, 0x00090092, 0x00090400,
    0x00090402, 0x00090410, 0x00090412, 0x00090480, 0x00090482, 0x00090490, 0x00090492, 0x00092000,
    0x00092002, 0x00092010, 0x00092012, 0x00092080, 0x00092082, 0x00092090, 0x00092092, 0x00092400,
    0x00092402, 0x00092410, 0x00092412, 0x00092480, 0x00092482, 0x00092490, 0x00092492, 0x00400000,
    0x00400002, 0x00400010, 0x00400012, 0x00400080, 0x00400082, 0x00400090, 0x00400092, 0x00400400,
    0x00400402, 0x00400410, 0x00400412, 0x00400480, 0x00400482, 0x00400490, 0x00400492, 0x00402000,
    0x00402002, 0x00402010, 0x00402012, 0x00402080, 0x00402082, 0x00402090, 0x00402092, 0x00402400,
    0x00402402, 0x00402410, 0x00402412, 0x00402480, 0x00402482, 0x00402490, 0x00402492, 0x00410000,
    0x00410002, 0x00410010, 0x00410012, 0x00410080, 0x00410082, 0x00410090, 0x00410092, 0x00410400,
    0x00410402, 0x00410410, 0x00410412, 0x00410480, 0x00410482, 0x00410490, 0x00410492, 0x00412000,
    0x00412002, 0x00412010, 0x00412012, 0x00412080, 0x00412082, 0x00412090, 0x00412092, 0x00412400,
    0x00412402, 0x00412410, 0x00412412, 0x00412480, 0x00412482, 0x00412490, 0x00412492, 0x00480000,
    0x00480002, 0x00480010, 0x00480012, 0x00480080, 0x00480082, 0x00480090, 0x00480092, 0x00480400,
    0x00480402, 0x00480410, 0x00480412, 0x00480480, 0x00480482, 0x00480490, 0x00480492, 0x00482000,
    0x00482002, 0x00482010, 0x00482012, 0x00482080, 0x00482082, 0x00482090, 0x00482092, 0x00482400,
    0x00482402, 0x00482410, 0x00482412, 0x00482480, 0x00482482, 0x00482490, 0x00482492, 0x00490000,
    0x00490002, 0x00490010, 0x00490012, 0x00490080, 0x00490082, 0x00490090, 0x00490092, 0x00490400,
    0x00490402, 0x00490410, 0x00490412, 0x00490480, 0x00490482, 0x00490490, 0x00490492, 0x00492000,
    0x00492002, 0x00492010, 0x00492012, 0x00492080, 0x00492082, 0x00492090, 0x00492092, 0x00492400,
    0x00492402, 0x00492410, 0x00492412, 0x00492480, 0x00492482, 0x00492490, 0x00492492
};

// Pre-shifted table for x (2 bits to the left)
static const uint32_t morton256_x[256] = {
    0x00000000,
    0x00000004, 0x00000020, 0x00000024, 0x00000100, 0x00000104, 0x00000120, 0x00000124, 0x00000800,
    0x00000804, 0x00000820, 0x00000824, 0x00000900, 0x00000904, 0x00000920, 0x00000924, 0x00004000,
    0x00004004, 0x00004020, 0x00004024, 0x00004100, 0x00004104, 0x00004120, 0x00004124, 0x00004800,
    0x00004804, 0x00004820, 0x00004824, 0x00004900, 0x00004904, 0x00004920, 0x00004924, 0x00020000,
    0x00020004, 0x00020020, 0x00020024, 0x00020100, 0x00020104, 0x00020120, 0x00020124, 0x00020800,
    0x00020804, 0x00020820, 0x00020824, 0x00020900, 0x00020904, 0x00020920, 0x00020924, 0x00024000,
    0x00024004, 0x00024020, 0x00024024, 0x00024100, 0x00024104, 0x00024120, 0x00024124, 0x00024800,
    0x00024804, 0x00024820, 0x00024824, 0x00024900, 0x00024904, 0x00024920, 0x00024924, 0x00100000,
    0x00100004, 0x00100020, 0x00100024, 0x00100100, 0x00100104, 0x00100120, 0x00100124, 0x00100800,
    0x00100804, 0x00100820, 0x00100824, 0x00100900, 0x00100904, 0x00100920, 0x00100924, 0x00104000,
    0x00104004, 0x00104020, 0x00104024, 0x00104100, 0x00104104, 0x00104120, 0x00104124, 0x00104800,
    0x00104804, 0x00104820, 0x00104824, 0x00104900, 0x00104904, 0x00104920, 0x00104924, 0x00120000,
    0x00120004, 0x00120020, 0x00120024, 0x00120100, 0x00120104, 0x00120120, 0x00120124, 0x00120800,
    0x00120804, 0x00120820, 0x00120824, 0x00120900, 0x00120904, 0x00120920, 0x00120924, 0x00124000,
    0x00124004, 0x00124020, 0x00124024, 0x00124100, 0x00124104, 0x00124120, 0x00124124, 0x00124800,
    0x00124804, 0x00124820, 0x00124824, 0x00124900, 0x00124904, 0x00124920, 0x00124924, 0x00800000,
    0x00800004, 0x00800020, 0x00800024, 0x00800100, 0x00800104, 0x00800120, 0x00800124, 0x00800800,
    0x00800804, 0x00800820, 0x00800824, 0x00800900, 0x00800904, 0x00800920, 0x00800924, 0x00804000,
    0x00804004, 0x00804020, 0x00804024, 0x00804100, 0x00804104, 0x00804120, 0x00804124, 0x00804800,
    0x00804804, 0x00804820, 0x00804824, 0x00804900, 0x00804904, 0x00804920, 0x00804924, 0x00820000,
    0x00820004, 0x00820020, 0x00820024, 0x00820100, 0x00820104, 0x00820120, 0x00820124, 0x00820800,
    0x00820804, 0x00820820, 0x00820824, 0x00820900, 0x00820904, 0x00820920, 0x00820924, 0x00824000,
    0x00824004, 0x00824020, 0x00824024, 0x00824100, 0x00824104, 0x00824120, 0x00824124, 0x00824800,
    0x00824804, 0x00824820, 0x00824824, 0x00824900, 0x00824904, 0x00824920, 0x00824924, 0x00900000,
    0x00900004, 0x00900020, 0x00900024, 0x00900100, 0x00900104, 0x00900120, 0x00900124, 0x00900800,
    0x00900804, 0x00900820, 0x00900824, 0x00900900, 0x00900904, 0x00900920, 0x00900924, 0x00904000,
    0x00904004, 0x00904020, 0x00904024, 0x00904100, 0x00904104, 0x00904120, 0x00904124, 0x00904800,
    0x00904804, 0x00904820, 0x00904824, 0x00904900, 0x00904904, 0x00904920, 0x00904924, 0x00920000,
    0x00920004, 0x00920020, 0x00920024, 0x00920100, 0x00920104, 0x00920120, 0x00920124, 0x00920800,
    0x00920804, 0x00920820, 0x00920824, 0x00920900, 0x00920904, 0x00920920, 0x00920924, 0x00924000,
    0x00924004, 0x00924020, 0x00924024, 0x00924100, 0x00924104, 0x00924120, 0x00924124, 0x00924800,
    0x00924804, 0x00924820, 0x00924824, 0x00924900, 0x00924904, 0x00924920, 0x00924924
};


static inline void spt_SwapValues(sptSparseTensor *tsr, sptNnzIndex ind1, sptNnzIndex ind2) {

    for(sptIndex i = 0; i < tsr->nmodes; ++i) {
        sptIndex eleind1 = tsr->inds[i].data[ind1];
        tsr->inds[i].data[ind1] = tsr->inds[i].data[ind2];
        tsr->inds[i].data[ind2] = eleind1;
    }
    sptValue val1 = tsr->values.data[ind1];
    tsr->values.data[ind1] = tsr->values.data[ind2];
    tsr->values.data[ind2] = val1;
}


/*************************************************
 * PUBLIC FUNCTIONS
 *************************************************/

/**
 * Determine the best mode order. Sort order: [mode, (ordered by increasing dimension sizes)]
 *
 * @param[out] mode_order a pointer to the array to be filled,
 * @param[in] mode mode to do product
 * @param[in] ndims tensor dimension sizes
 * @param[in] nmodes tensor order
 *
 */
void sptGetBestModeOrder(
    sptIndex * mode_order,
    sptIndex const mode,
    sptIndex const * ndims,
    sptIndex const nmodes)
{
    sptKeyValuePair * sorted_ndims = (sptKeyValuePair*)malloc(nmodes * sizeof(*sorted_ndims));
    for(sptIndex m=0; m<nmodes; ++m) {
        sorted_ndims[m].key = m;
        sorted_ndims[m].value = ndims[m];
    }

    /* Increasingly sort */
    sptPairArraySort(sorted_ndims, nmodes);

    for(sptIndex m=0; m<nmodes; ++m) {
        mode_order[m] = sorted_ndims[m].key;
    }
    /* Find the location of mode */
    sptIndex mode_loc = 0;
    for(sptIndex m=0; m<nmodes; ++m) {
        if(mode_order[m] == mode) {
            mode_loc = m;
        }
    }
    /* Shift mode to moder_order[0] */
    if(mode_loc != 0) {
        for(sptIndex m=mode_loc; m>=1; --m) {
            mode_order[m] = mode_order[m-1];
        }
        mode_order[0] = mode;
    }

    free(sorted_ndims);
}


/**
 * Determine the worst mode order. Sort order: [(ordered by decreasing dimension sizes)]
 *
 * @param[out] mode_order a pointer to the array to be filled,
 * @param[in] mode mode to do product
 * @param[in] ndims tensor dimension sizes
 * @param[in] nmodes tensor order
 *
 */
void sptGetWorstModeOrder(
    sptIndex * mode_order,
    sptIndex const mode,
    sptIndex const * ndims,
    sptIndex const nmodes)
{
    sptKeyValuePair * sorted_ndims = (sptKeyValuePair*)malloc(nmodes * sizeof(*sorted_ndims));
    for(sptIndex m=0; m<nmodes; ++m) {
        sorted_ndims[m].key = m;
        sorted_ndims[m].value = ndims[m];
    }

    /* Increasingly sort */
    sptPairArraySort(sorted_ndims, nmodes);

    for(sptIndex m=0; m<nmodes; ++m) {
        mode_order[m] = sorted_ndims[nmodes - 1 - m].key;
    }

    /* Find the location of mode */
    sptIndex mode_loc = 0;
    for(sptIndex m=0; m<nmodes; ++m) {
        if(mode_order[m] == mode) {
            mode_loc = m;
        }
    }
    /* Shift mode to moder_order[0] */
    if(mode_loc != nmodes - 1) {
        for(sptIndex m=mode_loc; m<nmodes; ++m) {
            mode_order[m] = mode_order[m+1];
        }
        mode_order[nmodes - 1] = mode;
    }

    free(sorted_ndims);
}


/**
 * Sort COO sparse tensor by Z-Morton order. (The same with "sptPreprocessSparseTensor" function in "convert.c" without setting kschr.) 
 * Kernels in Row-major order, blocks and elements are in Z-Morton order.
 * @param tsr    a pointer to a sparse tensor
 * @return      mode pointers
 */
int sptSparseTensorMixedOrder(
    sptSparseTensor *tsr, 
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    int const tk)
{
    sptNnzIndex nnz = tsr->nnz;
    int result;

    /* Sort tsr in a Row-major Block order to get all kernels. Not use Morton-order for kernels: 1. better support for higher-order tensors by limiting kernel size, because Morton key bit <= 128; */
    sptSparseTensorSortIndexRowBlock(tsr, 1, 0, nnz, sk_bits, tk);

    sptNnzIndexVector kptr;
    result = sptNewNnzIndexVector(&kptr, 0, 0);
    spt_CheckError(result, "HiSpTns New", NULL);
    result = sptSetKernelPointers(&kptr, tsr, sk_bits);
    spt_CheckError(result, "HiSpTns Preprocess", NULL);

    /* Sort blocks in each kernel in Morton-order */
    sptNnzIndex k_begin, k_end;
    /* Loop for all kernels, 0-kptr.len for OMP code */
    for(sptNnzIndex k=0; k<kptr.len - 1; ++k) {
        k_begin = kptr.data[k];
        k_end = kptr.data[k+1];   // exclusive
        /* Sort blocks in each kernel in Morton-order */
        sptSparseTensorSortIndexMorton(tsr, 1, k_begin, k_end, sb_bits, tk);

    }

    return 0;
}



/**
 * Sort COO sparse tensor by plain blocked order for modes except mode-n. Blocks are in Row-major order.
 * @param tsr    a pointer to a sparse tensor
 * @return      mode pointers
 */
int sptSparseTensorSortPartialIndex(
    sptSparseTensor *tsr, 
    sptIndex const * mode_order,
    const sptElementIndex sb_bits,
    int const tk)
{
    sptNnzIndex nnz = tsr->nnz;
    sptIndex * ndims = tsr->ndims;
    sptIndex const mode = mode_order[0];
    int result;

    sptSparseTensorSortIndexCustomOrder(tsr, mode_order, 1, tk);

    sptNnzIndexVector sptr;
    result = sptNewNnzIndexVector(&sptr, 0, 0);
    spt_CheckError(result, "HiSpTns New", NULL);
    sptNnzIndex slice_nnz = 0;
    sptIndex pre_idx = tsr->inds[mode].data[0];
    result = sptAppendNnzIndexVector(&sptr, 0);
    for (sptNnzIndex z = 0; z < nnz; ++z ) {
        ++ slice_nnz;
        if (tsr->inds[mode].data[z] > pre_idx ) {
            result = sptAppendNnzIndexVector(&sptr, slice_nnz-1);
            pre_idx = tsr->inds[mode].data[z];
        }        
    }
    result = sptAppendNnzIndexVector(&sptr, nnz);
    sptDumpNnzIndexVector(&sptr, stdout);

    sptNnzIndex s_begin, s_end;
    // Loop for slices
    for(sptNnzIndex s = 0; s < ndims[mode]; ++ s) {
        s_begin = sptr.data[s];
        s_end = sptr.data[s+1];   // exclusive
        /* Sort blocks in each kernel in plain row-order */
        sptSparseTensorSortIndexRowBlock(tsr, 1, s_begin, s_end, sb_bits, tk);
    }

    return 0;
}



/**
 * Randomly shuffle all nonzeros.
 *
 * @param[in] tsr tensor to be shuffled
 *
 */
void sptGetRandomShuffleElements(sptSparseTensor *tsr) {
    sptNnzIndex const nnz = tsr->nnz;
    for(sptNnzIndex z=0; z<nnz; ++z) {
        srand(z+1);
        sptValue rand_val = (sptValue) rand() / (sptValue) RAND_MAX;
        sptNnzIndex new_loc = (sptNnzIndex) ( rand_val * nnz ) % nnz;
        spt_SwapValues(tsr, z, new_loc);
    }
    
}


/**
 * Randomly shuffle all indices.
 *
 * @param[in] tsr tensor to be shuffled
 * @param[out] map_inds records the randomly generated mapping
 *
 */
void sptGetRandomShuffledIndices(sptSparseTensor *tsr, sptIndex ** map_inds) {
    /* Get randomly renumbering indices */
    for(sptIndex m = 0; m < tsr->nmodes; ++m) {
        sptIndex dim_len = tsr->ndims[m];
        for(sptIndex i = dim_len - 1; i > 0; --i) {
            srand(m+i+1+time(NULL));
            sptIndex new_loc = (sptIndex) (rand() % (i+1));            
            /* Swap i <-> new_loc */
            sptIndex tmp = map_inds[m][i];
            map_inds[m][i] = map_inds[m][new_loc];
            map_inds[m][new_loc] = tmp;
        }
    }
}


/**
 * Reorder the elements in a COO sparse tensor lexicographically, sorting by Morton-order.
 * @param hitsr  the sparse tensor to operate on
 */
void sptSparseTensorSortIndexMorton(
    sptSparseTensor *tsr, 
    int force,
    const sptNnzIndex begin,
    const sptNnzIndex end,
    const sptElementIndex sb_bits,
    int tk) 
{
    size_t m;
    int needsort = 0;

    for(m = 0; m < tsr->nmodes; ++m) {
        if(tsr->sortorder[m] != m) {
            tsr->sortorder[m] = m;
            needsort = 1;
        }
    }

    if(needsort || force) {
        /* TODO: add support for other order tensors */
        switch(tsr->nmodes) {
            case 3:
                #pragma omp parallel num_threads(tk)
                {
                    #pragma omp single nowait
                    {
                        spt_QuickSortIndexMorton3D(tsr, begin, end, sb_bits);
                    }
                }
                break;
            case 4:
                #pragma omp parallel num_threads(tk)
                {
                    #pragma omp single nowait
                    {
                        spt_QuickSortIndexMorton4D(tsr, begin, end, sb_bits);
                    }
                }
                break;
            default:
                printf("No support for more than 4th-order tensors yet.\n");
        }
        
    }
}



/**
 * Reorder the elements in a COO sparse tensor lexicographically, sorting by row major order.
 * @param tsr  the sparse tensor to operate on
 */
void sptSparseTensorSortIndexRowBlock(
    sptSparseTensor *tsr, 
    int force,
    const sptNnzIndex begin,
    const sptNnzIndex end,
    const sptElementIndex sk_bits,
    int const tk) 
{
    size_t m;
    int needsort = 0;

    for(m = 0; m < tsr->nmodes; ++m) {
        if(tsr->sortorder[m] != m) {
            tsr->sortorder[m] = m;
            needsort = 1;
        }
    }
    if(needsort || force) {
        #pragma omp parallel num_threads(tk)
        {
            #pragma omp single nowait
            {
                spt_QuickSortIndexRowBlock(tsr, begin, end, sk_bits);
            }
        }
    }
}


/**
 * Reorder the elements in a sparse tensor lexicographically, sorting all modes except one. The except mode is NOT ordered.
 * @param tsr  the sparse tensor to operate on
 */
void sptSparseTensorSortIndexExceptSingleMode(sptSparseTensor *tsr, int force, sptIndex * mode_order, int const tk) {
    sptIndex m;
    int needsort = 0;

    for(m = 0; m < tsr->nmodes; ++m) {
        if(tsr->sortorder[m] != m) {
            tsr->sortorder[m] = m;
            needsort = 1;
        }
    }

    if(needsort || force) {
        #pragma omp parallel num_threads(tk) 
        {
            #pragma omp single nowait 
            {
                spt_QuickSortIndexExceptSingleMode(tsr, 0, tsr->nnz, mode_order);
            }
        }
    }
}



/**
 * Reorder the elements in a sparse tensor lexicographically in a customized order.
 * @param tsr  the sparse tensor to operate on
 */
void sptSparseTensorSortIndexCustomOrder(sptSparseTensor *tsr, sptIndex const *  mode_order, int force, int tk) {
    sptIndex nmodes = tsr->nmodes;
    sptIndex m;
    sptSparseTensor tsr_temp; // Only copy pointers, not real data.

    if(!force && memcmp(tsr->sortorder, mode_order, nmodes * sizeof (sptIndex)) == 0) {
        return;
    }

    tsr_temp.nmodes = nmodes;
    tsr_temp.sortorder = tsr->sortorder;
    tsr_temp.ndims = malloc(nmodes * sizeof tsr_temp.ndims[0]);
    tsr_temp.nnz = tsr->nnz;
    tsr_temp.inds = malloc(nmodes * sizeof tsr_temp.inds[0]);
    tsr_temp.values = tsr->values;

    for(m = 0; m < nmodes; ++m) {
        tsr_temp.ndims[m] = tsr->ndims[mode_order[m]];
        tsr_temp.inds[m] = tsr->inds[mode_order[m]];
    }

    sptSparseTensorSortIndex(&tsr_temp, 1, tk);

    free(tsr_temp.inds);
    free(tsr_temp.ndims);

    for(m = 0; m < nmodes; ++m) {
        tsr->sortorder[m] = mode_order[m];
    }
}


/**
 * Reorder the elements in a sparse tensor lexicographically
 * @param tsr  the sparse tensor to operate on
 */
void sptSparseTensorSortIndex(sptSparseTensor *tsr, int force, int tk)
{
    sptIndex m;
    int needsort = 0;

    for(m = 0; m < tsr->nmodes; ++m) {
        if(tsr->sortorder[m] != m) {
            tsr->sortorder[m] = m;
            needsort = 1;
        }
    }

    if(needsort || force) {
        #pragma omp parallel num_threads(tk)
        {    
            #pragma omp single nowait
            {
                spt_QuickSortIndex(tsr, 0, tsr->nnz);
            }
        }
    }
}


/**
 * Reorder the elements in a sparse tensor lexicographically, but consider mode `mode` the last one
 * @param tsr  the sparse tensor to operate on
 * @param mode the mode to be considered the last
 */
void sptSparseTensorSortIndexAtMode(sptSparseTensor *tsr, sptIndex const mode, int force, int tk) {
    sptIndex m;
    int needsort = 0;

    for(m = 0; m < mode; ++m) {
        if(tsr->sortorder[m] != m) {
            tsr->sortorder[m] = m;
            needsort = 1;
        }
    }
    for(m = mode+1; m < tsr->nmodes; ++m) {
        if(tsr->sortorder[m-1] != m) {
            tsr->sortorder[m-1] = m;
            needsort = 1;
        }
    }
    if(tsr->sortorder[tsr->nmodes-1] != mode) {
        tsr->sortorder[tsr->nmodes-1] = mode;
        needsort = 1;
    }

    if(needsort || force) {
        #pragma omp parallel num_threads(tk)
        {    
            #pragma omp single nowait
            {
                spt_QuickSortAtMode(tsr, 0, tsr->nnz, mode);
            }
        }
    }
}


/**
 * compare two indices from two identical or distinct sparse tensors lexicographically
 * @param tsr1 the first sparse tensor
 * @param loc1 the order of the element in the first sparse tensor whose index is to be compared
 * @param tsr2 the second sparse tensor
 * @param loc2 the order of the element in the second sparse tensor whose index is to be compared
 * @return -1 for less, 0 for equal, 1 for greater
 */
int spt_SparseTensorCompareIndices(const sptSparseTensor *tsr1, sptNnzIndex loc1, const sptSparseTensor *tsr2, sptNnzIndex loc2) {
    sptIndex i;
    assert(tsr1->nmodes == tsr2->nmodes);
    for(i = 0; i < tsr1->nmodes; ++i) {
        sptIndex eleind1 = tsr1->inds[i].data[loc1];
        sptIndex eleind2 = tsr2->inds[i].data[loc2];
        if(eleind1 < eleind2) {
            return -1;
        } else if(eleind1 > eleind2) {
            return 1;
        }
    }
    return 0;
}


/*************************************************
 * PRIVATE FUNCTIONS
 *************************************************/
/**
 * compare two indices from two identical or distinct sparse tensors lexicographically in all modes except mode
 * @param tsr1 the first sparse tensor
 * @param loc1 the order of the element in the first sparse tensor whose index is to be compared
 * @param tsr2 the second sparse tensor
 * @param loc2 the order of the element in the second sparse tensor whose index is to be compared
 * @param mode the mode to be excluded in comparison
 * @return -1 for less, 0 for equal, 1 for greater
 */

/*************************************************
 * Comparison functions
 *************************************************/

static int spt_SparseTensorCompareAtMode(const sptSparseTensor *tsr1, sptNnzIndex const ind1, const sptSparseTensor *tsr2, sptNnzIndex const ind2, sptIndex const mode) {
    sptIndex i;
    sptIndex eleind1, eleind2;
    assert(tsr1->nmodes == tsr2->nmodes);
    for(i = 0; i < tsr1->nmodes; ++i) {
        if(i != mode) {
            eleind1 = tsr1->inds[i].data[ind1];
            eleind2 = tsr2->inds[i].data[ind2];
            if(eleind1 < eleind2) {
                return -1;
            } else if(eleind1 > eleind2) {
                return 1;
            }
        }
    }
    eleind1 = tsr1->inds[mode].data[ind1];
    eleind2 = tsr2->inds[mode].data[ind2];
    if(eleind1 < eleind2) {
        return -1;
    } else if(eleind1 > eleind2) {
        return 1;
    } else {
        return 0;
    }
}

int spt_SparseTensorCompareIndicesExceptSingleMode(const sptSparseTensor *tsr1, sptNnzIndex loc1, const sptSparseTensor *tsr2, sptNnzIndex loc2, sptIndex * mode_order) {
    sptIndex i, m;
    assert(tsr1->nmodes == tsr2->nmodes);
    for(i = 0; i < tsr1->nmodes - 1; ++ i) {
        m = mode_order[i];
        sptIndex eleind1 = tsr1->inds[m].data[loc1];
        sptIndex eleind2 = tsr2->inds[m].data[loc2];
        if(eleind1 < eleind2) {
            return -1;
        } else if(eleind1 > eleind2) {
            return 1;
        }
    }
    return 0;
}


/**
 * compare two indices from two identical or distinct sparse tensors lexicographically, using block index as keywords.
 * @param tsr1 the first sparse tensor
 * @param loc1 the order of the element in the first sparse tensor whose index is to be compared
 * @param tsr2 the second sparse tensor
 * @param loc2 the order of the element in the second sparse tensor whose index is to be compared
 * @return -1 for less, 0 for equal, 1 for greater
 */
static int spt_SparseTensorCompareIndicesRowBlock(
    const sptSparseTensor *tsr1, 
    sptNnzIndex loc1, 
    const sptSparseTensor *tsr2, 
    sptNnzIndex loc2,
    const sptElementIndex sk_bits) 
{
    sptIndex i;
    assert(tsr1->nmodes == tsr2->nmodes);

    for(i = 0; i < tsr1->nmodes; ++i) {
        sptIndex eleind1 = tsr1->inds[i].data[loc1];
        sptIndex eleind2 = tsr2->inds[i].data[loc2];
        sptIndex blkind1 = eleind1 >> sk_bits;
        sptIndex blkind2 = eleind2 >> sk_bits;

        if(blkind1 < blkind2) {
            return -1;
        } else if(blkind1 > blkind2) {
            return 1;
        }
    }
    return 0;
}


/**
 * compare two indices from two identical or distinct sparse tensors lexicographically, using Z-Morton ordering recursively, freely support 3-D, 4-D for uint32_t indices. 
 * When tensor order is large than 5, index ranges are limited.
 * @param tsr1 the first sparse tensor
 * @param loc1 the order of the element in the first sparse tensor whose index is to be compared
 * @param tsr2 the second sparse tensor
 * @param loc2 the order of the element in the second sparse tensor whose index is to be compared
 * @return -1 for less, 0 for equal, 1 for greater
 */
static int spt_SparseTensorCompareIndicesMorton3D(
    const sptSparseTensor *tsr1, 
    uint64_t loc1, 
    const sptSparseTensor *tsr2, 
    uint64_t loc2) 
{
    sptMortonIndex mkey1 = 0, mkey2 = 0;
    assert(tsr1->nmodes == tsr2->nmodes);

    /* Only support 3-D tensors, with 32-bit indices. */
    uint32_t x1 = tsr1->inds[0].data[loc1];
    uint32_t y1 = tsr1->inds[1].data[loc1];
    uint32_t z1 = tsr1->inds[2].data[loc1];
    uint32_t x2 = tsr2->inds[0].data[loc2];
    uint32_t y2 = tsr2->inds[1].data[loc2];
    uint32_t z2 = tsr2->inds[2].data[loc2];

    mkey1 =    morton256_z[(z1 >> 24) & 0xFF ] |
                morton256_y[(y1 >> 24) & 0xFF ] |
                morton256_x[(x1 >> 24) & 0xFF ];
    mkey1 =    mkey1 << 72 |
                morton256_z[(z1 >> 16) & 0xFF ] |
                morton256_y[(y1 >> 16) & 0xFF ] |
                morton256_x[(x1 >> 16) & 0xFF ];
    mkey1 =    mkey1 << 48 |
                morton256_z[(z1 >> 8) & 0xFF ] |
                morton256_y[(y1 >> 8) & 0xFF ] |
                morton256_x[(x1 >> 8) & 0xFF ];
    mkey1 =    mkey1 << 24 |
                morton256_z[(z1) & 0xFF ] |
                morton256_y[(y1) & 0xFF ] |
                morton256_x[(x1) & 0xFF ];

    mkey2 =    morton256_z[(z2 >> 24) & 0xFF ] |
                morton256_y[(y2 >> 24) & 0xFF ] |
                morton256_x[(x2 >> 24) & 0xFF ];
    mkey2 =    mkey2 << 72 |
                morton256_z[(z2 >> 16) & 0xFF ] |
                morton256_y[(y2 >> 16) & 0xFF ] |
                morton256_x[(x2 >> 16) & 0xFF ];
    mkey2 =    mkey2 << 48 |
                morton256_z[(z2 >> 8) & 0xFF ] |
                morton256_y[(y2 >> 8) & 0xFF ] |
                morton256_x[(x2 >> 8) & 0xFF ];
    mkey2 =    mkey2 << 24 |
                morton256_z[(z2) & 0xFF ] |
                morton256_y[(y2) & 0xFF ] |
                morton256_x[(x2) & 0xFF ];

    if(mkey1 < mkey2) {
        return -1;
    } else if(mkey1 > mkey2) {
        return 1;
    } else {
        return 0;
    }
    
}


/**
 * compare two indices from two identical or distinct sparse tensors lexicographically, using Z-Morton ordering recursively, freely support arbitrary tensor orders.
 * @param tsr1 the first sparse tensor
 * @param loc1 the order of the element in the first sparse tensor whose index is to be compared
 * @param tsr2 the second sparse tensor
 * @param loc2 the order of the element in the second sparse tensor whose index is to be compared
 * @return -1 for less, 0 for equal, 1 for greater
 */
static int spt_SparseTensorCompareIndicesMorton4D(
    const sptSparseTensor *tsr1, 
    uint64_t loc1, 
    const sptSparseTensor *tsr2, 
    uint64_t loc2) 
{
    sptMortonIndex mkey1, mkey2;
    assert(tsr1->nmodes == tsr2->nmodes);

    /* Only support 3-D tensors, with 32-bit indices. */
    uint32_t x1 = tsr1->inds[0].data[loc1];
    uint32_t y1 = tsr1->inds[1].data[loc1];
    uint32_t z1 = tsr1->inds[2].data[loc1];
    uint32_t w1 = tsr1->inds[3].data[loc1];
    uint32_t x2 = tsr2->inds[0].data[loc2];
    uint32_t y2 = tsr2->inds[1].data[loc2];
    uint32_t z2 = tsr2->inds[2].data[loc2];
    uint32_t w2 = tsr2->inds[3].data[loc2];

    static const uint64_t MASKS_64[]={0x5555555555555555, 0x3333333333333333, 0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF, 0x0000FFFF0000FFFF};
    static const uint64_t SHIFTS_64[]= {1, 2, 4, 8, 16};
    static sptMortonIndex MASKS_128[] = {
        (sptMortonIndex)0x5555555555555555 << 64 | 0x5555555555555555, 
        (sptMortonIndex)0x3333333333333333 << 64 | 0x3333333333333333, 
        (sptMortonIndex)0x0F0F0F0F0F0F0F0F << 64 | 0x0F0F0F0F0F0F0F0F, 
        (sptMortonIndex)0x00FF00FF00FF00FF << 64 | 0x00FF00FF00FF00FF, 
        (sptMortonIndex)0x0000FFFF0000FFFF << 64 | 0x0000FFFF0000FFFF, 
        (sptMortonIndex)0x00000000FFFFFFFF << 64 | 0x00000000FFFFFFFF};
    static const uint64_t SHIFTS_128[]= {1, 2, 4, 8, 16, 32};
    // sptMortonIndex tmp_mask = MASKS_128[2];
    // printf("tmp_mask: high: %"PRIX64 " ; low: %"PRIX64 " .\n", (uint64_t)(tmp_mask >> 64), (uint64_t)tmp_mask);

    uint64_t tmp_64;
    sptMortonIndex x, y, z, w;
    
    /**** compute mkey1 ****/
    /* compute correct x, 32bit -> 64bit first */
    tmp_64 = x1;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];
    /* compute correct x, 64bit -> 128bit */
    x = tmp_64;
    x = (x | (x << SHIFTS_128[5])) & MASKS_128[5];
    x = (x | (x << SHIFTS_128[4])) & MASKS_128[4];
    x = (x | (x << SHIFTS_128[3])) & MASKS_128[3];
    x = (x | (x << SHIFTS_128[2])) & MASKS_128[2];
    x = (x | (x << SHIFTS_128[1])) & MASKS_128[1];
    x = (x | (x << SHIFTS_128[0])) & MASKS_128[0];

    /* compute correct y, 32bit -> 64bit first */
    tmp_64 = y1;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];
    /* compute correct y, 64bit -> 128bit */
    y = tmp_64;
    y = (y | (y << SHIFTS_128[5])) & MASKS_128[5];
    y = (y | (y << SHIFTS_128[4])) & MASKS_128[4];
    y = (y | (y << SHIFTS_128[3])) & MASKS_128[3];
    y = (y | (y << SHIFTS_128[2])) & MASKS_128[2];
    y = (y | (y << SHIFTS_128[1])) & MASKS_128[1];
    y = (y | (y << SHIFTS_128[0])) & MASKS_128[0];

    /* compute correct z, 32bit -> 64bit first */
    tmp_64 = z1;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];
    /* compute correct z, 64bit -> 128bit */
    z = tmp_64;
    z = (z | (z << SHIFTS_128[5])) & MASKS_128[5];
    z = (z | (z << SHIFTS_128[4])) & MASKS_128[4];
    z = (z | (z << SHIFTS_128[3])) & MASKS_128[3];
    z = (z | (z << SHIFTS_128[2])) & MASKS_128[2];
    z = (z | (z << SHIFTS_128[1])) & MASKS_128[1];
    z = (z | (z << SHIFTS_128[0])) & MASKS_128[0];

    /* compute correct w, 32bit -> 64bit first */
    tmp_64 = w1;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];
    /* compute correct w, 64bit -> 128bit */
    w = tmp_64;
    w = (w | (w << SHIFTS_128[5])) & MASKS_128[5];
    w = (w | (w << SHIFTS_128[4])) & MASKS_128[4];
    w = (w | (w << SHIFTS_128[3])) & MASKS_128[3];
    w = (w | (w << SHIFTS_128[2])) & MASKS_128[2];
    w = (w | (w << SHIFTS_128[1])) & MASKS_128[1];
    w = (w | (w << SHIFTS_128[0])) & MASKS_128[0];

    // mkey1 = x | (y << 1) | (z << 2) | (w << 3);
    mkey1 = w | (z << 1) | (y << 2) | (x << 3);


    /**** compute mkey2 ****/
    /* compute correct x, 32bit -> 64bit first */
    tmp_64 = x2;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];
    /* compute correct x, 64bit -> 128bit */
    x = tmp_64;
    x = (x | (x << SHIFTS_128[5])) & MASKS_128[5];
    x = (x | (x << SHIFTS_128[4])) & MASKS_128[4];
    x = (x | (x << SHIFTS_128[3])) & MASKS_128[3];
    x = (x | (x << SHIFTS_128[2])) & MASKS_128[2];
    x = (x | (x << SHIFTS_128[1])) & MASKS_128[1];
    x = (x | (x << SHIFTS_128[0])) & MASKS_128[0];

    /* compute correct y, 32bit -> 64bit first */
    tmp_64 = y2;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];
    /* compute correct y, 64bit -> 128bit */
    y = tmp_64;
    y = (y | (y << SHIFTS_128[5])) & MASKS_128[5];
    y = (y | (y << SHIFTS_128[4])) & MASKS_128[4];
    y = (y | (y << SHIFTS_128[3])) & MASKS_128[3];
    y = (y | (y << SHIFTS_128[2])) & MASKS_128[2];
    y = (y | (y << SHIFTS_128[1])) & MASKS_128[1];
    y = (y | (y << SHIFTS_128[0])) & MASKS_128[0];

    /* compute correct z, 32bit -> 64bit first */
    tmp_64 = z2;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];
    /* compute correct z, 64bit -> 128bit */
    z = tmp_64;
    z = (z | (z << SHIFTS_128[5])) & MASKS_128[5];
    z = (z | (z << SHIFTS_128[4])) & MASKS_128[4];
    z = (z | (z << SHIFTS_128[3])) & MASKS_128[3];
    z = (z | (z << SHIFTS_128[2])) & MASKS_128[2];
    z = (z | (z << SHIFTS_128[1])) & MASKS_128[1];
    z = (z | (z << SHIFTS_128[0])) & MASKS_128[0];

    /* compute correct w, 32bit -> 64bit first */
    tmp_64 = w2;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];
    /* compute correct w, 64bit -> 128bit */
    w = tmp_64;
    w = (w | (w << SHIFTS_128[5])) & MASKS_128[5];
    w = (w | (w << SHIFTS_128[4])) & MASKS_128[4];
    w = (w | (w << SHIFTS_128[3])) & MASKS_128[3];
    w = (w | (w << SHIFTS_128[2])) & MASKS_128[2];
    w = (w | (w << SHIFTS_128[1])) & MASKS_128[1];
    w = (w | (w << SHIFTS_128[0])) & MASKS_128[0];

    mkey2 = w | (z << 1) | (y << 2) | (x << 3);

    if(mkey1 < mkey2) {
        return -1;
    } else if(mkey1 > mkey2) {
        return 1;
    } else {
        return 0;
    }
    
}


/*************************************************
 * Quicksort functions
 *************************************************/

static void spt_QuickSortAtMode(sptSparseTensor *tsr, sptNnzIndex const l, sptNnzIndex const r, sptIndex const mode) {
    sptNnzIndex i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(spt_SparseTensorCompareAtMode(tsr, i, tsr, p, mode) < 0) {
            ++i;
        }
        while(spt_SparseTensorCompareAtMode(tsr, p, tsr, j, mode) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        spt_SwapValues(tsr, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    #pragma omp task firstprivate(l,i) shared(tsr)
    {
        spt_QuickSortAtMode(tsr, l, i, mode);
    }
    spt_QuickSortAtMode(tsr, i, r, mode);
    #pragma omp taskwait
}


static void spt_QuickSortIndexMorton3D(sptSparseTensor *tsr, sptNnzIndex l, sptNnzIndex r, const sptElementIndex sb_bits) {

    uint64_t i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(spt_SparseTensorCompareIndicesMorton3D(tsr, i, tsr, p) < 0) {
            ++i;
        }
        while(spt_SparseTensorCompareIndicesMorton3D(tsr, p, tsr, j) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        spt_SwapValues(tsr, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    #pragma omp task firstprivate(l,i) shared(tsr)
    {
        spt_QuickSortIndexMorton3D(tsr, l, i, sb_bits);
    }
    spt_QuickSortIndexMorton3D(tsr, i, r, sb_bits);
    #pragma omp taskwait
}


static void spt_QuickSortIndexMorton4D(sptSparseTensor *tsr, sptNnzIndex l, sptNnzIndex r, const sptElementIndex sb_bits) {

    uint64_t i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(spt_SparseTensorCompareIndicesMorton4D(tsr, i, tsr, p) < 0) {
            ++i;
        }
        while(spt_SparseTensorCompareIndicesMorton4D(tsr, p, tsr, j) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        spt_SwapValues(tsr, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    #pragma omp task firstprivate(l,i) shared(tsr)
    {
        spt_QuickSortIndexMorton4D(tsr, l, i, sb_bits);
    }
    spt_QuickSortIndexMorton4D(tsr, i, r, sb_bits);
    #pragma omp taskwait
}


static void spt_QuickSortIndexRowBlock(sptSparseTensor *tsr, sptNnzIndex l, sptNnzIndex r, const sptElementIndex sk_bits) {

    sptNnzIndex i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(spt_SparseTensorCompareIndicesRowBlock(tsr, i, tsr, p, sk_bits) < 0) {
            ++i;
        }
        while(spt_SparseTensorCompareIndicesRowBlock(tsr, p, tsr, j, sk_bits) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        spt_SwapValues(tsr, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    #pragma omp task firstprivate(l,i) shared(tsr)
    {
        spt_QuickSortIndexRowBlock(tsr, l, i, sk_bits);
    }
    spt_QuickSortIndexRowBlock(tsr, i, r, sk_bits);
    #pragma omp taskwait
}


static void spt_QuickSortIndexExceptSingleMode(sptSparseTensor *tsr, sptNnzIndex l, sptNnzIndex r, sptIndex * mode_order) 
{
    sptNnzIndex i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(spt_SparseTensorCompareIndicesExceptSingleMode(tsr, i, tsr, p, mode_order) < 0) {
            ++i;
        }
        while(spt_SparseTensorCompareIndicesExceptSingleMode(tsr, p, tsr, j, mode_order) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        spt_SwapValues(tsr, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    #pragma omp task firstprivate(l,i) shared(tsr, mode_order)
    {
        spt_QuickSortIndexExceptSingleMode(tsr, l, i, mode_order);
    }
    spt_QuickSortIndexExceptSingleMode(tsr, i, r, mode_order);
    #pragma omp taskwait
}


static void spt_QuickSortIndex(sptSparseTensor *tsr, sptNnzIndex l, sptNnzIndex r) {
    sptNnzIndex i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(spt_SparseTensorCompareIndices(tsr, i, tsr, p) < 0) {
            ++i;
        }
        while(spt_SparseTensorCompareIndices(tsr, p, tsr, j) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        spt_SwapValues(tsr, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    #pragma omp task firstprivate(l,i) shared(tsr)
    {
        spt_QuickSortIndex(tsr, l, i);
    }
    spt_QuickSortIndex(tsr, i, r);
    #pragma omp taskwait
}

