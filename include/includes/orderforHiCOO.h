#ifndef ORDERFORHICOO_H_
#define ORDERFORHICOO_H_

#include <stdarg.h>
#include <fcntl.h>
#include "types.h"

typedef sptIndex idxType;

typedef struct{
    idxType nvrt; /* number of vertices. This nvrt = n_0 + n_1 + ... + n_{d-1} for a d-dimensional tensor
                     where the ith dimension is of size n_i.*/
    idxType *vptrs, *vHids; /*starts of hedges containing vertices, and the ids of the hedges*/
    
    idxType nhdg; /*this will be equal to the number of nonzeros in the tensor*/
    idxType *hptrs, *hVids; /*starts of vertices in the hedges, and the ids of the vertices*/
} basicHypergraph;


void allocateHypergraphData(basicHypergraph *hg, idxType nvrt, idxType nhdg, idxType npins);
void freeHypergraphData(basicHypergraph *hg);
void fillHypergraphFromCoo(basicHypergraph *hg, int N, idxType nnz, idxType *dims, sptIndexVector *coord);
void setVList(basicHypergraph *hg);

/*heap functions.
    we keep ids in locations heapIds[1...sz]
 */
/*key is the key value, vptrs will be used to use the size of the vertex as the second key value.
 for key:
    the larger the more priority; and when ties occur, the smaller the vertex size, the more the priority.
 */
 
void heapInsert(idxType *heapIds, idxType *key, idxType *vptrs, idxType *sz, idxType id, idxType *inheap);
void heapify(idxType *heapIds, idxType *key, idxType *vptrs, idxType sz, idxType i,  idxType *inheap);
idxType heapExtractMax(idxType *heapIds, idxType *key, idxType *vptrs, idxType *sz, idxType *inheap);
void heapBuild(idxType *heapIds, idxType *key, idxType *vptrs, idxType sz, idxType *inheap);
void heapVerify(idxType *heapIds, idxType *key, idxType *vptrs, idxType sz, idxType *inheap);

void heapIncreaseKey(idxType *heapIds, idxType *key, idxType *vptrs, idxType sz, idxType id, idxType *inheap, idxType newKey);
void bu_errexit(char * f_str,...);

void orderforHiCOO(int N, idxType nnz, idxType *dims, sptIndexVector *coord, idxType **newIndices_out);

#endif
