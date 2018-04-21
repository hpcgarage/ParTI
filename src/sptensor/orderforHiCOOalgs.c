#include <stdio.h>
#include <stdlib.h>

#include "ParTI.h"

inline static idxType locateVertex(idxType indStart, idxType indEnd, idxType *lst, idxType sz)
{
    idxType i;
    for (i = 0; i < sz; i++)
        if(lst[i] >= indStart && lst[i] <= indEnd)
            return lst[i];
    
    bu_errexit("could not locate in a hyperedge !!!\n");
    return -1;
}

void orderforHiCOOaDim(basicHypergraph *hg, idxType *newIndicesHg, idxType indStart, idxType indEnd)
{
    /* we re-order the vertices of the hypergraph with ids in the range [indStart, indEnd]*/
    
    idxType i, v, j, jj, k, w, ww, hedge, hedge2;
    idxType *inHeap, *heapIds, heapSz;
    idxType *vptrs = hg->vptrs, *vHids = hg->vHids, *hptrs = hg->hptrs, *hVids = hg->hVids;
    
    idxType *keyvals, newKeyval;
    int *markers, mark;
    
    mark = 0;
    
    heapIds = (idxType*) malloc(sizeof(idxType) * (indEnd-indStart + 2));
    inHeap = (idxType*) malloc(sizeof(idxType) * hg->nvrt);/*this is large*/
    keyvals = (idxType *) malloc(sizeof(idxType) * hg->nvrt);
    markers = (int*) malloc(sizeof(int)* hg->nvrt);
    
    heapSz = 0;
    
    for (i = indStart; i<=indEnd; i++)
    {
        keyvals[i] = 0;
        heapIds[++heapSz] = i;
        inHeap[i] = heapSz;
        markers[i] = -1;
    }
    heapBuild(heapIds, keyvals, vptrs, heapSz, inHeap);
    
    for (i = indStart; i <= indEnd; i++)
    {
        v = heapExtractMax(heapIds, keyvals, vptrs, &heapSz, inHeap);
        newIndicesHg[v] = i;
        markers[v] = mark;
        for (j = vptrs[v]; j < vptrs[v+1]; j++)
        {
            hedge = vHids[j];
            for (k = hptrs[hedge]; k < hptrs[hedge+1]; k++)
            {
                w = hVids[k];
                if (markers[w] != mark)
                {
                    markers[w] = mark;
                    for(jj = vptrs[w]; jj < vptrs[w+1]; jj++)
                    {
                        hedge2 = vHids[jj];
                        ww = locateVertex(indStart, indEnd, hVids + hptrs[hedge2], hptrs[hedge2+1]-hptrs[hedge2]);
                        if(inHeap[ww]) {
                            newKeyval = keyvals[ww] + 1;
                            heapIncreaseKey(heapIds, keyvals, vptrs, heapSz, ww, inHeap, newKeyval);
                        }
                    }
                }
            }
        }
    }
    
    free(markers);
    free(keyvals);
    free(inHeap);
    free(heapIds);
}

void orderforHiCOO(int N, idxType nnz, idxType *dims, sptIndexVector *coord, idxType **newIndices_out)
{
    /*PRE: newIndices is allocated
     
     POST:
     newIndices[0][0...n_0-1] gives the new ids for dim 0
     newIndices[1][0...n_1-1] gives the new ids for dim 1
     ...
     newIndices[d-1][0...n_{d-1}-1] gives the new ids for dim d-1

     This implements a simple idea close to BFS/Cuthill-McKee/Maximum cardinality search.
     */
    int d, i;
    idxType *dimsPrefixSum;
    
    basicHypergraph hg;
    
    idxType *newIndicesHg;
    
    dimsPrefixSum = (idxType*) calloc(N, sizeof(idxType));
    for (d = 1; d < N; d++)
        dimsPrefixSum[d] = dims[d-1] + dimsPrefixSum[d-1];
    
    fillHypergraphFromCoo(&hg, N,  nnz, dims, coord);
    newIndicesHg = (idxType*) malloc(sizeof(idxType) * hg.nvrt);
    
    for (i = 0; i < hg.nvrt; i++)
        newIndicesHg[i] = i;
    
    for (d = 0; d < N; d++) /*assume all others fixed and order for d*/
    {
        // printf("ordering %d. Indices: [%ld %ld]\n", d, dimsPrefixSum[d], dimsPrefixSum[d] + dims[d]-1);
        orderforHiCOOaDim(&hg, newIndicesHg, dimsPrefixSum[d], dimsPrefixSum[d] + dims[d]-1);
    }
    /*copy from newIndices to newIndicesOut*/
    for (d = 0; d < N; d++)
        for (i = 0; i < dims[d]; i++)
            newIndices_out[d][i] = newIndicesHg[dimsPrefixSum[d] + i] - dimsPrefixSum[d];
    
    free(newIndicesHg);
    freeHypergraphData(&hg);
    free(dimsPrefixSum);
    
}
