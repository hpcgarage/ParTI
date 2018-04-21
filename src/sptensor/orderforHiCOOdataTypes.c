#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <fcntl.h>

#include "ParTI.h"

void bu_errexit(char * f_str,...)
{
    /*copied from Umit's*/
    va_list argp;
    
    fflush(stdout);
    fflush(stderr);
    fprintf(stderr, "\n****** Error ******\n");
    va_start(argp, f_str);
    vfprintf(stderr, f_str, argp);
    va_end(argp);
    
    fprintf(stderr,"*******************\n");
    fflush(stderr);
    
    printf("Error in Execution\n");
    exit(12);
}


/**********************************************************************************************/
void allocateHypergraphData(basicHypergraph *hg, idxType nvrt, idxType nhdg, idxType npins)
{
    hg->nvrt = nvrt;
    hg->vptrs = (idxType *) malloc(sizeof(idxType) * (nvrt+1));
    hg->vHids = (idxType *) malloc(sizeof(idxType) * npins);
    
    hg->nhdg = nhdg;
    hg->hptrs = (idxType *) malloc(sizeof(idxType) * (nhdg+1));
    hg->hVids = (idxType *) malloc(sizeof(idxType) * npins);
}
void freeHypergraphData(basicHypergraph *hg)
{
    hg->nvrt = 0;
    if (hg->vptrs) free(hg->vptrs);
    if (hg->vHids) free(hg->vHids);
    
    hg->nhdg = 0;
    if (hg->hptrs) free(hg->hptrs);
    if (hg->hVids) free(hg->hVids);
}

/**********************************************************************************************/
void setVList(basicHypergraph *hg)
{
    /*PRE: We assume hg->hptrs and hg->hVids are set; hg->nvrts is set, and
     hg->vptrs and hg->vHids are allocated appropriately.
     */
    
    idxType j, h, v, nvrt = hg->nvrt, nhdg = hg->nhdg;
    
    /*vertices */
    idxType *vptrs = hg->vptrs, *vHids = hg->vHids;
    /*hyperedges*/
    idxType *hptrs = hg->hptrs, *hVids = hg->hVids;
    
    for (v = 0; v <= nvrt; v++)
        vptrs[v] = 0;
    
    for (h = 0; h < nhdg; h++)
    {
        for (j = hptrs[h]; j < hptrs[h+1]; j++)
        {
            v = hVids[j];
            vptrs[v] ++;
        }
    }
    for (v=1; v <= nvrt; v++)
        vptrs[v] += vptrs[v-1];
    
    for (h = nhdg-1; h >= 0; h--)
    {
        for (j = hptrs[h]; j < hptrs[h+1]; j++)
        {
            v = hVids[j];
            vHids[--(vptrs[v])] = h;
        }
    }
}
/**********************************************************************************************/
void fillHypergraphFromCoo(basicHypergraph *hg, int N, idxType nnz, idxType *dims, sptIndexVector *coord)
{
    
    idxType h, totalSizes, toAddress;
    idxType *dimSizesPrefixSum;
    
    int i;
    
    dimSizesPrefixSum = (idxType *) malloc(sizeof(idxType) * (N+1));
    totalSizes = 0;
    for (i=0; i < N; i++)
    {
        dimSizesPrefixSum[i] = totalSizes;
        totalSizes += dims[i];
    }
    printf("allocating hyp %d %u\n", N, nnz);
    
    allocateHypergraphData(hg, totalSizes, nnz, nnz * N);
    
    toAddress = 0;
    for (h = 0; h < nnz; h++)
    {
        hg->hptrs[h] = toAddress;
        for (i = 0;  i < N; i++)
            hg->hVids[toAddress + i] = dimSizesPrefixSum[i] + coord[i].data[h];
        toAddress += N;
    }
    hg->hptrs[hg->nhdg] = toAddress;
    
    setVList(hg);
    free(dimSizesPrefixSum);
}
/**********************************************************************************************/
void heapBuild(idxType *heapIds, idxType *key, idxType *vptrs, idxType sz, idxType *inheap)
{
    idxType i;
    for (i=sz/2; i>=1; i--)
        heapify(heapIds, key, vptrs, sz, i, inheap);
}

idxType heapExtractMax(idxType *heapIds, idxType *key, idxType *vptrs, idxType *sz, idxType *inheap)
{
    idxType maxind ;
    if (*sz < 1)
        bu_errexit("heap underflow\n");
    
    maxind = heapIds[1];
    heapIds[1] = heapIds[*sz];
    inheap[heapIds[1]] = 1;
    
    *sz = *sz - 1;
    inheap[maxind] = 0;
    
    heapify(heapIds, key, vptrs, *sz, 1, inheap);
    return maxind;
    
}
#define SIZEV( vid ) vptrs[(vid)+1]-vptrs[(vid)]

void heapIncreaseKey(idxType *heapIds, idxType *key, idxType *vptrs, idxType sz, idxType id, idxType *inheap, idxType newKey)
{
    idxType i = inheap[id]; /*location in heap*/
    key[id] = newKey;
    
    while ((i>>1)>0 && ( (key[id] > key[heapIds[i>>1]]) ||
                         (key[id] == key[heapIds[i>>1]] && SIZEV(id) > SIZEV(heapIds[i>>1])))
          )
    {
        heapIds[i] = heapIds[i>>1];
        inheap[heapIds[i]] = i;
        i = i>>1;
    }
    heapIds[i] = id;
    inheap[id] = i;
}

void heapInsert(idxType *heapIds, idxType *key, idxType *vptrs, idxType *sz, idxType id, idxType *inheap)
{
    idxType j, kv = key[id], kv2 = SIZEV(id);
    *sz = *sz+1;
    heapIds[*sz] = id;
    j = *sz;
    if(inheap[id] != 0)
        bu_errexit("heapInsert: inserting id %d but already in (index %d)\n", id, inheap[id]);
    while(j > 1 && ((key[heapIds[j/2]] < kv) ||
                    (key[heapIds[j/2]] == kv && kv2 > SIZEV(heapIds[j/2]))
                    )
          )
    {
        heapIds[j] = heapIds[j/2] ;
        inheap[heapIds[j/2]] = j;
        j = j/2;
    }
    heapIds[j] = id;
    inheap[id] = j;
}

void heapify(idxType *heapIds, idxType *key, idxType *vptrs, idxType sz, idxType i,  idxType *inheap)
{
    idxType largest, j, l,r, tmp;
    
    largest = j = i;
    while(j<=sz/2)
    {
        l = 2*j;
        r = 2*j + 1;
        
        if ( (key[heapIds[l]] > key[heapIds[j]] ) ||
             (key[heapIds[l]] == key[heapIds[j]]  && SIZEV(heapIds[l]) < SIZEV(heapIds[j]) )
            )
            largest = l;
        else
            largest = j;
        
        if (r<=sz && (key[heapIds[r]]>key[heapIds[largest]] ||
                     (key[heapIds[r]]==key[heapIds[largest]] && SIZEV(heapIds[r]) < SIZEV(heapIds[largest])))
            )
            largest = r;
        
        if (largest != j)
        {
            tmp = heapIds[largest];
            heapIds[largest] = heapIds[j];
            inheap[heapIds[j]] = largest;
            
            heapIds[j] = tmp;
            inheap[heapIds[j]] = j;
            j = largest;
        }
        else
            break;
    }
}

void heapVerify(idxType *heapIds, idxType *key, idxType *vptrs, idxType sz, idxType *inheap)
{
    idxType i;
    for (i = 1; i <= sz/2; i++)
    {
        idxType l, r;
        
        l = i << 1 ;
        r = l + 1;
        if (inheap[heapIds[i]] != i)
            bu_errexit("heapVerify: location in heap is not correct i.\n");
        if(inheap[heapIds[l]] != l)
            bu_errexit("heapVerify: location in heap is not correct l.\n");
        
        if(key[heapIds[i]] < key[heapIds[l]] || (key[heapIds[i]] == key[heapIds[l]] && SIZEV(heapIds[i])<SIZEV(heapIds[l]) ))
            bu_errexit("heapVerify: not a maxheap l. %.0f %.0f (%d %d)\n", (double) key[heapIds[i]], (double) key[heapIds[l]], i, l);
        
        if (r <= sz)
        {
            if(inheap[heapIds[r]] != r)
                bu_errexit("heapVerify: location in heap is not correct r.\n");
            if(key[heapIds[i]] < key[heapIds[r]] || (key[heapIds[i]] == key[heapIds[r]] && SIZEV(heapIds[i])<SIZEV(heapIds[r])))
                bu_errexit("heapVerify: not a maxheap r.\n");
        }
    }
    for (i = sz/2+1; i <= sz; i++)
        if (inheap[heapIds[i]] != i)
            bu_errexit("heapVerify: location in heap is not correct i.\n");
}

