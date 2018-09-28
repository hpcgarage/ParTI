#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "ParTI.h"
#include "sptensor.h"

/*Interface to everything in this file is orderit(.., ..)*/

/*function declarations*/
void sptLexiOrderPerMode(sptSparseTensor * tsr, sptIndex const mode, sptIndex ** orgIds);
void sptBFSLike(sptSparseTensor * tsr, sptIndex ** newIndices);

static double u_seconds(void)
{
    struct timeval tp;
    
    gettimeofday(&tp, NULL);
    
    return (double) tp.tv_sec + (double) tp.tv_usec / 1000000.0;
    
};

void sptIndexRenumber(sptSparseTensor * tsr, sptIndex ** newIndices, int const renumber, sptIndex const iterations)
{
    /*
     newIndices is of size [nmodes][ndims[modes]] and assumed to be allocted.
     It will be overwritten. No need to initialize.
     
     We will need to reshuffle nonzeros. In order to not to touch tsr, we copy the indices of nonzeros
     to a local variable coords. This is sort of transposed wrt tsr: its size is nnz * n, instead of n * nnz used in tsr.
     */
    sptIndex const nmodes = tsr->nmodes;
    sptNnzIndex const nnz = tsr->nnz;  

    sptIndex i, m;
    sptNnzIndex z;
    sptIndex its;

    if (renumber == 1) {    /* Lexi-order renumbering */
        /* copy the indices */
        sptSparseTensor tsr_temp;
        sptCopySparseTensor(&tsr_temp, tsr);

        sptIndex ** orgIds = (sptIndex **) malloc(sizeof(sptIndex*) * nmodes);

        for (m = 0; m < nmodes; m++)
        {
            orgIds[m] = (sptIndex *) malloc(sizeof(sptIndex) * tsr->ndims[m]);
            for (i = 0; i < tsr->ndims[m]; i++)
                orgIds[m][i] = i;
        }

        // FILE * debug_fp = fopen("new.txt", "w");
        // fprintf(stdout, "orgIds:\n");
        for (its = 0; its < iterations; its++)
        {
            printf("[Lexi-order] Optimizing the numbering for its %u\n", its+1);
            for (m = 0; m < nmodes; m++)
                sptLexiOrderPerMode(&tsr_temp, m, orgIds);

            // fprintf(stdout, "\niter %u:\n", its);
            // for(sptIndex m = 0; m < tsr->nmodes; ++m) {
            //     sptDumpIndexArray(orgIds[m], tsr->ndims[m], stdout);
            // }
        }
        // fclose(debug_fp);

        /* compute newIndices from orgIds. Reverse perm */
        for (m = 0; m < nmodes; m++)
            for (i = 0; i < tsr->ndims[m]; i++)
                newIndices[m][orgIds[m][i]] = i;

        sptFreeSparseTensor(&tsr_temp);
        for (m = 0; m < nmodes; m++)
            free(orgIds[m]);
        free(orgIds);

    } else if (renumber == 2 ) {    /* BFS-like renumbering */
       /*
        REMARK (10 May 2018): this is the old bfs-like kind of thing. I hoped it would reduce the number of iterations,
        but on a few cases it did not help much. Just leaving it in case we want to use it.
        */
        printf("[BFS-like]\n");
        sptBFSLike(tsr, newIndices);
    }    
    
}


static void lexOrderThem( sptNnzIndex m, sptIndex n, sptNnzIndex *ia, sptIndex *cols, sptIndex *cprm)
{
    /*m, n are the num of rows and cols, respectively. We lex order cols,
     given rows.
     
     BU notes as of 4 May 2018: I am hoping that I will not be asked the details of this function, and its memory use;) A quick and dirty update from something else I had since some time. I did not think through if the arrays could be reduced. Right now we have 10 arrays of size n each (where n is the length of a single dimension of the tensor.
     */
    
    sptNnzIndex *flag, j, jcol, jend;
    sptIndex *svar,  *var, numBlocks, jj;
    sptIndex *prev, *next, *sz, *setnext, *setprev, *tailset;
    
    sptIndex *freeIdList, freeIdTop;
    
    sptIndex k, s, acol;
    
    sptIndex firstset, set, pos;
    
    svar = (sptIndex*) calloc(sizeof(sptIndex),(n+2));
    flag = (sptNnzIndex*) calloc(sizeof(sptNnzIndex),(n+2));
    var  = (sptIndex*) calloc(sizeof(sptIndex),(n+2));
    prev = (sptIndex*) calloc(sizeof(sptIndex),(n+2));
    next = (sptIndex*) calloc(sizeof(sptIndex),(n+2));
    sz   = (sptIndex*) calloc(sizeof(sptIndex),(n+2));
    setprev = (sptIndex*)calloc(sizeof(sptIndex),(n+2));
    setnext = (sptIndex*)calloc(sizeof(sptIndex),(n+2));
    tailset = (sptIndex*)calloc(sizeof(sptIndex),(n+2));
    freeIdList = (sptIndex*)calloc(sizeof(sptIndex),(n+2));
    
    next[1] = 2;
    prev[0] =  prev[1] = 0;
    next[n] = 0;
    prev[n] = n-1;
    svar[1] = svar[n] = 1;
    flag[1] = flag[n] = flag[n+1] = 0;
    cprm[1] = cprm[n] = 2 * n ;
    setprev[1] = setnext[1] = 0;
    for(jj = 2; jj<=n-1; jj++)/*init all in a single svar*/
    {
        svar[jj] = 1;
        next[jj] = jj+1;
        prev[jj] = jj-1;
        flag[jj] = 0;
        sz[jj] = 0;
        setprev[jj] = setnext[jj] = 0;
        cprm[jj] = 2 * n;
    }
    var[1] = 1;
    sz[1] = n;
    sz[n] = sz[n+1] =  0;
    
    setprev[n] = setnext[n] = 0;
    setprev[n+1] = setnext[n+1] = 0;
    
    tailset[1] = n;
    
    firstset = 1;
    freeIdList[0] = 0;
    
    for(jj= 1; jj<=n; jj++)
        freeIdList[jj] = jj+1;/*1 is used as a set id*/
    
    freeIdTop = 1;
    for(j=1; j<=m; j++)
    {
        jend = ia[j+1]-1;
        for(jcol = ia[j]; jcol <= jend ; jcol++)
        {
            acol= cols[jcol];
            s = svar[acol];
            if( flag[s] < j)/*first occurence of supervar s in j*/
            {
                flag[s] = j;
                if(sz[s] == 1 && tailset[s] != acol)
                {
                    printf("this should not happen (sz 1 but tailset not ok)\n");
                    exit(12);
                }
                if(sz[s] > 1)
                {
                    sptIndex newId;
                    /*remove acol from s*/
                    if(tailset[s] == acol) tailset[s] = prev[acol];
                    
                    next[prev[acol]] = next[acol];
                    prev[next[acol]] = prev[acol];
                    
                    sz[s] = sz[s] - 1;
                    /*create a new supervar ns=newId
                     and make i=acol its only var*/
                    if(freeIdTop == n+1) {
                        printf("this should not happen (no index)\n");
                        exit(12);
                    }
                    newId = freeIdList[freeIdTop++];
                    svar[acol] = newId;
                    var[newId] = acol;
                    flag[newId] = j;
                    sz[newId ] = 1;
                    next[acol] = 0;
                    prev[acol] = 0;
                    var[s] = acol;
                    tailset[newId] = acol;
                    
                    setnext[newId] = s;
                    setprev[newId] = setprev[s];
                    if(setprev[s])
                        setnext[setprev[s]] = newId;
                    setprev[s] = newId;
                    
                    if(firstset == s)
                        firstset = newId;
                    
                }
            }
            else/*second or later occurence of s for row j*/
            {
                k = var[s];
                svar[acol] = svar[k];
                
                /*remove acol from its current chain*/
                if(tailset[s] == acol) tailset[s] = prev[acol];
                
                next[prev[acol]] = next[acol];
                prev[next[acol]] = prev[acol];
                
                sz[s] = sz[s] - 1;
                if(sz[s] == 0)/*s is a free id now..*/
                {
                    
                    freeIdList[--freeIdTop] = s; /*add s to the free id list*/
                    
                    if(setnext[s])
                        setprev[setnext[s]] = setprev[s];
                    if(setprev[s])
                        setnext[setprev[s]] = setnext[s];
                    
                    setprev[s] = setnext[s] = 0;
                    tailset[s] = 0;
                    var[s] = 0;
                    flag[s] = 0;
                }
                /*add to chain containing k (as the last element)*/
                prev[acol] = tailset[svar[k]];
                next[acol]  = 0;/*BU next[tailset[svar[k]]];*/
                next[tailset[svar[k]]] = acol;
                tailset[svar[k]] = acol;
                sz[svar[k]] = sz[svar[k]] + 1;
            }
        }
    }
    
    pos = 1;
    numBlocks = 0;
    for(set = firstset; set != 0; set = setnext[set])
    {
        sptIndex item = tailset[set];
        sptIndex headset = 0;
        numBlocks ++;
        
        while(item != 0 )
        {
            headset = item;
            item = prev[item];
        }
        /*located the head of the set. output them (this is for keeping the initial order*/
        while(headset)
        {
            cprm[pos++] = headset;
            headset = next[headset];
        }
    }
    
    free(tailset);
    free(sz);
    free(next);
    free(prev);
    free(var);
    free(flag);
    free(svar);
    free(setnext);
    free(setprev);
    if(pos-1 != n){
        printf("**************** Error ***********\n");
        printf("something went wrong and we could not order everyone\n");
        exit(12);
    }
    
    return ;
}
/**************************************************************/
#define myAbs(x) (((x) < 0) ? -(x) : (x))

void sptLexiOrderPerMode(sptSparseTensor * tsr, sptIndex const mode, sptIndex ** orgIds)
{
    sptIndexVector * inds = tsr->inds;
    sptNnzIndex const nnz = tsr->nnz;
    sptIndex const nmodes = tsr->nmodes;
    sptIndex * ndims = tsr->ndims;
    sptIndex const mode_dim = ndims[mode];
    sptNnzIndex * rowPtrs = NULL;
    sptIndex * colIds = NULL;
    sptIndex * cprm = NULL, * invcprm = NULL, * saveOrgIds = NULL;
    sptNnzIndex atRowPlus1, mtxNrows, mtrxNnz;
    sptIndex * mode_order = (sptIndex *) malloc (sizeof(sptIndex) * (nmodes - 1));

    sptIndex c;
    sptNnzIndex z;
    double t1, t0;

    t0 = u_seconds();
    sptIndex i = 0;
    for(sptIndex m = 0; m < nmodes; ++m) {
        if (m != mode) {
            mode_order[i] = m;
            ++ i;
        }
    }
    sptSparseTensorSortIndexExceptSingleMode(tsr, 1, mode_order);
    // mySort(coords,  nnz-1, nmodes, ndims, mode);
    t1 = u_seconds()-t0;
    printf("\nmode %u, sort time %.2f\n", mode, t1);
    // sptAssert(sptDumpSparseTensor(tsr, 0, stdout) == 0);

    /* we matricize this (others x thisDim), whose columns will be renumbered */
    /* on the matrix all arrays are from 1, and all indices are from 1. */
    
    rowPtrs = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * (nnz + 2)); /*large space*/
    colIds = (sptIndex *) malloc(sizeof(sptIndex) * (nnz + 2)); /*large space*/
    
    if(rowPtrs == NULL || colIds == NULL)
    {
        printf("could not allocate.exiting \n");
        exit(12);
    }
    
    rowPtrs[0] = 0; /* we should not access this, that is why. */
    rowPtrs [1] = 1;
    colIds[1] = inds[mode].data[0] + 1;
    atRowPlus1 = 2;
    mtrxNnz = 2;/* start filling from the second element */
    
    t0 = u_seconds();
    for (z = 1; z < nnz; z++)
    {
        if(spt_SparseTensorCompareIndicesExceptSingleMode(tsr, z, tsr, z-1, mode_order) != 0)
            rowPtrs[atRowPlus1++] = mtrxNnz; /* close the previous row and start a new one. */
        
        colIds[mtrxNnz ++] = inds[mode].data[z] + 1;
    }
    rowPtrs[atRowPlus1] = mtrxNnz;
    mtxNrows = atRowPlus1-1;
    t1 =u_seconds()-t0;
    printf("mode %u, create time %.2f\n", mode, t1);
    
    rowPtrs = realloc(rowPtrs, (sizeof(sptNnzIndex) * (mtxNrows + 2)));
    cprm = (sptIndex *) malloc(sizeof(sptIndex) * (mode_dim + 1));
    invcprm = (sptIndex *) malloc(sizeof(sptIndex) * (mode_dim + 1));
    saveOrgIds = (sptIndex *) malloc(sizeof(sptIndex) * (mode_dim + 1));

    // printf("rowPtrs: \n");
    // sptDumpNnzIndexArray(rowPtrs, mtxNrows + 2, stdout);
    // printf("colIds: \n");
    // sptDumpIndexArray(colIds, nnz + 2, stdout);    
    
    t0 = u_seconds();
    lexOrderThem(mtxNrows, mode_dim, rowPtrs, colIds, cprm);
    t1 =u_seconds()-t0;
    printf("mode %u, lexorder time %.2f\n", mode, t1);
    // printf("cprm: \n");
    // sptDumpIndexArray(cprm, mode_dim + 1, stdout);

    /* update orgIds and modify coords */
    for (c=0; c < mode_dim; c++)
    {
        invcprm[cprm[c+1]-1] = c;
        saveOrgIds[c] = orgIds[mode][c];
    }
    for (c=0; c < mode_dim; c++)
        orgIds[mode][c] = saveOrgIds[cprm[c+1]-1];

    // printf("invcprm: \n");
    // sptDumpIndexArray(invcprm, mode_dim + 1, stdout);
    
    /* rename the dim component of nonzeros */
    for (z = 0; z < nnz; z++)
        inds[mode].data[z] = invcprm[inds[mode].data[z]];
    
    free(saveOrgIds);
    free(invcprm);
    free(cprm);
    free(colIds);
    free(rowPtrs);
    free(mode_order);
}

/**************************************************************/

typedef struct{
    sptIndex nvrt; /* number of vertices. This nvrt = n_0 + n_1 + ... + n_{d-1} for a d-dimensional tensor
                   where the ith dimension is of size n_i.*/
    sptNnzIndex *vptrs, *vHids; /*starts of hedges containing vertices, and the ids of the hedges*/
    
    sptNnzIndex nhdg; /*this will be equal to the number of nonzeros in the tensor*/
    sptNnzIndex *hptrs, *hVids; /*starts of vertices in the hedges, and the ids of the vertices*/
} basicHypergraph;

static void allocateHypergraphData(basicHypergraph *hg, sptIndex nvrt, sptNnzIndex nhdg, sptNnzIndex npins)
{
    hg->nvrt = nvrt;
    hg->vptrs = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * (nvrt+1));
    hg->vHids = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * npins);
    
    hg->nhdg = nhdg;
    hg->hptrs = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * (nhdg+1));
    hg->hVids = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * npins);
}


static void freeHypergraphData(basicHypergraph *hg)
{
    hg->nvrt = 0;
    if (hg->vptrs) free(hg->vptrs);
    if (hg->vHids) free(hg->vHids);
    
    hg->nhdg = 0;
    if (hg->hptrs) free(hg->hptrs);
    if (hg->hVids) free(hg->hVids);
}


static void setVList(basicHypergraph *hg)
{
    /*PRE: We assume hg->hptrs and hg->hVids are set; hg->nvrts is set, and
     hg->vptrs and hg->vHids are allocated appropriately.
     */
    
    sptNnzIndex j, h, v, nhdg = hg->nhdg;
    
    sptIndex nvrt = hg->nvrt;
    
    /*vertices */
    sptNnzIndex *vptrs = hg->vptrs, *vHids = hg->vHids;
    /*hyperedges*/
    sptNnzIndex *hptrs = hg->hptrs, *hVids = hg->hVids;
    
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
    
    for (h = nhdg; h >= 1; h--)
    {
        for (j = hptrs[h-1]; j < hptrs[h]; j++)
        {
            v = hVids[j];
            vHids[--(vptrs[v])] = h-1;
        }
    }
}

static void fillHypergraphFromCoo(basicHypergraph *hg, sptIndex nm, sptNnzIndex nnz, sptIndex *ndims, sptIndexVector * inds)
{
    
    sptIndex  totalSizes;
    sptNnzIndex h, toAddress;
    sptIndex *dimSizesPrefixSum;
    
    sptIndex i;
    
    dimSizesPrefixSum = (sptIndex *) malloc(sizeof(sptIndex) * (nm+1));
    totalSizes = 0;
    for (i=0; i < nm; i++)
    {
        dimSizesPrefixSum[i] = totalSizes;
        totalSizes += ndims[i];
    }
    printf("allocating hyp %u %llu\n", nm, nnz);
    
    allocateHypergraphData(hg, totalSizes, nnz, nnz * nm);
    
    toAddress = 0;
    for (h = 0; h < nnz; h++)
    {
        hg->hptrs[h] = toAddress;
        for (i = 0;  i < nm; i++)
            hg->hVids[toAddress + i] = dimSizesPrefixSum[i] + inds[i].data[h];
        toAddress += nm;
    }
    hg->hptrs[hg->nhdg] = toAddress;
    
    setVList(hg);
    free(dimSizesPrefixSum);
}

static inline sptIndex locateVertex(sptNnzIndex indStart, sptNnzIndex indEnd, sptNnzIndex *lst, sptNnzIndex sz)
{
    sptNnzIndex i;
    for (i = 0; i < sz; i++)
        if(lst[i] >= indStart && lst[i] <= indEnd)
            return lst[i];
    
    printf("could not locate in a hyperedge !!!\n");
    exit(1);
    return sz+1;
}

#define SIZEV( vid ) vptrs[(vid)+1]-vptrs[(vid)]
static void heapIncreaseKey(sptIndex *heapIds, sptNnzIndex *key, sptNnzIndex *vptrs, sptIndex sz, sptIndex id, sptIndex *inheap, sptNnzIndex newKey)
{
    
    sptIndex i = inheap[id]; /*location in heap*/
    if( i > 0 && i <=sz )
    {
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
}


static void heapify(sptIndex *heapIds, sptNnzIndex *key, sptNnzIndex *vptrs, sptIndex sz, sptIndex i,  sptIndex *inheap)
{
    sptIndex largest, j, l,r, tmp;
    
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

static sptIndex heapExtractMax(sptIndex *heapIds, sptNnzIndex *key, sptNnzIndex *vptrs, sptIndex *sz, sptIndex *inheap)
{
    sptIndex maxind ;
    if (*sz < 1){
        printf("Error: heap underflow\n"); exit(12);
    }
    maxind = heapIds[1];
    heapIds[1] = heapIds[*sz];
    inheap[heapIds[1]] = 1;
    
    *sz = *sz - 1;
    inheap[maxind] = 0;
    
    heapify(heapIds, key, vptrs, *sz, 1, inheap);
    return maxind;
    
}

static void heapBuild(sptIndex *heapIds, sptNnzIndex *key, sptNnzIndex *vptrs, sptIndex sz, sptIndex *inheap)
{
    sptIndex i;
    for (i=sz/2; i>=1; i--)
        heapify(heapIds, key, vptrs, sz, i, inheap);
}

static void orderforHiCOOaDim(basicHypergraph *hg, sptIndex *newIndicesHg, sptIndex indStart, sptIndex indEnd)
{
    /* we re-order the vertices of the hypergraph with ids in the range [indStart, indEnd]*/
    
    sptIndex i, v, heapSz, *inHeap, *heapIds;
    sptNnzIndex j, jj, hedge, hedge2, k, w, ww;
    sptNnzIndex *vptrs = hg->vptrs, *vHids = hg->vHids, *hptrs = hg->hptrs, *hVids = hg->hVids;
    
    sptNnzIndex *keyvals, newKeyval;
    int *markers, mark;
    
    mark = 0;
    
    heapIds = (sptIndex*) malloc(sizeof(sptIndex) * (indEnd-indStart + 2));
    inHeap = (sptIndex*) malloc(sizeof(sptIndex) * hg->nvrt);/*this is large*/
    keyvals = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * hg->nvrt);
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
                        if( inHeap[ww] )
                        {
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


/**************************************************************/
void sptBFSLike(sptSparseTensor * tsr, sptIndex ** newIndices)
{
    /*PRE: newIndices is allocated
     
     POST:
     newIndices[0][0...n_0-1] gives the new ids for dim 0
     newIndices[1][0...n_1-1] gives the new ids for dim 1
     ...
     newIndices[d-1][0...n_{d-1}-1] gives the new ids for dim d-1
     
     This implements a simple idea close to BFS/Cuthill-McKee/Maximum cardinality search.
     */
    sptIndex const nmodes = tsr->nmodes;
    sptNnzIndex const nnz = tsr->nnz;
    sptIndex * ndims = tsr->ndims;
    sptIndexVector * inds = tsr->inds;
    
    sptIndex *dimsPrefixSum;
    basicHypergraph hg;
    sptIndex *newIndicesHg;
    sptIndex d, i;
    
    dimsPrefixSum = (sptIndex*) calloc(nmodes, sizeof(sptIndex));
    for (d = 1; d < nmodes; d++)
        dimsPrefixSum[d] = ndims[d-1] + dimsPrefixSum[d-1];
    
    fillHypergraphFromCoo(&hg, nmodes,  nnz, ndims, inds);

    newIndicesHg = (sptIndex*) malloc(sizeof(sptIndex) * hg.nvrt);
    for (i = 0; i < hg.nvrt; i++)
        newIndicesHg[i] = i;
    
    for (d = 0; d < nmodes; d++) /*order d*/
        orderforHiCOOaDim(&hg, newIndicesHg, dimsPrefixSum[d], dimsPrefixSum[d] + ndims[d]-1);
    
    /*copy from newIndices to newIndicesOut*/
    for (d = 0; d < nmodes; d++)
        for (i = 0; i < ndims[d]; i++)
            newIndices[d][i] = newIndicesHg[dimsPrefixSum[d] + i] - dimsPrefixSum[d];
    
    free(newIndicesHg);
    freeHypergraphData(&hg);
    free(dimsPrefixSum);
    
}
/********************** Internals end *************************/
