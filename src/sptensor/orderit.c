#include <stdio.h>
#include <stdlib.h>
#include "ParTI.h"

/*Interface to everything in this file is orderit(.., ..)*/

/*function declarations*/
void orderDim(sptIndex **coords, sptNnzIndex nnz, sptIndex nm, sptIndex *ndims, sptIndex dim, sptIndex **newIndices);

void orderit(sptSparseTensor *tsr, sptIndex **newIndices, sptIndex iterations)
{
    /*
     newIndices is of size [nmodes][ndims[modes]] and assumed to be allocted.
     It will be overwritten. No need to initialize.
     
     We will need to reshuffle nonzeros. In order to not to touch tsr, we copy the indices of nonzeros
     to a local variable coords. This is sort of transposed wrt tsr: its size is nnz * n, instead of n * nnz used in tsr.
     */
    sptIndex i, m, nm = tsr->nmodes;
    sptIndex **coords;
    sptNnzIndex z, nnz = tsr->nnz;
    
    sptIndex its;
    
    /*copy the indices*/
    coords = (sptIndex **) malloc(sizeof(sptIndex*) * nnz);
    for (z = 0; z < nnz; z++)
    {
        coords[z] = (sptIndex*) malloc(sizeof(sptIndex) * nm);
        for (m = 0; m < nm; m++)
            coords[z][m] = tsr->inds[m].data[z];
    }
    
    /*initialize to the identity */
    for (m = 0; m < nm; m++)
        for (i = 0; i < tsr->ndims[m]; i++)
            newIndices[m][i] = i;
    
    for (its = 0; its < iterations; its++)
    {
        printf("Optimizing the numbering for its %u\n", its+1);
        for (m = 0; m < nm; m++)
            orderDim(coords, nnz, nm, tsr->ndims, m, newIndices);
    }
    
    for (z = 0; z < nnz; z++)
        free(coords[z]);
    free(coords);
    
    
}
/******************** Internals begin ***********************/
/*beyond this line svages....
 **************************************************************/
void printCoords(sptIndex **coords, sptNnzIndex nnz, sptIndex nm)
{
    sptNnzIndex z;
    sptIndex m;
    for (z = 0; z < nnz; z++)
    {
        for (m=0; m < nm; m++)
            printf("%d ", coords[z][m]);
        printf("\n");
    }
}

/**************************************************************/
static inline int isLessThanOrEqualTo(sptIndex *z1, sptIndex *z2, sptIndex nm, sptIndex **newIndices, sptIndex *ndims, sptIndex dim)
{
    /*is z1 less than or equal to z2 for all indices except dim?*/
    sptIndex m;
    
    for (m = 0; m < nm; m++)
    {
        if(m != dim)
        {
            if (newIndices[m][z1[m]] < newIndices[m][z2[m]])
                return -1;
            if (newIndices[m][z1[m]] > newIndices[m][z2[m]])
                return 1;
        }
    }
    return 0; /*are equal*/
}

/**************************************************************/
static inline void buSwap(sptIndex *z1, sptIndex *z2, sptIndex nm, sptIndex *wspace)
{
    sptIndex m;
    
    for (m=0; m < nm; m++)
        wspace[m] = z2[m];
    
    for (m=0; m < nm; m++)
        z2[m] = z1[m];
    
    for (m=0; m < nm; m++)
        z1[m] = wspace[m];
    
}

/**************************************************************/
static inline sptNnzIndex buPartition(sptIndex **coords, sptNnzIndex lo, sptNnzIndex hi, sptIndex nm, sptIndex **newIndices, sptIndex *ndims, sptIndex dim, sptIndex *tmpNnz, sptIndex *wspace)
{
    /*cannot have lo - 1 because unsigned, and starts from 0
     the array is between [lo, ..., hi), the hi is not included.
     */
    sptNnzIndex left, right, mid=(lo+hi)/2;
    sptIndex m;
    
    buSwap(coords[lo], coords[mid], nm, wspace);/*bring the middle to the pivot position; to avoid worst case behaivor. totally random would be safer but...*/
    
    for (m = 0; m < nm; m++)
        tmpNnz[m] = coords[lo][m];
    
    left = lo;
    right = hi;
    while(1)
    {
        while(left <= hi && isLessThanOrEqualTo(coords[left], tmpNnz, nm, newIndices, ndims, dim) <= 0)
            left ++;
        while(isLessThanOrEqualTo(coords[right], tmpNnz, nm, newIndices, ndims, dim) > 0)
            right --;
        if(left >= right)
            break;
        buSwap(coords[left], coords[right], nm, wspace);
    }
    buSwap(coords[lo], coords[right], nm, wspace);

    return right;
}

/**************************************************************/
void mySort(sptIndex **coords,  sptNnzIndex last, sptIndex nm,  sptIndex **newIndices, sptIndex *ndims, sptIndex dim)
{
    /*sorts coords accourding to all dims except dim, where items are refereed with newIndices*/
    /*an iterative quicksort*/
    sptNnzIndex *stack, top, lo, hi, pv;
    sptIndex *tmpNnz, *wspace;
    sptNnzIndex total = 0;
    
    tmpNnz = (sptIndex*) malloc(sizeof(sptIndex) * nm);
    wspace = (sptIndex*) malloc(sizeof(sptIndex) * nm);
    stack = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * 2 * (last+2));
    
    top = 0;
    stack[top++] = 0;
    stack[top++] = last;
    while (top>=2)
    {
        hi = stack[--top];
        lo = stack[--top];
        total += hi-lo+1;
        
        pv = buPartition(coords, lo, hi, nm, newIndices, ndims, dim, tmpNnz, wspace);
        if(pv > lo+1)
        {
            stack[top++] = lo;
            stack[top++] = pv-1 ;
        }
        if(pv + 1 < hi)
        {
            stack[top++] = pv + 1 ;
            stack[top++] = hi;
        }
    }
    free(stack);
    free(wspace);
    free(tmpNnz);
    
}


/**************************************************************/
void lexOrderThem( sptNnzIndex m, sptIndex n, sptNnzIndex *ia, sptIndex *cols, sptIndex *cprm)
/*m, n are the num of rows and cols, respectively. We lex order cols,
 given rows.
 
 BU notes as of 4 May 2018: I am hoping that I will not be asked the details of this function, and its memory use;) A quick and dirty update from something else I had since some time. I did not think through if the arrays could be reduced. Right now we have 13 arrays of size n each (where n is the length of a single dimension of the tensor.
 */
{
    sptNnzIndex *flag, j, jcol, jend;
    sptIndex *svar,  *var;
    sptIndex *prev, *next, *sz, *setnext, *setprev, *tailset;
    
    sptIndex freeId, *freeIdNext, *freeIdPrev;
    sptIndex cnt = 0, k, s, acol;
    
    sptIndex *ids, *starts;
    
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
    freeIdNext  =(sptIndex*)calloc(sizeof(sptIndex),(n+2));
    freeIdPrev  =(sptIndex*)calloc(sizeof(sptIndex),(n+2));
    ids =(sptIndex*)calloc(sizeof(sptIndex),(n+2));
    starts =(sptIndex*)calloc(sizeof(sptIndex),(n+2));
    
    cnt = 1;

    next[1] = 2;
    prev[0] =  prev[1] = 0;
    next[n] = 0;
    prev[n] = n-1;
    svar[1] = svar[n] = 1;
    flag[1] = flag[n] = flag[n+1] = 0;
    sz[n] = sz[n+1] =  0;
    
    for(j = 2; j<=n-1; j++)/*init all in a single svar*/
    {
        svar[j] = 1;
        next[j] = j+1;
        prev[j] = j-1;
        flag[j] = 0;
        sz[j] = 0;
        setprev[j] = setnext[j] = 0;
    }
    var[1] = 1;
    sz[1] = n;
    
    setprev[1] = setprev[n] = 0;
    setnext[1] = setnext[n] = 0;
    tailset[1] = n;
    
    firstset = 1;
    
    freeIdNext[1] = freeIdPrev[1] = 1;
    freeIdNext[2] = 3;
    freeIdPrev[2] = n+1;
    freeIdNext[n+1] = 2;
    freeIdPrev[n+1] = n;
    
    for(j=3; j<=n; j++)
    {
        freeIdNext[j] = j+1;
        freeIdPrev[j] = j-1;
    }
    freeId = 2;
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
                if(sz[s] > 1)
                {
                    sptIndex newId, store;
                    /*remove i=acol from s*/
                    
                    if(tailset[s] == acol) tailset[s] = prev[acol];
                    
                    next[prev[acol]] = next[acol];
                    prev[next[acol]] = prev[acol];
                    prev[acol] = next[acol] = acol;
                    sz[s] = sz[s] - 1;
                    /*create a new supervar ns=newId
                     and make i=acol its only var*/
                    
                    newId = freeId;
                    store = freeId;
                    freeId = freeIdNext[freeId];
                    
                    freeIdNext [freeIdPrev[store]] = freeIdNext[store];
                    freeIdPrev [freeIdNext[store]] = freeIdPrev[store];
                    
                    svar[acol] = newId;
                    var[newId] = acol;
                    flag[newId] = j;
                    sz[newId ] = 1;
                    next[acol] = 0;
                    prev[acol] = 0;
                    var[s] = acol;
                    
                    setnext[newId] = s;
                    setprev[newId] = setprev[s];
                    if(setprev[s])
                        setnext[setprev[s]] = newId;
                    setprev[s] = newId;
                    
                    if(firstset == s)
                        firstset = newId;
                    tailset[newId] = acol;
                }
            }
            else/*second or later occurence of s for column j*/
            {
                k = var[s];
                svar[acol] = svar[k];
                /*remove from i=acol from current chain*/
                
                if(tailset[s] == acol) tailset[s] = prev[acol];
                
                next[prev[acol]] = next[acol];
                prev[next[acol]] = prev[acol];
                
                sz[s] = sz[s] - 1;
                if(sz[s] == 0)/*s is a free id now..*/
                {
                    freeIdNext[s] = freeId;
                    freeIdPrev[s] = freeIdPrev[freeId] ;
                    freeIdNext[freeIdPrev[freeId] ] = s;
                    freeIdPrev[freeId] = s;
                    
                    if(setnext[s])
                        setprev[setnext[s]] = setprev[s];
                    if(setprev[s])
                        setnext[setprev[s]] = setnext[s];
                    
                    if(firstset == s) firstset = setnext[s];
                    setprev[s] = setnext[s] = 0;
                    tailset[s] = 0;
                }
                /*add to chain containing k (as the last element)*/
                prev[acol] = tailset[svar[k]];
                next[acol]  = next[tailset[svar[k]]];
                next[tailset[svar[k]]] = acol;
                tailset[svar[k]] = acol;
                sz[svar[k]] = sz[svar[k]] + 1;
            }
        }
    }
    
    /*overwriting prev array.*/
    pos = 1;
    for(set = firstset; set != 0; set = setnext[set])
    {
        sptIndex item = tailset[set];
        sptIndex headset = 0;
        while(item != 0 )
        {
            headset = item;
            item = prev[item];
        }
        /*located head of the set. output them (this is for keeping the initial order*/
        while(headset)
        {
            cprm[pos++] = headset;
            headset = next[headset];
        }
    }
    
    free(tailset);
    free(sz);
    free(freeIdNext);
    free(freeIdPrev);
    free(next);
    free(prev);
    free(var);
    free(flag);
    free(svar);
    free(setnext);
    free(setprev);
    free(ids);
    free(starts);
    return ;
}

void printCSR(sptNnzIndex m, sptIndex n, sptNnzIndex *ia, sptIndex *cols)
{
    sptNnzIndex r, jend, jcol;
    printf("matrix of size %llu %u with %llu\n", m, n, ia[m+1]);
    
    for (r = 1; r <=m; r++)
    {
        jend = ia[r+1]-1;
        printf("r=%llu (%llu %llu)): ", r, ia[r], ia[r+1]);
        for(jcol = ia[r]; jcol <= jend ; jcol++)
            printf("%u ", cols[jcol]);
        printf("\n");
    }
}

#define myAbs(x) (((x) < 0) ? -(x) : (x))

void orderDim(sptIndex **coords, sptNnzIndex nnz, sptIndex nm, sptIndex *ndims, sptIndex dim, sptIndex **newIndices)
{
    sptNnzIndex *rowPtrs, z, atRowPlus1, mtxNrows;
    sptIndex *colIds, c;
    sptIndex *cprm;
    sptNnzIndex mtrxNnz;
    double totalDisplacement;
    
    mySort(coords,  nnz-1, nm,  newIndices, ndims, dim);
    /*we matricize this (others x thisDim), whose columns will be renumbered*/
    
    /*on the matrix all arrays are from 1, and all indices are from 1.*/
    rowPtrs = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * (nnz+2));/*large space*/
    colIds = (sptIndex *) malloc(sizeof(sptIndex) * (nnz+2));/*large space*/
    
    rowPtrs[0] = -99998888; /*we should not access this, that is why.*/
    rowPtrs [1] = 1;
    colIds[1] = coords[0][dim]+1;
    atRowPlus1 = 2;
    mtrxNnz = 2;/*start filling from the second element*/
    
    for (z = 1; z < nnz; z++)
    {
        if(isLessThanOrEqualTo(coords[z], coords[z-1], nm, newIndices, ndims, dim) != 0)
            rowPtrs[atRowPlus1++] = mtrxNnz;/*close the previous row and start a new one.*/
        colIds[mtrxNnz++] = coords[z][dim]+1;
    }
    rowPtrs[atRowPlus1] = mtrxNnz;
    mtxNrows = atRowPlus1-1;
    
    rowPtrs = realloc(rowPtrs, (sizeof(sptNnzIndex) * (mtxNrows+2)));
    cprm = (sptIndex *) malloc(sizeof(sptIndex) * (ndims[dim]+1));
    
    lexOrderThem( mtxNrows, ndims[dim], rowPtrs, colIds, cprm);
    
    totalDisplacement = 0;
    for (c=1; c <= ndims[dim]; c++)
    {
        totalDisplacement +=  myAbs((double) newIndices[dim][cprm[c]-1] - (double)(c-1));
        newIndices[dim][cprm[c]-1] = c-1;
    }
    printf("dim %u disp %.4f\n", dim, totalDisplacement/((double)(ndims[dim])));
    free(cprm);
    free(colIds);
    free(rowPtrs);
}
/********************** Internals end *************************/
