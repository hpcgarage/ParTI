#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "ParTI.h"

/*Interface to everything in this file is orderit(.., ..)*/

/*function declarations*/
void orderDim(sptIndex ** coords, sptNnzIndex const nnz, sptIndex const nm, sptIndex * ndims, sptIndex const dim, sptIndex ** newIndices);

void orderforHiCOObfsLike(sptIndex const nm, sptNnzIndex const nnz, sptIndex * ndims, sptIndex ** coords, sptIndex ** newIndices);

static double u_seconds(void)
{
    struct timeval tp;
    
    gettimeofday(&tp, NULL);
    
    return (double) tp.tv_sec + (double) tp.tv_usec / 1000000.0;
    
};
static void printCSR(sptNnzIndex m, sptIndex n, sptNnzIndex *ia, sptIndex *cols)
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

static void checkRepeatIndex(sptNnzIndex mtxNrows, sptNnzIndex *rowPtrs, sptIndex *cols, sptIndex n )
{
    printf("\tChecking repeat indices\n");
    sptIndex *marker = (sptIndex *) calloc(n+1, sizeof(sptIndex));
    sptNnzIndex r,  jcol, jend;
    for (r = 1; r <= mtxNrows; r++)
    {
        jend = rowPtrs[r+1]-1;
        for (jcol = rowPtrs[r]; jcol <= jend; jcol++)
        {
            if( marker[cols[jcol]] < r )
                marker[cols[jcol]] = r;
            else if (marker[cols[jcol]] == r)
            {
                printf("*************************\n");
                printf("error duplicate col index %u at row %llu\n", cols[jcol], r);
                printf("*************************\n");
                
                exit(12);
            }
        }
        
    }
    free(marker);
}
static void checkEmptySlices(sptIndex **coords, sptNnzIndex nnz, sptIndex nm, sptIndex *ndims)
{
    sptIndex m, i;
    sptNnzIndex z;
    sptIndex **marker;
    
    marker = (sptIndex **) malloc(sizeof(sptIndex*) * nm);
    for (m = 0; m < nm; m++)
        marker[m] = (sptIndex*) calloc(ndims[m], sizeof(sptIndex) );
    
    for (z = 0; z < nnz; z++)
        for (m=0; m < nm; m++)
            marker[m][coords[z][m]] = m + 1;
    
    for (m=0; m < nm; m++)
    {
        sptIndex emptySlices = 0;
        for (i = 0; i < ndims[m]; i++)
            if(marker[m][i] != m+1)
                emptySlices ++;
        if(emptySlices)
            printf("dim %u, empty slices %u of %u\n", m, emptySlices,ndims[m] );
    }
    for (m = 0; m < nm; m++)
        free(marker[m]);
    free(marker);
}

static void checkNewIndices(sptIndex **newIndices, sptIndex nm, sptIndex *ndims)
{
    sptIndex m, i;
    sptIndex **marker, leftVoid;
    
    marker = (sptIndex **) malloc(sizeof(sptIndex*) * nm);
    for (m = 0; m < nm; m++)
        marker[m] = (sptIndex*) calloc(ndims[m], sizeof(sptIndex) );
    
    for (m=0; m < nm; m++)
        for (i = 0; i < ndims[m]; i++)
            marker[m][newIndices[m][i]] = m + 1;
    
    leftVoid = 0;
    for (m=0; m < nm; m++)
    {
        for (i = 0; i < ndims[m]; i++)
            if(marker[m][i] != m+1)
                leftVoid ++;
        if(leftVoid)
            printf("dim %u, left void %u of %u\n", m, leftVoid, ndims[m] );
    }
    for (m = 0; m < nm; m++)
        free(marker[m]);
    free(marker);
}


void orderit(sptSparseTensor * tsr, sptIndex ** newIndices, int const renumber, sptIndex const iterations)
{
    /*
     newIndices is of size [nmodes][ndims[modes]] and assumed to be allocted.
     It will be overwritten. No need to initialize.
     
     We will need to reshuffle nonzeros. In order to not to touch tsr, we copy the indices of nonzeros
     to a local variable coords. This is sort of transposed wrt tsr: its size is nnz * n, instead of n * nnz used in tsr.
     */
    sptIndex i, m, nm = tsr->nmodes;
    sptNnzIndex z, nnz = tsr->nnz;
    sptIndex ** coords;   
    sptIndex its;
    
    /* copy the indices */
    sptTimer copy_coord_timer;
    sptNewTimer(&copy_coord_timer, 0);
    sptStartTimer(copy_coord_timer);

    coords = (sptIndex **) malloc(sizeof(sptIndex*) * nnz);
    for (z = 0; z < nnz; z++)
    {
        coords[z] = (sptIndex *) malloc(sizeof(sptIndex) * nm);
        for (m = 0; m < nm; m++) {
            coords[z][m] = tsr->inds[m].data[z];
        }
    }

    sptStopTimer(copy_coord_timer);
    sptPrintElapsedTime(copy_coord_timer, "Copy coordinate time");
    sptFreeTimer(copy_coord_timer);
    
    /* checkEmptySlices(coords, nnz, nm, tsr->ndims); */

    if (renumber == 1) {    /* Lexi-order renumbering */

        sptIndex ** orgIds = (sptIndex **) malloc(sizeof(sptIndex*) * nm);

        for (m = 0; m < nm; m++)
        {
            orgIds[m] = (sptIndex *) malloc(sizeof(sptIndex) * tsr->ndims[m]);
            for (i = 0; i < tsr->ndims[m]; i++)
                orgIds[m][i] = i;
        }

        // FILE * debug_fp = fopen("old.txt", "w");
        // fprintf(stdout, "orgIds:\n");
        for (its = 0; its < iterations; its++)
        {
            printf("[Lexi-order] Optimizing the numbering for its %u\n", its+1);
            for (m = 0; m < nm; m++)
                orderDim(coords, nnz, nm, tsr->ndims, m, orgIds);
            
            // fprintf(stdout, "\niter %u:\n", its);
            // for(sptIndex m = 0; m < tsr->nmodes; ++m) {
            //     sptDumpIndexArray(orgIds[m], tsr->ndims[m], stdout);
            // }
        }
        // fclose(debug_fp);

        /* compute newIndices from orgIds. Reverse perm */
        for (m = 0; m < nm; m++)
            for (i = 0; i < tsr->ndims[m]; i++)
                newIndices[m][orgIds[m][i]] = i;

        for (m = 0; m < nm; m++)
            free(orgIds[m]);
        free(orgIds);

    } else if (renumber == 2 ) {    /* BFS-like renumbering */
       /*
        REMARK (10 May 2018): this is the old bfs-like kind of thing. I hoped it would reduce the number of iterations,
        but on a few cases it did not help much. Just leaving it in case we want to use it.
        */
        printf("[BFS-like]\n");
        orderforHiCOObfsLike(nm, nnz, tsr->ndims, coords, newIndices);
    }
    
    // printf("set the new indices\n");
/*    checkNewIndices(newIndices, nm, tsr->ndims);*/
    
    for (z = 0; z < nnz; z++)
        free(coords[z]);
    free(coords);
    
}
/******************** Internals begin ***********************/
/*beyond this line savages....
 **************************************************************/
static void printCoords(sptIndex **coords, sptNnzIndex nnz, sptIndex nm)
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
// static inline int isLessThanOrEqualToCoord(sptIndex *z1, sptIndex *z2, sptIndex nm, sptIndex *ndims, sptIndex dim)
static inline int isLessThanOrEqualTo(sptIndex *z1, sptIndex *z2, sptIndex nm, sptIndex *ndims, sptIndex dim)
{
    /*is z1 less than or equal to z2 for all indices except dim?*/
    sptIndex m;
    
    for (m = 0; m < nm; m++)
    {
        if(m != dim)
        {
            if (z1[m] < z2[m])
                return -1;
            if (z1[m] > z2[m])
                return 1;
        }
    }
    return 0; /*are equal*/
}

static inline int isLessThanOrEqualToNewSum(sptIndex *z1, sptIndex *z2, sptIndex nm, sptIndex *ndims, sptIndex dim)
// static inline int isLessThanOrEqualTo(sptIndex *z1, sptIndex *z2, sptIndex nm, sptIndex *ndims, sptIndex dim)
{
    /*
     to sort the nonzeros first on i_1+i_2+...+i_4, if ties then on
     i_1+i_2+...+3, if ties then on i_1+i_2, if ties then on i_1 only.
     We do not include dim in the comparisons.
     
    */
    sptIndex m;
    sptIndex v1 = 0, v2 = 0;
    
    for (m = 0; m < nm; m++)
    {
        if(m != dim)
        {
            v1 += z1[m];
            v2 += z2[m];
        }
    }
    if(v1 < v2) return -1;
    else if(v1 > v2) return 1;
    else{
        for (m = 0; m < nm; m++)
        {
            if(m != dim)
            {
                v1 -= z1[m];
                v2 -= z2[m];
                if (v1 < v2) return -1;
                else if (v1 > v2) return 1;
            }
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

static inline void writeInto(sptIndex *target, sptIndex *source, sptIndex nm)
{
    sptIndex m;
    for (m = 0; m < nm; m++)
        target[m] = source[m];
}

static void insertionSort(sptIndex **coords, sptNnzIndex lo, sptNnzIndex hi, sptIndex nm, sptIndex *ndims, sptIndex dim, sptIndex *tmpNnz, sptIndex *wspace)
{
    sptNnzIndex z, z2plus;
    for (z = lo+1; z <= hi; z++)
    {
        writeInto(tmpNnz, coords[z], nm);
        /*find place for z*/
        z2plus = z;
        while ( z2plus > 0  && isLessThanOrEqualTo(coords[z2plus-1], tmpNnz, nm, ndims, dim)== 1)
        {
            writeInto(coords[z2plus], coords[z2plus-1], nm);
            z2plus --;
        }
        writeInto(coords[z2plus], tmpNnz, nm);
    }
}
static inline sptNnzIndex buPartition(sptIndex **coords, sptNnzIndex lo, sptNnzIndex hi, sptIndex nm, sptIndex *ndims, sptIndex dim, sptIndex *tmpNnz, sptIndex *wspace)
{
    /* copied from the web http://ndevilla.free.fr/median/median/src/quickselect.c */
    sptNnzIndex low, high, median, middle, ll, hh;
    
    low = lo; high = hi; median = (low+high)/2;
    for(;;)
    {
        if (high<=low) return median;
        if(high == low + 1)
        {
            if(isLessThanOrEqualTo(coords[low], coords[high], nm, ndims, dim)== 1)
                buSwap (coords[high], coords[low], nm, wspace);
            return median;
        }
        middle = (low+high)/2;
        if(isLessThanOrEqualTo(coords[middle], coords[high], nm, ndims, dim) == 1)
            buSwap (coords[middle], coords[high], nm, wspace);
        
        if(isLessThanOrEqualTo(coords[low], coords[high], nm, ndims, dim) == 1)
            buSwap (coords[low], coords[high], nm, wspace);
        
        if(isLessThanOrEqualTo(coords[middle], coords[low], nm, ndims, dim) == 1)
            buSwap (coords[low], coords[middle], nm, wspace);
        
        buSwap (coords[middle], coords[low+1], nm, wspace);
        
        ll = low + 1;
        hh = high;
        for (;;){
            do ll++; while (isLessThanOrEqualTo(coords[low], coords[ll], nm, ndims, dim) == 1);
            do hh--; while (isLessThanOrEqualTo(coords[hh], coords[low], nm, ndims, dim) == 1);
            
            if (hh < ll) break;
            
            buSwap (coords[ll], coords[hh], nm, wspace);
        }
        buSwap (coords[low], coords[hh], nm,wspace);
        if (hh <= median) low = ll;
        if (hh >= median) high = hh - 1;
    }
    
}
/**************************************************************/
static void mySort(sptIndex ** coords,  sptNnzIndex last, sptIndex nm, sptIndex * ndims, sptIndex dim)
{
    /* sorts coords accourding to all dims except dim, where items are refereed with newIndices*/
    /* an iterative quicksort */
    sptNnzIndex *stack, top, lo, hi, pv;
    sptIndex *tmpNnz, *wspace;
    
    tmpNnz = (sptIndex*) malloc(sizeof(sptIndex) * nm);
    wspace = (sptIndex*) malloc(sizeof(sptIndex) * nm);
    stack = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * 2 * (last+2));
    
    if(stack == NULL) {
        printf("could not allocated stack. returning\n");
        exit(14);
    }
    top = 0;
    stack[top++] = 0;
    stack[top++] = last;
    while (top>=2)
    {
        hi = stack[--top];
        lo = stack[--top];
        pv = buPartition(coords, lo, hi, nm, ndims, dim, tmpNnz, wspace);
        
        if(pv > lo+1)
        {
            if(pv - lo > 128)
            {
                stack[top++] = lo;
                stack[top++] = pv-1 ;
            }
            else
                insertionSort(coords, lo, pv-1,  nm, ndims, dim, tmpNnz, wspace);
        }
        if(top >= 2 * (last+2)){
            printf("\thow come this tight?\n");
            exit(13);
        }
        if(pv + 1 < hi)
        {
            if(hi - pv > 128)
            {
                stack[top++] = pv + 1 ;
                stack[top++] = hi;
            }
            else
                insertionSort(coords, pv+1, hi,  nm, ndims, dim, tmpNnz, wspace);
        }
        if( top >= 2 * (last+2)) {
            printf("\thow come this tight?\n");
            exit(13);
        }
    }
    free(stack);
    free(wspace);
    free(tmpNnz);
}

static sptIndex countNumItems(sptIndex *setnext, sptIndex *tailset, sptIndex firstset, sptIndex *prev)
{
    sptIndex cnt = 0, set;
    for(set = firstset; set != 0; set = setnext[set])
    {
        sptIndex item = tailset[set];
        
        while(item != 0 )
        {
            cnt ++;
            item = prev[item];
        }
    }
    return cnt;
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

void orderDim(sptIndex ** coords, sptNnzIndex const nnz, sptIndex const nm, sptIndex * ndims, sptIndex const dim, sptIndex ** orgIds)
{
    sptNnzIndex * rowPtrs=NULL, z, atRowPlus1, mtxNrows;
    sptIndex * colIds=NULL, c;
    sptIndex * cprm=NULL, * invcprm = NULL, * saveOrgIds;
    sptNnzIndex mtrxNnz;
    
    double t1, t0;
    t0 = u_seconds();
    mySort(coords,  nnz-1, nm, ndims, dim);
    t1 = u_seconds()-t0;
    printf("\ndim %u, sort time %.2f\n", dim, t1);
    // printCoords(coords, nnz, nm);
    /* we matricize this (others x thisDim), whose columns will be renumbered */
    
    /* on the matrix all arrays are from 1, and all indices are from 1. */
    
    rowPtrs = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * (nnz+2)); /*large space*/
    colIds = (sptIndex *) malloc(sizeof(sptIndex) * (nnz+2)); /*large space*/
    
    if(rowPtrs == NULL || colIds == NULL)
    {
        printf("could not allocate.exiting \n");
        exit(12);
    }
    
    rowPtrs[0] = 0; /* we should not access this, that is why. */
    rowPtrs [1] = 1;
    colIds[1] = coords[0][dim]+1;
    atRowPlus1 = 2;
    mtrxNnz = 2;/* start filling from the second element */
    
    t0 = u_seconds();
    for (z = 1; z < nnz; z++)
    {
        if(isLessThanOrEqualTo( coords[z], coords[z-1], nm, ndims, dim) != 0)
            rowPtrs[atRowPlus1 ++] = mtrxNnz; /* close the previous row and start a new one. */
        
        colIds[mtrxNnz++] = coords[z][dim]+1;
    }
    rowPtrs[atRowPlus1] = mtrxNnz;
    mtxNrows = atRowPlus1-1;
    t1 =u_seconds()-t0;
    printf("dim %u create time %.2f\n", dim, t1);
    
    rowPtrs = realloc(rowPtrs, (sizeof(sptNnzIndex) * (mtxNrows+2)));
    cprm = (sptIndex *) malloc(sizeof(sptIndex) * (ndims[dim]+1));
    invcprm = (sptIndex *) malloc(sizeof(sptIndex) * (ndims[dim]+1));
    saveOrgIds = (sptIndex *) malloc(sizeof(sptIndex) * (ndims[dim]+1));
    /*    checkRepeatIndex(mtxNrows, rowPtrs, colIds, ndims[dim] );*/

    // printf("rowPtrs: \n");
    // sptDumpNnzIndexArray(rowPtrs, mtxNrows + 2, stdout);
    // printf("colIds: \n");
    // sptDumpIndexArray(colIds, nnz + 2, stdout); 
    
    t0 = u_seconds();
    lexOrderThem(mtxNrows, ndims[dim], rowPtrs, colIds, cprm);
    t1 =u_seconds()-t0;
    printf("dim %u lexorder time %.2f\n", dim, t1);
    // printf("cprm: \n");
    // sptDumpIndexArray(cprm, ndims[dim] + 1, stdout);

    /* update orgIds and modify coords */
    for (c=0; c < ndims[dim]; c++)
    {
        invcprm[cprm[c+1]-1] = c;
        saveOrgIds[c] = orgIds[dim][c];
    }
    for (c=0; c < ndims[dim]; c++)
        orgIds[dim][c] = saveOrgIds[cprm[c+1]-1];
    
    // printf("invcprm: \n");
    // sptDumpIndexArray(invcprm, ndims[dim] + 1, stdout);

    /*rename the dim component of nonzeros*/
    for (z = 0; z < nnz; z++)
        coords[z][dim] = invcprm[coords[z][dim]];
    
    free(saveOrgIds);
    free(invcprm);
    free(cprm);
    free(colIds);
    free(rowPtrs);
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

static void fillHypergraphFromCoo(basicHypergraph *hg, sptIndex nm, sptNnzIndex nnz, sptIndex *ndims, sptIndex **coords)
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
            hg->hVids[toAddress + i] = dimSizesPrefixSum[i] + coords[h][i];
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
void orderforHiCOObfsLike(sptIndex const nm, sptNnzIndex const nnz, sptIndex * ndims, sptIndex ** coords, sptIndex ** newIndices)
{
    /*PRE: newIndices is allocated
     
     POST:
     newIndices[0][0...n_0-1] gives the new ids for dim 0
     newIndices[1][0...n_1-1] gives the new ids for dim 1
     ...
     newIndices[d-1][0...n_{d-1}-1] gives the new ids for dim d-1
     
     This implements a simple idea close to BFS/Cuthill-McKee/Maximum cardinality search.
     */
    sptIndex d, i;
    sptIndex *dimsPrefixSum;
    
    basicHypergraph hg;
    
    sptIndex *newIndicesHg;
    
    dimsPrefixSum = (sptIndex*) calloc(nm, sizeof(sptIndex));
    for (d = 1; d < nm; d++)
        dimsPrefixSum[d] = ndims[d-1] + dimsPrefixSum[d-1];
    
    fillHypergraphFromCoo(&hg, nm,  nnz, ndims, coords);
    newIndicesHg = (sptIndex*) malloc(sizeof(sptIndex) * hg.nvrt);
    
    for (i = 0; i < hg.nvrt; i++)
        newIndicesHg[i] = i;
    
    for (d = 0; d < nm; d++) /*order d*/
        orderforHiCOOaDim(&hg, newIndicesHg, dimsPrefixSum[d], dimsPrefixSum[d] + ndims[d]-1);
    
    /*copy from newIndices to newIndicesOut*/
    for (d = 0; d < nm; d++)
        for (i = 0; i < ndims[d]; i++)
            newIndices[d][i] = newIndicesHg[dimsPrefixSum[d] + i] - dimsPrefixSum[d];
    
    free(newIndicesHg);
    freeHypergraphData(&hg);
    free(dimsPrefixSum);
    
}
/********************** Internals end *************************/
