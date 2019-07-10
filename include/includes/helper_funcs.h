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

#ifndef PARTI_HELPER_FUNCS_H
#define PARTI_HELPER_FUNCS_H

#define max(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   _a > _b ? _a : _b; })
 
int sptGetLastError(const char **module, const char **file, unsigned *line, const char **reason);
void sptClearLastError(void);
void spt_Panic(const char *file, unsigned line, const char *expr);
/**
 * The assert function that always execute even when `NDEBUG` is set
 *
 * Quick & dirty error checking. Useful when writing small programs.
 */
#define sptAssert(expr) ((expr) ? (void) 0 : spt_Panic(__FILE__, __LINE__, #expr))

/* Helper function for pure C module */
int sptCudaSetDevice(int device);
int sptCudaGetLastError(void);

/* Timer functions, using either CPU or GPU timer */
int sptNewTimer(sptTimer *timer, int use_cuda);
int sptStartTimer(sptTimer timer);
int sptStopTimer(sptTimer timer);
double sptElapsedTime(const sptTimer timer);
double sptPrintElapsedTime(const sptTimer timer, const char *name);
double sptPrintAverageElapsedTime(const sptTimer timer, const int niters, const char *name);
int sptFreeTimer(sptTimer timer);

/* Base functions */
char * sptBytesString(uint64_t const bytes);
sptValue sptRandomValue(void);


/**
 * OMP Lock functions
 */
sptMutexPool * sptMutexAlloc();
sptMutexPool * SptMutexAllocCustom(
    sptIndex const num_locks,
    sptIndex const pad_size);
void sptMutexFree(sptMutexPool * pool);

static inline sptIndex sptMutexTranslateId(
    sptIndex const id,
    sptIndex const num_locks,
    sptIndex const pad_size)
{
  return (id % num_locks) * pad_size;
}

static inline void sptMutexSetLock(
    sptMutexPool * const pool,
    sptIndex const id)
{
  sptIndex const lock_id = sptMutexTranslateId(id, pool->nlocks, pool->padsize);
  omp_set_lock(pool->locks + lock_id);
}

static inline void sptMutexUnsetLock(
    sptMutexPool * const pool,
    sptIndex const id)
{
  sptIndex const lock_id = sptMutexTranslateId(id, pool->nlocks, pool->padsize);
  omp_unset_lock(pool->locks + lock_id);
}


#endif