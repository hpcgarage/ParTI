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

#include <ParTI/algorithm.hpp>
#include <cassert>
#include <memory>
#include <vector>
#include <ParTI/error.hpp>
#include <ParTI/utils.hpp>


namespace pti {

namespace {
}


void set_semisparse_indices_by_sparse_ref_scan_cuda(SparseTensor& dest, std::vector<size_t>& fiber_idx, SparseTensor& ref, size_t mode, CudaDevice* cuda_dev) {
#if 0
    assert(dest.nmodes == ref.nmodes);

    dest.num_chunks = 0;
    std::unique_ptr<size_t[]> indices(new size_t [ref.nmodes]);
    std::unique_ptr<Scalar[]> chunk(new Scalar [dest.chunk_size] ());

    /* Temporary array */
    // TODO: malloc start_flags on GPU mem; malloc and memcpy indices.
    std::unique_ptr<size_t[]> start_flags(new size_t [ref.num_chunks]);
    memset(start_flags.get(), 0, ref.num_chunks * sizeof(size_t));

    int const nthreads = 256;
    size_t nblocks = (ref.num_chunks + nthreads - 1) / nthreads;
    set_start_tags<<<nblocks, nthreads>>>(start_flags, ref.num_chunks, )

    Timer timer_tags(cpu);
    timer_tags.start();
    #pragma omp parallel
    {
        /* Mark fiber starts */
        
        
        // if (tid == 0) {
        //     printf("nthreads: %d\n", nthreads);
        //     printf("num_chunks_aver: %zu\n", num_chunks_aver);
        //     fflush(stdout);
        // }
        size_t num_chunks_end = ((tid + 1) * num_chunks_aver > ref.num_chunks) ? ref.num_chunks : (tid + 1) * num_chunks_aver;
        size_t lastidx = (tid == 0) ? ref.num_chunks : tid * num_chunks_aver - 1;
        // printf("[%d] begin: %zu, end: %zu\n", tid, tid * num_chunks_aver, num_chunks_end);
        for(size_t i = tid * num_chunks_aver; i < num_chunks_end; ++i) {
            if(lastidx == ref.num_chunks || compare_indices(ref, lastidx, i, mode) != 0) {
                start_flags[i] = 1;
                lastidx = i;
            }
        }

    }
    timer_tags.stop();
    timer_tags.print_elapsed_time("OMP Set Tag");
    // printf("start_flags 1: \n");
    // for(size_t i = 0; i < ref.num_chunks; ++i) {
    //     printf("%zu ", start_flags[i]);
    // }
    // printf("\n"); 
    // fflush(stdout);

    Timer timer_scan(cpu);
    timer_scan.start();
    scan_omp (start_flags.get(), ref.num_chunks);
    size_t dest_num_chunks = start_flags[ref.num_chunks - 1];
    timer_scan.stop();
    timer_scan.print_elapsed_time("OMP Scan");
    // printf("start_flags 2: \n");
    // for(size_t i = 0; i < ref.num_chunks; ++i) {
    //     printf("%zu ", start_flags[i]);
    // }
    // printf("\n"); 
    // fflush(stdout);

    Timer timer_allocate(cpu);
    timer_allocate.start();
    fiber_idx.resize(dest_num_chunks + 1);
    dest.reserve(dest_num_chunks, false);
    dest.num_chunks = dest_num_chunks;
    timer_allocate.stop();
    timer_allocate.print_elapsed_time("Allocate");
    

    Timer timer_fiberidx(cpu);
    timer_fiberidx.start();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t nthreads = omp_get_num_threads();
        size_t num_chunks_aver = (ref.num_chunks + nthreads - 1) / nthreads;
        size_t num_chunks_end = ((tid + 1) * num_chunks_aver > ref.num_chunks) ? ref.num_chunks : (tid + 1) * num_chunks_aver;
        size_t lastidx = (tid == 0) ? dest_num_chunks : start_flags[tid * num_chunks_aver - 1] - 1;
        for(size_t i = tid * num_chunks_aver; i < num_chunks_end; ++i) {
            size_t loc = start_flags[i] - 1;
            if(lastidx == dest_num_chunks || loc != lastidx) {
                bool inbound = ref.offset_to_indices(indices.get(), i * ref.chunk_size);
                assert(inbound);
                dest.put(loc, indices.get(), chunk.get());
                fiber_idx[loc] = i;
                lastidx = loc;
            }
        }

    }
    fiber_idx[dest_num_chunks] = ref.num_chunks;
    timer_fiberidx.stop();
    timer_fiberidx.print_elapsed_time("Assign Fiberidx");
    
#endif
}


}