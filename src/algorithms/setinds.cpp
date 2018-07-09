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
#include <ParTI/errcode.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/utils.hpp>
#include <ParTI/timer.hpp>
#include <omp.h>

namespace pti {

namespace {

int compare_indices(SparseTensor& tsr, size_t i, size_t j, size_t except) {
    size_t* sort_order = tsr.sparse_order(cpu);
    for(size_t m = 0; m < tsr.sparse_order.size(); ++m) {
        size_t mode = sort_order[m];
        if(mode == except) {
            continue;
        }
        size_t idx_i = tsr.indices[mode](cpu)[i];
        size_t idx_j = tsr.indices[mode](cpu)[j];
        if(idx_i < idx_j) {
            return -1;
        } else if(idx_i > idx_j) {
            return 1;
        }
    }
    return 0;
}

}

void set_semisparse_indices_by_sparse_ref(SparseTensor& dest, std::vector<size_t>& fiber_idx, SparseTensor& ref, size_t mode) {
    size_t lastidx = ref.num_chunks;
    assert(dest.nmodes == ref.nmodes);

    fiber_idx.clear();
    dest.num_chunks = 0;
    std::unique_ptr<size_t[]> indices(new size_t [ref.nmodes]);
    std::unique_ptr<Scalar[]> chunk(new Scalar [dest.chunk_size] ());
    for(size_t i = 0; i < ref.num_chunks; ++i) {
        if(lastidx == ref.num_chunks || compare_indices(ref, lastidx, i, mode) != 0) {
            bool inbound = ref.offset_to_indices(indices.get(), i * ref.chunk_size);
            if(!inbound) {
                std::fprintf(stderr, "[TTM SetIdx] Internal error: indices [%s] is out of bound.\n", array_to_string(indices.get(), ref.nmodes).c_str());
                ptiCheckError(!inbound, ERR_UNKNOWN, "Internal error");
            }
            dest.append(indices.get(), chunk.get());
            lastidx = i;
            fiber_idx.push_back(i);
        }
    }
    fiber_idx.push_back(ref.num_chunks);
}


void set_semisparse_indices_by_sparse_ref_scan_seq(SparseTensor& dest, std::vector<size_t>& fiber_idx, SparseTensor& ref, size_t mode) {
    assert(dest.nmodes == ref.nmodes);

    // fiber_idx.clear();
    dest.num_chunks = 0;
    std::unique_ptr<size_t[]> indices(new size_t [ref.nmodes]);
    std::unique_ptr<Scalar[]> chunk(new Scalar [dest.chunk_size] ());

    /* Temporary array */
    std::unique_ptr<size_t[]> start_flags(new size_t [ref.num_chunks]);
    memset(start_flags.get(), 0, ref.num_chunks * sizeof(size_t));

    size_t lastidx = ref.num_chunks;
    for(size_t i = 0; i < ref.num_chunks; ++i) {
        if(lastidx == ref.num_chunks || compare_indices(ref, lastidx, i, mode) != 0) {
            start_flags[i] = 1;
            lastidx = i;
        }
    }
    // printf("start_flags 1: \n");
    // for(size_t i = 0; i < ref.num_chunks; ++i) {
    //     printf("%zu ", start_flags[i]);
    // }
    // printf("\n"); 
    // fflush(stdout);

    scan_seq (start_flags.get(), ref.num_chunks);
    size_t dest_num_chunks = start_flags[ref.num_chunks - 1];
    // printf("start_flags 2: \n");
    // for(size_t i = 0; i < ref.num_chunks; ++i) {
    //     printf("%zu ", start_flags[i]);
    // }
    // printf("\n"); 
    // fflush(stdout);


    fiber_idx.resize(dest_num_chunks + 1);
    dest.reserve(dest_num_chunks, false);
    dest.num_chunks = dest_num_chunks;
    

    lastidx = dest_num_chunks;
    for(size_t i = 0; i < ref.num_chunks; ++i) {
        size_t loc = start_flags[i] - 1;
        if(lastidx == dest_num_chunks || loc != lastidx) {
            bool inbound = ref.offset_to_indices(indices.get(), i * ref.chunk_size);
            assert(inbound);
            dest.put(loc, indices.get(), chunk.get());
            fiber_idx[loc] = i;
            lastidx = loc;
        }
    }
    fiber_idx[dest_num_chunks] = ref.num_chunks;
}


void set_semisparse_indices_by_sparse_ref_scan_omp_task(SparseTensor& dest, std::vector<size_t>& fiber_idx, SparseTensor& ref, size_t mode) {
    assert(dest.nmodes == ref.nmodes);

    dest.num_chunks = 0;
    std::unique_ptr<size_t[]> indices(new size_t [ref.nmodes]);
    std::unique_ptr<Scalar[]> chunk(new Scalar [dest.chunk_size] ());

    /* Temporary array */
    std::unique_ptr<size_t[]> start_flags(new size_t [ref.num_chunks]);
    memset(start_flags.get(), 0, ref.num_chunks * sizeof(size_t));

    Timer timer_tags(cpu);
    timer_tags.start();
    #pragma omp parallel
    {
        /* Mark fiber starts */
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        size_t num_chunks_aver = (ref.num_chunks + nthreads - 1) / nthreads;
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
    
}



void set_semisparse_indices_by_sparse_ref_scan_omp(SparseTensor& dest, std::vector<size_t>& fiber_idx, SparseTensor& ref, size_t mode) {
    assert(dest.nmodes == ref.nmodes);

    dest.num_chunks = 0;
    std::unique_ptr<size_t[]> indices(new size_t [ref.nmodes]);
    std::unique_ptr<Scalar[]> chunk(new Scalar [dest.chunk_size] ());

    /* Temporary array */
    std::unique_ptr<size_t[]> start_flags(new size_t [ref.num_chunks]);
    memset(start_flags.get(), 0, ref.num_chunks * sizeof(size_t));
    size_t dest_num_chunks;
    size_t * partial;

    Timer timer_tags(cpu);
    timer_tags.start();
    #pragma omp parallel shared(dest_num_chunks)
    {
        /* Mark fiber starts */
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        size_t num_chunks_aver = (ref.num_chunks + nthreads - 1) / nthreads;
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
        #pragma omp barrier

        /* Prefix sum */
        #pragma omp single
        {
            partial = (size_t*)malloc((nthreads+1) * sizeof(*partial));
            partial[0] = 0;
        }
        for(size_t i = tid * num_chunks_aver + 1; i < num_chunks_end; ++i) {
            start_flags[i] += start_flags[i-1];
        }
        partial[tid + 1] = start_flags[num_chunks_end - 1];
        
        #pragma omp barrier

        #pragma omp single
        {
            for(int i = 1; i < nthreads; ++i) {
                partial[i] += partial[i-1];
            }
        }
        #pragma omp barrier

        for(size_t i = tid * num_chunks_aver; i < num_chunks_end; ++i) {
            start_flags[i] += partial[tid];
        }
        #pragma omp barrier

        /* Allocate space */
        #pragma omp single
        {
            dest_num_chunks = start_flags[ref.num_chunks - 1];
            fiber_idx.resize(dest_num_chunks + 1);
            dest.reserve(dest_num_chunks, false);
            dest.num_chunks = dest_num_chunks;
        }

        /* Write to fiberidx in parallel */
        lastidx = (tid == 0) ? dest_num_chunks : start_flags[tid * num_chunks_aver - 1] - 1;
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
        #pragma omp single
        {
            fiber_idx[dest_num_chunks] = ref.num_chunks;
        }

    }   // End omp parallel
    
}

}
