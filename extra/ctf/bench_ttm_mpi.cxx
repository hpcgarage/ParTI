/** Copyright (c) 2011, Edgar Solomonik, all rights reserved.
  * \addtogroup benchmarks
  * @{
  * \addtogroup bench_contractions
  * @{
  * \brief Benchmarks arbitrary NS contraction
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <vector>
#include "../src/shared/util.h"
#include "../src/interface/world.h"
#include "../src/interface/tensor.h"
#include "../src/interface/back_comp.h"

using namespace CTF;

CTF_Tensor *make_prod(
    bool is_sparse,
    CTF_World &dw,
    std::string const & name,
    std::vector<int>  ndimsA,
    std::vector<int> &ndimsB,
    int  mode
) {
    int nmodes = ndimsA.size();
    ndimsA[mode] = ndimsB[1];
    std::vector<int> ns(nmodes, NS);
    CTF_Tensor *result = new CTF_Tensor(nmodes, is_sparse, ndimsA.data(), ns.data(), dw, Ring<double>(), name.c_str(), true);
    return result;
}

CTF_Tensor *make_A(
    bool is_sparse,
    CTF_World &dw,
    std::string const & name,
    std::vector<int>  ndimsA
) {
    int nmodes = ndimsA.size();
    std::vector<int> ns(nmodes, NS);
    CTF_Tensor *result = new CTF_Tensor(nmodes, is_sparse, ndimsA.data(), ns.data(), dw, Ring<double>(), name.c_str(), true);
    return result;
}


CTF_Tensor *make_B(
    CTF_World &dw,
    std::string const & name,
    std::vector<int> const &ndimsA,
    std::vector<int> &ndimsB,
    int mode,
    int sizeB
) {
    ndimsB.resize(2);
    ndimsB[0] = ndimsA[mode];
    ndimsB[1] = sizeB;
    std::vector<int> ns(2, NS);
    CTF_Tensor *result = new CTF_Tensor(2, false, ndimsB.data(), ns.data(), dw, Ring<double>(), name.c_str(), true);
    return result;
}

void load_tensor(
    std::string const & filename,
    std::vector<int> &ndims,
    std::vector<int64_t> &inds,
    std::vector<double> &values
) 
{
    int nmodes;
    FILE *f;
    f = fopen(filename.c_str(), "r");
    if(!f) {
        fprintf(stderr, "unable to open %s\n", filename.c_str());
        exit(2);
    }
    printf("Reading from %s.\n", filename.c_str());
    fscanf(f, "%d", &nmodes);
    ndims.resize(nmodes);
    for(int i = 0; i < nmodes; ++i) {
        fscanf(f, "%d", &ndims[i]);
    }

    for(;;) {
        long ii, jj, kk;
        assert(nmodes == 3);
        if(fscanf(f, "%ld%ld%ld", &ii, &jj, &kk) == 3) {
            ii--; jj--; kk--; // offset 1
            int64_t global_idx = ii + jj*ndims[0] + kk*ndims[0]*ndims[1];
            inds.push_back(global_idx);
        } else {
            goto read_done;
        }
        double v;
        if(fscanf(f, "%lf", &v) == 1) {
            values.push_back(v);
        } else {
            goto read_done;
        }
    }
read_done:
    fclose(f);
    printf("Read from %s, %ld records.\n", filename.c_str(), (long) values.size());

    return;
}


CTF_Tensor *read_tensor(
    std::string const & filename,
    bool is_sparse,
    CTF_World &dw,
    std::string const & name,
    std::vector<int> &ndims
) 
{
    int rank, num_pes;
    MPI_Comm comm = dw.comm;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_pes);

    int nmodes = 3;
    ndims.push_back(500);
    ndims.push_back(500);
    ndims.push_back(500);

    CTF_Tensor *result = make_A(true, dw, "A", ndims);
    result->fill_sp_random(-1,1,.001);
    for(int n: ndims) {
      std::cout<< n << " ";
    }
    std::cout<<"\n";


    // int nmodes;
    // std::vector<int64_t> inds;
    // std::vector<double> values;

    // if(rank == 0) {
    //   load_tensor(filename, ndims, inds, values);
    //   nmodes = ndims.size();      
    // }
    // MPI_Bcast(&nmodes, 1, MPI_INT, 0, comm); 
    // printf("[P%d] nmodes: %d\n", rank, nmodes);
    // fflush(stdout);
    // if(rank != 0) ndims.resize(nmodes);
    // MPI_Bcast(ndims.data(), nmodes, MPI_INT, 0, comm); 
    // for(int n: ndims) {
    //   std::cout<< n << " ";
    // }
    // std::cout<<"\n";

    // CTF_Tensor *result = make_A(true, dw, "A", ndims);

    // int values_size;
    // int64_t * inds_vec;
    // double * values_vec;
    // if(rank == 0) values_size = values.size();
    // MPI_Bcast(&values_size, 1, MPI_INT, 0, comm); 
    // printf("values_size: %d\n", values_size);
    // int own_values_size;
    // own_values_size = values_size % num_pes == 0 ? values_size / num_pes: values_size / num_pes + 1;
    // inds_vec = (int64_t *)malloc(own_values_size * sizeof(int64_t));
    // MPI_Scatter(inds.data(), own_values_size, MPI_INT64_T, inds_vec, own_values_size, MPI_INT64_T, 0, comm);
    // values_vec = (double *)malloc(own_values_size * sizeof(double));
    // MPI_Scatter(values.data(), own_values_size, MPI_DOUBLE, values_vec, own_values_size, MPI_DOUBLE, 0, comm);
    // result->write(own_values_size, inds_vec, values_vec);

    return result;
}

int bench_contraction(
                      int          niter,
                      int          mode,
                      std::string const & filenamea,
                      std::string const & filenameb,
                      CTF_World   &dw
                 ){

  int rank, i, num_pes;
  MPI_Comm comm = dw.comm;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_pes);

  char iA[] = "012";
  char iB[] = "-3";
  char iC[] = "012";
  iB[0] = iA[mode];
  iC[mode] = '3';

  static std::vector<int> ndimsA;
  std::vector<int> ndimsB;
  static CTF_Tensor *A = NULL;
  if(!A) { // read only once
      A = read_tensor(filenamea, true, dw, "A", ndimsA);
  }
  CTF_Tensor *B = make_B(dw, "B", ndimsA, ndimsB, mode, 16);
  B->fill_random(-1, 1);
  CTF_Tensor *C = make_prod(false, dw, "C", ndimsA, ndimsB, mode);

  (*C)[iC] = (*A)[iA]*(*B)[iB]; // warm-up

  double st_time = MPI_Wtime();

  for (i=0; i<niter; i++){
    (*C)[iC] = (*A)[iA]*(*B)[iB];
  }

  double end_time = MPI_Wtime();

    printf("Performed %d iterations of mode %d in %lf sec/iter\n",
           niter, mode, (end_time-st_time)/niter);

  // A->print();
  // B->print();
  // C->print();

  delete C;
  delete B;

  return 1;
}

char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}


int main(int argc, char ** argv){
  int rank, np, niter;
  int const in_num = argc;
  char ** input_str = argv;
  char const * A;
  char const * B;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
  } else niter = 5;

  if (getCmdOption(input_str, input_str+in_num, "-A")){
    A = getCmdOption(input_str, input_str+in_num, "-A");
  } else A = "a.tns";
  if (getCmdOption(input_str, input_str+in_num, "-B")){
    B = getCmdOption(input_str, input_str+in_num, "-B");
  } else B = "b.tns";


  {
    CTF_World dw(MPI_COMM_WORLD, argc, argv);
    for(int mode = 2; mode >= 0; --mode) {
      bench_contraction(niter, mode, A, B, dw);
    }
  }


  MPI_Finalize();
  return 0;
}
/**
 * @}
 * @}
 */
