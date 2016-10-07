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

CTF_Tensor *read_tensor(
    std::string const & filename,
    bool is_sparse,
    CTF_World &dw,
    std::string const & name,
    std::vector<int> &ndims
) {
    FILE *f = fopen(filename.c_str(), "r");
    if(!f) {
        fprintf(stderr, "unable to open %s\n", filename.c_str());
        exit(2);
    }
    int nmodes;
    fscanf(f, "%d", &nmodes);
    ndims.resize(nmodes);
    for(int i = 0; i < nmodes; ++i) {
        fscanf(f, "%d", &ndims[i]);
    }
    std::vector<int> ns(nmodes, NS);
    CTF_Tensor *result = new CTF_Tensor(nmodes, is_sparse, ndims.data(), ns.data(), dw, Ring<double>(), name.c_str(), true);
    std::vector<int64_t> &inds = *new std::vector<int64_t>;
    std::vector<double> &values = *new std::vector<double>; // leak it.
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
    result->write(values.size(), inds.data(), values.data());
    fclose(f);
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

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  char iA[] = "012";
  char iB[] = "-3";
  char iC[] = "012";
  iB[0] = iA[mode];
  iC[mode] = '3';

  std::vector<int> ndimsA, ndimsB;
  CTF_Tensor *A = read_tensor(filenamea, true, dw, "A", ndimsA);
  CTF_Tensor *B = make_B(dw, "B", ndimsA, ndimsB, mode, 16);
  B->fill_random(-1, 1);
  CTF_Tensor *C = make_prod(false, dw, "C", ndimsA, ndimsB, mode);

  //////////////////////////////////////////////////
  // TODO: read tensor from "filename" into A, B  //
  //////////////////////////////////////////////////

  double st_time = MPI_Wtime();

  for (i=0; i<niter; i++){
    (*C)[iC] = (*A)[iA]*(*B)[iB];
  }

  double end_time = MPI_Wtime();

    printf("Performed %d iterations of mode %d in %lf sec/iter\n",
           niter, mode, (end_time-st_time)/niter);

  A->print();
  B->print();
  C->print();

  delete C;
  delete B;
  delete A;

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
  } else niter = 1000;

  if (getCmdOption(input_str, input_str+in_num, "-A")){
    A = getCmdOption(input_str, input_str+in_num, "-A");
  } else A = "a.tns";
  if (getCmdOption(input_str, input_str+in_num, "-B")){
    B = getCmdOption(input_str, input_str+in_num, "-B");
  } else B = "b.tns";


  {
    CTF_World dw(argc, argv);
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
