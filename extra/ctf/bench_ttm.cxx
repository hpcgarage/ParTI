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
#include "../src/shared/util.h"
#include "../src/interface/world.h"
#include "../src/interface/tensor.h"
#include "../src/interface/back_comp.h"

using namespace CTF;

int bench_contraction(
                      int          niter,
                      int          mode,
                      std::string  filename,
                      CTF_World   &dw){

  int rank, i, num_pes;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  int n_A[3] = {0, 1, 2};
  int n_B[2] = {0, 1};
  int n_C[3] = {0, 1, 2};
  int NS_[3] = {NS, NS, NS};
  const char *iB = "01";
  const char *iAC;
  switch(mode) {
  case 0:
    iAC = "120";
    break;
  case 1:
    iAC = "021";
    break;
  case 2:
    iAC = "012";
    break;
  default:
    assert(0);
  }

  //* Creates distributed tensors initialized with zeros
  CTF_Tensor A(3, true, n_A, NS_, dw, Ring<double>(), "A", true);
  CTF_Tensor B(2, true, n_B, NS_, dw, Ring<double>(), "B", true);
  CTF_Tensor C(3, true, n_C, NS_, dw, Ring<double>(), "C", true);

  //////////////////////////////////////////////////
  // TODO: read tensor from "filename" into A, B  //
  //////////////////////////////////////////////////

  double st_time = MPI_Wtime();

  for (i=0; i<niter; i++){
    C[iAC] += A[iAC]*B[iB];
  }

  double end_time = MPI_Wtime();

    printf("Performed %d iterations of mode %d in %lf sec/iter\n",
           niter, mode, (end_time-st_time)/niter);

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
  int rank, np, niter, n;
  int const in_num = argc;
  char ** input_str = argv;
  char const * A;
  char const * B;
  char const * C;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 4;
  } else n = 4;

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 3;
  } else niter = 3;

  if (getCmdOption(input_str, input_str+in_num, "-A")){
    A = getCmdOption(input_str, input_str+in_num, "-A");
  } else A = "ik";
  if (getCmdOption(input_str, input_str+in_num, "-B")){
    B = getCmdOption(input_str, input_str+in_num, "-B");
  } else B = "kj";
  if (getCmdOption(input_str, input_str+in_num, "-C")){
    C = getCmdOption(input_str, input_str+in_num, "-C");
  } else C = "ij";



  {
    CTF_World dw(argc, argv);
    for(int mode = 0; mode < 3; ++mode) {
      bench_contraction(niter, mode, "", dw);
    }
  }


  MPI_Finalize();
  return 0;
}
/**
 * @}
 * @}
 */
