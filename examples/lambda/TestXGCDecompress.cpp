/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "adios2.h"
#include "mgard/compress_x.hpp"
#include "LagrangeOptimizer.hpp"

#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_RESET "\x1b[0m"

using namespace std::chrono;

void print_usage_message(char *argv[], FILE *fp) {
  fprintf(fp,
          "Usage: %s [input file] [num. of dimensions] [1st dim.] [2nd dim.] "
          "[3rd. dim] ... [tolerance] [s]\n",
          argv[0]);
}

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  int rank, np_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np_size);
  //    np_size = 150;
  /*
      int deviceCount;
      cudaGetDeviceCount(&deviceCount);               // How many GPUs?
      int device_id = rank % deviceCount;
      cudaSetDevice(device_id);
      std::cout << "total number of devices: " << deviceCount << ", rank " <<
     rank << " used " << device_id << "\n";
  */

  double compress_time = 0.0;
  double decompress_time = 0.0;
  double gpu_compress_time = 0.0;
  double gpu_decompress_time = 0.0;
  double in_time = 0.0;
  double gpu_in_time = 0.0;
  if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
    print_usage_message(argv, stdout);
    return 0;
  }

  char *infile; //, *outfile;
  double tol, s = 0, bigtest = 1;

  int i = 1;
  infile = argv[i++];
  char* meshfile = argv[i++];
  char* lagfile = argv[i++];
  double job_sz = atof(argv[i++]);
  if (rank == 0) {
    printf("Input data: %s ", infile);
    printf("Abs. error bound: %.2e ", tol);
    printf("S: %.2f\n", s);
  }

  adios2::ADIOS ad("", MPI_COMM_WORLD);
  adios2::IO reader_io = ad.DeclareIO("XGC");
  adios2::Engine reader = reader_io.Open(infile, adios2::Mode::Read);
  adios2::IO reader_lag_io = ad.DeclareIO("Lambdas");
  adios2::Engine reader_lag = reader_lag_io.Open(lagfile, adios2::Mode::Read);

  adios2::Variable<double> var_i_f_in;
  var_i_f_in = reader_io.InquireVariable<double>("i_f");
  if (!var_i_f_in) {
    std::cout << "Didn't find i_f...exit\n";
    exit(1);
  }
  int vxIndex = 1;
  int vyIndex = 3;
  int nodeIndex = 2;
  int planeIndex = 0;
  mgard_x::SIZE vx = var_i_f_in.Shape()[vxIndex];
  mgard_x::SIZE vy = var_i_f_in.Shape()[vyIndex];
  mgard_x::SIZE nnodes = var_i_f_in.Shape()[nodeIndex];
  mgard_x::SIZE nphi = var_i_f_in.Shape()[planeIndex];
  size_t gb_elements = nphi * vx * nnodes * vy;
  size_t num_iter =
      (size_t)(std::ceil)((double)gb_elements * sizeof(double) / 1024.0 /
                          1024.0 / 1024.0 / job_sz / np_size);
  size_t div_nnodes = (size_t)(std::ceil)((double)nnodes / num_iter);
  size_t iter_nnodes =
      (size_t)(std::ceil)((double)div_nnodes / (double)np_size);
  //    size_t iter_elements = iter_nnodes * vx * vy * nphi;
  mgard_x::SIZE local_nnodes =
      (rank == np_size - 1) ? (div_nnodes - rank * iter_nnodes) : iter_nnodes;
  size_t local_elements = nphi * vx * local_nnodes * vy;
  size_t lSize = sizeof(double) * gb_elements;
  double *in_buff = (double*)malloc(sizeof(double) * local_elements);
  if (rank == 0) {
    std::cout << "total data size: {" << nphi << ", " << nnodes << ", "
      << vx << ", " << vy << "}, number of iters: " << num_iter << "\n";
  }
  size_t out_size = 0;
  size_t lagrange_size = 0;
  for (size_t iter = 0; iter < num_iter; iter++) {
    if (iter == num_iter - 1) {
      iter_nnodes = (size_t)(std::ceil)(
          ((double)(nnodes - div_nnodes * iter)) /
          (double)np_size); // local_nnodes - iter_nnodes*iter;
      local_nnodes =
          (rank == np_size - 1)
              ? (nnodes - div_nnodes * iter - iter_nnodes * (np_size - 1))
              : iter_nnodes;
      local_elements = local_nnodes * vx * vy * nphi;
    }
    std::vector<mgard_x::SIZE> shape = {nphi, vx, local_nnodes, vy};
    long unsigned int offset = div_nnodes * iter + iter_nnodes * rank;
    if (bigtest) {
        std::cout << "rank " << rank << " read from {0, 0, "
              << offset << ", 0} for {" << nphi << ", " << vx << ", "
              << local_nnodes << ", " << vy << "}\n";
    }
    else {
        std::cout << "rank " << rank << " read from {0, "
              << offset << ", 0, 0} for {" << nphi
              << ", " << local_nnodes << ", " << vx<< ", " << vy << "}\n";
    }
    std::vector<unsigned long> dim1 = {0, 0, offset, 0};
    std::vector<unsigned long> dim2 = {nphi, vx, local_nnodes, vy};
    std::pair<std::vector<unsigned long>, std::vector<unsigned long>> dim;
    dim.first = dim1;
    dim.second = dim2;
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>(dim));
    reader.Get<double>(var_i_f_in, in_buff);
    reader.PerformGets();

    adios2::Variable<double> var_i_f_lag;
    var_i_f_lag = reader_lag_io.InquireVariable<double>("lag_p");
    if (!var_i_f_lag) {
      std::cout << "Didn't find lag_p...exit\n";
      exit(1);
    }
    std::vector<unsigned long> lag_dim1 = {offset*4};
    std::vector<unsigned long> lag_dim2 = {nphi*local_nnodes*4};
    std::pair<std::vector<unsigned long>, std::vector<unsigned long>> lag_dim;
    lag_dim.first = lag_dim1;
    lag_dim.second = lag_dim2;
    var_i_f_lag.SetSelection(adios2::Box<adios2::Dims>(lag_dim));
    std::vector<double> lag_buff;
    reader_lag.Get<double>(var_i_f_lag, lag_buff);
    reader_lag.PerformGets();

    LagrangeOptimizer optim("ion", "single");
    // in_buff gets modified after applying the Lagrange transformation
    optim.setDataFromCharBufferV1(in_buff, lag_buff.data(), meshfile);
  }
  reader.Close();
  reader_lag.Close();

  MPI_Finalize();

  return 0;
}
