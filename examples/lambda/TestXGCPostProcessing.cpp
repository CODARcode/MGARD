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
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "LagrangeOptimizer.hpp"
#include "adios2.h"
#include "mgard/compress_x.hpp"

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

  double tol, s = 0, bigtest = 1;

  int i = 1;
  std::string infile = argv[i++];
  std::string meshfile = argv[i++];

  tol = atof(argv[i++]);
  s = atof(argv[i++]);
  double job_sz = 1.0;
  // double job_sz = atof(argv[i++]);
  // bigtest = atof(argv[i++]);
  if (rank == 0) {
    printf("Input data: %s ", infile.c_str());
    printf("Abs. error bound: %.2e ", tol);
    printf("S: %.2f\n", s);
  }

  adios2::ADIOS ad("", MPI_COMM_WORLD);
  adios2::IO reader_io = ad.DeclareIO("XGC");
  adios2::Engine reader = reader_io.Open(infile.c_str(), adios2::Mode::Read);
  adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
  adios2::Engine writer =
      bpIO.Open("xgc_compressed.mgard.bp", adios2::Mode::Write);
  adios2::Engine writer_lag =
      bpIO.Open("xgc_lagrange.mgard.bp", adios2::Mode::Write);

  adios2::Variable<double> var_i_f_in;
  var_i_f_in = reader_io.InquireVariable<double>("i_f");
  if (!var_i_f_in) {
    std::cout << "Didn't find i_f...exit\n";
    exit(1);
  }
  int vxIndex = 2;
  int vyIndex = 3;
  int nodeIndex = 1;
  int planeIndex = 0;
  if (bigtest) {
    vxIndex = 1;
    vyIndex = 3;
    nodeIndex = 2;
    planeIndex = 0;
  }
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
  double *in_buff = (double *)malloc(sizeof(double) * local_elements);
  // if (rank == 0) {
  // std::cout << "total data size: {" << nphi << ", " << nnodes << ", "
  // << vx << ", " << vy << "}, number of iters: " << num_iter << "\n";
  // }
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
    std::vector<mgard_x::SIZE> shape = {nphi, local_nnodes, vx, vy};
    if (bigtest) {
      shape[1] = vx;
      shape[2] = local_nnodes;
    }
    long unsigned int offset = div_nnodes * iter + iter_nnodes * rank;
    long unsigned int offset_lag = offset * 4;
    adios2::Variable<double> bp_ldata = bpIO.DefineVariable<double>(
        "lag_p", {nphi * nnodes * 4}, {offset * 4}, {nphi * local_nnodes * 4});
    // std::cout << "rank " << rank << " read from {0, 0, "
    // << offset << ", 0} for {" << nphi << ", " << vx << ", "
    // << local_nnodes << ", " << vy << "}\n";
    std::vector<unsigned long> dim1 = {0, 0, offset, 0};
    std::vector<unsigned long> dim2 = {nphi, vx, local_nnodes, vy};
    std::pair<std::vector<unsigned long>, std::vector<unsigned long>> dim;
    dim.first = dim1;
    dim.second = dim2;
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>(dim));
    reader.Get<double>(var_i_f_in, in_buff);
    reader.PerformGets();
    double maxv = 0;
    for (size_t i = 0; i < local_elements; i++)
      maxv = (maxv > in_buff[i]) ? maxv : in_buff[i];
    // std::cout << "max element: " << maxv << "\n";
    if (rank == 0) {
      in_time = -MPI_Wtime();
    }
    if (rank == 0) {
      gpu_in_time += (in_time + MPI_Wtime());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      compress_time = -MPI_Wtime();
    }

    void *mgard_compressed_buff = NULL;
    size_t mgard_compressed_size;
    mgard_x::Config config;
    mgard_x::compress(4, mgard_x::data_type::Double, shape, tol, s,
                      mgard_x::error_bound_type::ABS, in_buff,
                      mgard_compressed_buff, mgard_compressed_size, config,
                      false);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      gpu_compress_time += (compress_time + MPI_Wtime());
    }
    out_size += mgard_compressed_size;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      decompress_time = -MPI_Wtime();
    }

    void *mgard_out_buff = NULL;
    mgard_x::decompress(mgard_compressed_buff, mgard_compressed_size,
                        mgard_out_buff, false);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      gpu_decompress_time += (decompress_time + MPI_Wtime());
    }

    LagrangeOptimizer optim("ion", "double");
    optim.computeParamsAndQoIs(meshfile.c_str(), dim1, dim2, in_buff);
    const double *lagranges =
        optim.computeLagrangeParameters((double *)mgard_out_buff);

    lagrange_size += nphi * local_nnodes * 4;

    adios2::Variable<double> bp_xgcdata = bpIO.DefineVariable<double>(
        "i_f", {nphi, vx, nnodes, vy}, {0, 0, offset, 0},
        {nphi, vx, local_nnodes, vy});
    bp_xgcdata.SetSelection(adios2::Box<adios2::Dims>(
        {0, 0, offset, 0}, {nphi, vx, local_nnodes, vy}));
    writer.Put<double>(bp_xgcdata, (double *)mgard_compressed_buff);
    writer.PerformPuts();

    bp_ldata.SetSelection(
        adios2::Box<adios2::Dims>({offset * 4}, {nphi * local_nnodes * 4}));
    writer_lag.Put<double>(bp_ldata, lagranges);
    writer_lag.PerformPuts();
    free(mgard_out_buff);
    free(mgard_compressed_buff);
  }
  writer_lag.Close();
  writer.Close();
  if (rank == 0) {
    std::cout << " CPU to GPU time: " << gpu_in_time
              << ", compression time: " << gpu_compress_time
              << ", decompress time: " << gpu_decompress_time << "\n";
  }

  free(in_buff);
  size_t gb_compressed, gb_compressed_lag;
  MPI_Allreduce(&out_size, &gb_compressed, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&lagrange_size, &gb_compressed_lag, 1, MPI_UNSIGNED_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("In size:  %10ld  Out size: %10ld  Lagrange size: %10ld  "
           "Compression ratio: %f \n",
           lSize, gb_compressed, gb_compressed_lag,
           (double)lSize / (gb_compressed + gb_compressed_lag));
  }
  reader.Close();

  MPI_Finalize();

  return 0;
}
