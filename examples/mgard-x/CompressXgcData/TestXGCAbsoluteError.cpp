/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
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
#include "mgard/mgard_api.h"

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
  double tol, s = 0;

  int i = 1;
  infile = argv[i++];
  tol = atof(argv[i++]);
  s = atof(argv[i++]);
  double job_sz = atof(argv[i++]);
  if (rank == 0) {
    printf("Input data: %s ", infile);
    printf("Abs. error bound: %.2e ", tol);
    printf("S: %.2f\n", s);
  }

  adios2::ADIOS ad("", MPI_COMM_WORLD);
  adios2::IO reader_io = ad.DeclareIO("XGC");
  adios2::Engine reader = reader_io.Open(infile, adios2::Mode::Read);
  adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
  adios2::Engine writer = bpIO.Open("xgc.mgard.bp", adios2::Mode::Write);

  adios2::Variable<double> var_i_f_in;
  var_i_f_in = reader_io.InquireVariable<double>("i_f");
  if (!var_i_f_in) {
    std::cout << "Didn't find i_f...exit\n";
    exit(1);
  }
  size_t vx = var_i_f_in.Shape()[1];
  size_t vy = var_i_f_in.Shape()[3];
  size_t nnodes = var_i_f_in.Shape()[2];
  size_t nphi = var_i_f_in.Shape()[0];
  size_t gb_elements = nphi * vx * nnodes * vy;
  size_t num_iter =
      (size_t)(std::ceil)((double)gb_elements * sizeof(double) / 1024.0 /
                          1024.0 / 1024.0 / job_sz / np_size);
  size_t div_nnodes = (size_t)(std::ceil)((double)nnodes / num_iter);
  size_t iter_nnodes =
      (size_t)(std::ceil)((double)div_nnodes / (double)np_size);
  //    size_t iter_elements = iter_nnodes * vx * vy * nphi;
  size_t local_nnodes =
      (rank == np_size - 1) ? (div_nnodes - rank * iter_nnodes) : iter_nnodes;
  size_t local_elements = nphi * vx * local_nnodes * vy;
  size_t lSize = sizeof(double) * gb_elements;
  double *in_buff;
  mgard_x::cudaMallocHostHelper((void **)&in_buff,
                                sizeof(double) * local_elements);
  if (rank == 0) {
    std::cout << "total data size: {" << nphi << ", " << vx << ", " << nnodes
              << ", " << vy << "}, number of iters: " << num_iter << "\n";
  }
  size_t out_size = 0;
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
    std::vector<size_t> shape = {nphi, vx, local_nnodes, vy};
    std::cout << "rank " << rank << " read from {0, 0, "
              << div_nnodes * iter + iter_nnodes * rank << ", 0} for {" << nphi
              << ", " << vx << ", " << local_nnodes << ", " << vy << "}\n";
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>(
        {0, 0, div_nnodes * iter + iter_nnodes * rank, 0},
        {nphi, vx, local_nnodes, vy}));
    reader.Get<double>(var_i_f_in, in_buff);
    reader.PerformGets();

    double maxv = 0;
    for (size_t i = 0; i < local_elements; i++)
      maxv = (maxv > in_buff[i]) ? maxv : in_buff[i];
    std::cout << "max element: " << maxv << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    double *mgard_out_buff = NULL;
    //        printf("Start compressing and decompressing with GPU\n");
    mgard_x::Array<4, double> in_array(shape);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      in_time = -MPI_Wtime();
    }
    in_array.loadData(in_buff);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      gpu_in_time += (in_time + MPI_Wtime());
    }
    //        std::cout << "loadData: " << shape[0] << ", " << shape[1] << ", "
    //        << shape[2] << ", " << shape[3] << "\n";

    mgard_x::Handle<4, double> handle(shape);
    //        std::cout << "before compression\n";
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      compress_time = -MPI_Wtime();
    }
    mgard_x::Array<1, unsigned char> compressed_array =
        mgard_x::compress(handle, in_array, mgard_x::ABS, tol, s);
    //        std::cout << "after compression\n";
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      gpu_compress_time += (compress_time + MPI_Wtime());
    }
    out_size += compressed_array.getShape()[0];

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      decompress_time = -MPI_Wtime();
    }
    mgard_x::Array<4, double> out_array =
        mgard_x::decompress(handle, compressed_array);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      gpu_decompress_time += (decompress_time + MPI_Wtime());
    }
    mgard_out_buff = new double[local_elements];
    memcpy(mgard_out_buff, out_array.getDataHost(),
           local_elements * sizeof(double));

    double error_L_inf_norm = 0;
    for (int i = 0; i < local_elements; ++i) {
      double temp = fabs(in_buff[i] - mgard_out_buff[i]);
      if (temp > error_L_inf_norm)
        error_L_inf_norm = temp;
    }
    double absolute_L_inf_error = error_L_inf_norm;

    printf("Abs. L^infty error bound: %10.5E \n", tol);
    printf("Abs. L^infty error: %10.5E \n", absolute_L_inf_error);

    if (absolute_L_inf_error < tol) {
      printf(ANSI_GREEN "SUCCESS: Error tolerance met!" ANSI_RESET "\n");
    } else {
      printf(ANSI_RED "FAILURE: Error tolerance NOT met!" ANSI_RESET "\n");
      return -1;
    }

    char write_f[2048];
    sprintf(write_f, "largeXGC/xgc.mgard.rank%i_%d.bin", rank, iter);
    FILE *pFile = fopen(write_f, "wb");
    fwrite(mgard_out_buff, sizeof(double), out_size, pFile);
    fclose(pFile);
    delete mgard_out_buff;
  }
  if (rank == 0) {
    std::cout << " CPU to GPU time: " << gpu_in_time
              << ", compression time: " << gpu_compress_time
              << ", decompress time: " << gpu_decompress_time << "\n";
  }

  mgard_x::cudaFreeHostHelper(in_buff);
  size_t gb_compressed;
  MPI_Allreduce(&out_size, &gb_compressed, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  if (rank == 0) {
    printf("In size:  %10ld  Out size: %10ld  Compression ratio: %f \n", lSize,
           gb_compressed, (double)lSize / gb_compressed);
  }
  reader.Close();

  //    size_t exscan;
  //    size_t *scan_counts = (size_t *)malloc(np_size * sizeof(size_t));
  //    MPI_Exscan(&out_size, &exscan, 1, MPI_UNSIGNED_LONG, MPI_SUM,
  //    MPI_COMM_WORLD); std::cout << "rank " << rank << " compressed size: " <<
  //    out_size << ", exscan: " << exscan << ", total compressed: " <<
  //    gb_compressed << "\n"; MPI_Gather(&exscan, 1, MPI_UNSIGNED_LONG,
  //    scan_counts, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD); if (rank == 0) {
  //        scan_counts[0] = 0;
  //        exscan = 0;
  //        std::cout << "scanned counts: ";
  //        for (int i=0; i<np_size; i ++)
  //            std::cout << scan_counts[i] << ", ";
  //        std::cout << "\n";
  //    }
  //    unsigned char *mgard_compress_buff = new unsigned char[out_size];
  //    memcpy(mgard_compress_buff, compressed_array.getDataHost(), out_size);

  //    adios2::Variable<unsigned char> bp_fdata = bpIO.DefineVariable<unsigned
  //    char>(
  //      "mgard_f", {gb_compressed}, {exscan}, {out_size},
  //      adios2::ConstantDims);
  //    writer.Put<unsigned char>(bp_fdata, mgard_compress_buff);
  //    if (rank==0) {
  //        adios2::Variable<size_t> bp_count = bpIO.DefineVariable<size_t>(
  //        "block_start", {(size_t)np_size}, {0}, {(size_t)np_size},
  //        adios2::ConstantDims); writer.Put<size_t>(bp_count, scan_counts);
  //    }
  //    writer.Close();

  MPI_Finalize();

  return 0;
}
