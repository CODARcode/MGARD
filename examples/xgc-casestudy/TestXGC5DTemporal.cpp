#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <vector>

#include "adios2.h"
#include "mgard/compress.hpp"

#define SECONDS(d) std::chrono::duration_cast<std::chrono::seconds>(d).count()

template <typename Type>
void FileWriter_ad(const char *filename, Type *data,
                   std::vector<size_t> global_dim,
                   std::vector<size_t> local_dim, size_t para_dim,
                   size_t timeSteps) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  adios2::ADIOS ad(MPI_COMM_WORLD);
  adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
  std::vector<std::string> var_nn(timeSteps);
  adios2::Engine bpFileWriter = bpIO.Open(filename, adios2::Mode::Write);
  for (size_t t = 0; t < timeSteps; t++) {
    size_t start_sz =
        t * local_dim[0] * local_dim[1] * local_dim[2] * local_dim[3];
    var_nn.at(t).assign("i_f_5d_t" + std::to_string(t));
    adios2::Variable<Type> bp_fdata = bpIO.DefineVariable<Type>(
        var_nn.at(t).c_str(), global_dim, {0, para_dim * rank, 0, 0}, local_dim,
        adios2::ConstantDims);
    // Engine derived class, spawned to start IO operations //
    bpFileWriter.Put<Type>(bp_fdata, &data[start_sz]);
  }
  bpFileWriter.Close();
}

template <typename Type>
void CheckReconstruction(Type *ori_buff, Type *rct_buff, double tol,
                         size_t local_sz) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  double gb_L_inf, error_L_inf_norm = 0;
  double gb_L2, error_L2 = 0;
  for (size_t it = 0; it < local_sz; it++) {
    double temp = fabs(ori_buff[it] - rct_buff[it]);
    if (temp > error_L_inf_norm)
      error_L_inf_norm = temp;
    error_L2 += temp * temp;
  }
  error_L2 = sqrt(error_L2 / local_sz);
  MPI_Allreduce(&error_L2, &gb_L2, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&error_L_inf_norm, &gb_L_inf, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  if (rank == 0) {
    if (gb_L2 < tol) {
      printf("SUCCESS: Error tolerance met!...requested %f, measured L2=%f, "
             "L-inf=%f\n",
             tol, gb_L2, gb_L_inf);
    } else {
      printf("FAILURE: Error tolerance NOT met!...requested %f, measured L2= "
             "%f, L-inf=%f\n",
             tol, gb_L2, gb_L_inf);
    }
  }
}

// MPI parallelize the second dimension -- # of mesh nodes
// argv[1]: error type
// argv[2]: number of timesteps
// argv[3],[4]: data path and filename
// input: n_phi x n_nodes x vx x vy
int main(int argc, char **argv) {
  if (argc != 6) {
    printf("Inputs: \n");
    printf("-- data files directory\n");
    printf("-- data file prefix (suffix is the timestep, 0, 1, 2, ...)\n");
    printf("-- eb \n");
    printf("-- snorm (default is 0)\n");
    printf("-- number of timesteps\n");
    return -1;
  }

  MPI_Init(&argc, &argv);
  int rank, np_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np_size);

  int nargv = 0;
  char datapath[2048], filename[2048], readin_f[2048], write_f[2048];
  strcpy(datapath, argv[++nargv]);
  strcpy(filename, argv[++nargv]);
  double tol = atof(argv[++nargv]);
  double s_norm = atof(argv[++nargv]);
  size_t timeSteps = atoi(argv[++nargv]);
  if (timeSteps < 2) {
    printf("Timestep must > 2\n");
    MPI_Finalize();
    return -1;
  }
  sprintf(write_f, "%s%s.ts_%ld.bp", filename, ".mgard", timeSteps);

  if (rank == 0) {
    printf("Read in: %s\n", readin_f);
    printf(" 4D + temporal copression\n");
    printf("Absolute error tolerance = %f\n", tol);
    printf("Snorm: %f\n", s_norm);
    printf("number of timeSteps: %ld\n", timeSteps);
  }
  adios2::ADIOS ad(MPI_COMM_WORLD);
  adios2::IO reader_io = ad.DeclareIO("XGC");

  unsigned char *compressed_data = 0;
  double *i_f_5d;
  size_t temp_dim, temp_sz, local_dim, local_sz;
  std::vector<std::size_t> shape(4);
  for (size_t ts = 0; ts < timeSteps; ts++) {
    if (ts == 0)
      sprintf(readin_f, "%s%s", datapath, filename);
    else {
      std::string ts_fn(filename);
      ts_fn.resize(ts_fn.size() - 5);
      sprintf(readin_f, "%s%s%ld.bp", datapath, ts_fn.c_str(), (ts)*10);
    }
    adios2::Engine reader = reader_io.Open(readin_f, adios2::Mode::Read);
    // Inquire variable
    adios2::Variable<double> var_i_f_in;
    var_i_f_in = reader_io.InquireVariable<double>("i_f");

    shape = var_i_f_in.Shape();
    if (ts == 0) {
      temp_dim = (size_t)ceil((double)shape[1] / np_size);
      local_dim =
          ((rank == np_size - 1) ? (shape[1] - temp_dim * rank) : temp_dim);
      temp_sz = temp_dim * shape[0] * shape[2] * shape[3];
      local_sz = local_dim * shape[0] * shape[2] * shape[3];
      i_f_5d = new double[local_sz * timeSteps];
      memset(i_f_5d, 0, sizeof(double) * (local_sz * timeSteps));
    }

    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>(
        {0, temp_dim * rank, 0, 0}, {shape[0], local_dim, shape[2], shape[3]}));
    reader.Get<double>(var_i_f_in, &i_f_5d[ts * local_sz]);
    reader.Close();
  }
  if (rank == 0) {
    printf("begin compression...\n");
  }
  size_t compressed_sz, gb_compressed;
  const std::array<std::size_t, 5> dims = {timeSteps, shape[0], local_dim,
                                           shape[2], shape[3]};
  const mgard::TensorMeshHierarchy<5, double> hierarchy(dims);
  const size_t ndof = hierarchy.ndof();
  const mgard::CompressedDataset<5, double> compressed =
      mgard::compress(hierarchy, i_f_5d, s_norm, tol);
  compressed_sz = compressed.size();
  MPI_Allreduce(&compressed_sz, &gb_compressed, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  const mgard::DecompressedDataset<5, double> decompressed =
      mgard::decompress(compressed);
  double data_L_inf_norm = 0;
  double *mgard_out_buff = (double *)decompressed.data();
  CheckReconstruction(i_f_5d, mgard_out_buff, tol, local_sz);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("Compression ratio = %.3f\n",
           ((double)8.0 * shape[0] * shape[1] * shape[2] * shape[3]) /
               gb_compressed);
  }
  FileWriter_ad(write_f, ((double *)decompressed.data()),
                {shape[0], shape[1], shape[2], shape[3]},
                {shape[0], local_dim, shape[2], shape[3]}, temp_dim, timeSteps);
  delete i_f_5d;
  MPI_Finalize();
  return 0;
}
