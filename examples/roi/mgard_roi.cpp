#include <chrono>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <vector>

#include "adios2.h"
#include "mgard/compress.hpp"

#define SECONDS(d) std::chrono::duration_cast<std::chrono::seconds>(d).count()

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, np_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np_size);

  adios2::ADIOS ad(MPI_COMM_WORLD);
  int cnt_argv = 1;
  std::string dpath(argv[cnt_argv++]);
  std::string fname(argv[cnt_argv++]);
  std::string var_name = argv[cnt_argv++];
  float tol = std::stof(argv[cnt_argv++]);
  if (rank == 0)
    std::cout << "var_name: " << var_name << ", eb = " << tol << "\n";
  size_t ndims = std::stoi(argv[cnt_argv++]);
  std::vector<float> thresh;
  std::vector<size_t> init_bw;
  std::vector<size_t> bw_ratio;
  size_t l_th, amr_lv;
  amr_lv = std::stoi(argv[cnt_argv++]);
  l_th = std::stoi(argv[cnt_argv++]);
  if (rank == 0)
    std::cout << "amr_lv = " << amr_lv << ", l_th = " << l_th << "\n";
  for (size_t i = 0; i < ndims; i++) {
    init_bw.push_back(std::stoi(argv[cnt_argv++]));
    if (rank == 0)
      std::cout << "init_bw[" << i << "] = " << init_bw[i] << " ";
  }
  if (rank == 0)
    std::cout << "\n";
  for (size_t i = 0; i < amr_lv - 1; i++) {
    bw_ratio.push_back(std::stoi(argv[cnt_argv++]));
    if (rank == 0)
      std::cout << "bw_ratio[" << i << "] = " << bw_ratio[i] << " ";
  }
  if (rank == 0)
    std::cout << "\n";
  for (size_t i = 0; i < amr_lv; i++) {
    thresh.push_back(std::stof(argv[cnt_argv++]));
    if (rank == 0)
      std::cout << "thresh[" << i << "] = " << thresh[i] << " ";
  }
  if (rank == 0)
    std::cout << "\n";

  size_t compressed_size = 0;
  size_t lSize = 0;

  adios2::IO reader_io = ad.DeclareIO("Input");
  adios2::IO writer_io = ad.DeclareIO("Output");
  if (rank == 0) {
    std::cout << "write: "
              << "./" + fname + ".mgard"
              << "\n";
    std::cout << "readin: " << dpath + fname << "\n";
  }
  adios2::Engine reader = reader_io.Open(dpath + fname, adios2::Mode::Read);
  adios2::Engine writer =
      writer_io.Open("./" + fname + ".mgard", adios2::Mode::Write);
  float gb_maxv, gb_minv;
  float abs_tol;
  size_t dim_t, dim_c;
  adios2::Variable<float> var_ad2;
  var_ad2 = reader_io.InquireVariable<float>(var_name);
  std::vector<std::size_t> shape = var_ad2.Shape();
  adios2::Variable<float> var_out =
      writer_io.DefineVariable<float>(var_name, shape, {0, 0, 0}, shape);
  lSize += shape[0] * shape[1] * shape[2];
  float s = 0.0;
  if (ndims == 3) {
    float max_v = -1e8, min_v = 1e8;
    std::vector<float> var_in;
    if ((np_size == 1) ||
        ((np_size % 4 == 0) &&
         (((rank % 4 == 0) && (shape[2] == 240)) || (shape[2] == 960)))) {
      size_t n_t = (np_size > 4) ? (rank / 4) : 0;
      dim_t = (np_size > 4) ? (shape[0] / (np_size / 4)) : shape[0];
      size_t n_c = (shape[2] == 960) ? (rank % 4) : 0;
      dim_c = ((np_size > 1) && (shape[2] == 960)) ? (shape[2] / 4) : shape[2];
      std::cout << dim_t << ", " << dim_c << "\n";
      var_ad2.SetSelection(adios2::Box<adios2::Dims>(
          {n_t * dim_t, 0, n_c * dim_c}, {dim_t, shape[1], dim_c}));
      reader.Get<float>(var_ad2, var_in, adios2::Mode::Sync);
      reader.PerformGets();
      var_out.SetSelection(adios2::Box<adios2::Dims>(
          {n_t * dim_t, 0, n_c * dim_c}, {dim_t, shape[1], dim_c}));
      // convert tol to relative tol
      size_t msize = dim_t * shape[1] * dim_c;
      for (size_t k = 0; k < msize; k++) {
        max_v = std::max(max_v, var_in.at(k));
        min_v = std::min(min_v, var_in.at(k));
      }
    }
    MPI_Allreduce(&max_v, &gb_maxv, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&min_v, &gb_minv, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    abs_tol = (gb_maxv - gb_minv) * tol;

    if (rank == 0) {
      std::cout << "Variable " << var_name << ", shape = {" << shape[0] << ", "
                << shape[1] << ", " << shape[2] << "},";
      std::cout << "Get {" << dim_t << ", " << shape[1] << ", " << dim_c
                << "}\n";
      std::cout << "absolute tol: " << abs_tol << ", maxv: " << gb_maxv
                << ", min_v: " << gb_minv << "\n";
    }
    if ((np_size == 1) ||
        ((np_size % 4 == 0) &&
         (((rank % 4 == 0) && (shape[2] == 240)) || (shape[2] == 960)))) {
      const std::array<std::size_t, 3> dims = {dim_t, shape[1], dim_c};
      const mgard::TensorMeshHierarchy<3, float> hierarchy(dims);
      if (rank == 0)
        std::cout << "Lmax: " << hierarchy.L << "\n";
      const mgard::CompressedDataset<3, float> compressed =
          mgard::compress_roi(hierarchy, var_in.data(), s, abs_tol, thresh,
                              init_bw, bw_ratio, l_th, NULL, false);
      const mgard::DecompressedDataset<3, float> decompressed =
          mgard::decompress(compressed);
      writer.Put<float>(var_out, decompressed.data(), adios2::Mode::Sync);
      writer.PerformPuts();
      compressed_size += compressed.size();
    }
  } else if (ndims == 2) {
    size_t r_step = rank;
    const std::array<std::size_t, 2> dims = {shape[1], shape[2]};
    const mgard::TensorMeshHierarchy<2, float> hierarchy(dims);
    if (rank == 0)
      std::cout << "Lmax: " << hierarchy.L << "\n";
    while (r_step < shape[0]) {
      float max_v = -1e8, min_v = 1e8;
      std::vector<float> var_in;
      var_ad2.SetSelection(
          adios2::Box<adios2::Dims>({r_step, 0, 0}, {1, shape[1], shape[2]}));
      reader.Get<float>(var_ad2, var_in, adios2::Mode::Sync);
      reader.PerformGets();
      var_out.SetSelection(
          adios2::Box<adios2::Dims>({r_step, 0, 0}, {1, shape[1], shape[2]}));

      size_t msize = shape[1] * shape[2];
      for (size_t k = 0; k < msize; k++) {
        max_v = std::max(max_v, var_in.at(k));
        min_v = std::min(min_v, var_in.at(k));
      }
      MPI_Allreduce(&max_v, &gb_maxv, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&min_v, &gb_minv, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
      abs_tol = (gb_maxv - gb_minv) * tol;
      if (rank == 0) {
        std::cout << "Get " << var_name << ", shape = {" << shape[0] << ", "
                  << shape[1] << ", " << shape[2] << "}\n";
        std::cout << "absolute tol: " << abs_tol << ", maxv: " << gb_maxv
                  << ", min_v: " << gb_minv << "\n";
      }

      const mgard::CompressedDataset<2, float> compressed =
          mgard::compress_roi(hierarchy, var_in.data(), s, abs_tol, thresh,
                              init_bw, bw_ratio, l_th, NULL, false);
      const mgard::DecompressedDataset<2, float> decompressed =
          mgard::decompress(compressed);
      writer.Put<float>(var_out, decompressed.data(), adios2::Mode::Sync);
      writer.PerformPuts();
      compressed_size += compressed.size();
      r_step += np_size;
      if (rank == 0)
        std::cout << r_step << "\n";
    }
  }

  reader.Close();
  writer.Close();

  size_t gb_compressed;
  MPI_Allreduce(&compressed_size, &gb_compressed, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  if (rank == 0) {
    printf("%s: compression ratio = %.4f\n", var_name.c_str(),
           ((double)lSize * 4) / gb_compressed);
  }

  MPI_Finalize();
  return 0;
}
