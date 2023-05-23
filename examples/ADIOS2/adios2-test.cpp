/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include <chrono>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <adios2.h>
#include <mpi.h>

#include "mgard/compress_x.hpp"
#include "mgard/mgard-x/Utilities/ErrorCalculator.h"

#define OUTPUT_SAFTY_OVERHEAD 1e6

using namespace std::chrono;

void print_usage_message(std::string error) {
  if (error.compare("") != 0) {
    std::cout << mgard_x::log::log_err << error << std::endl;
  }
  printf("Options\n\
\t -z: compress data\n\
\t\t -i <path to data file to be compressed>\n\
\t\t -c <path to compressed file>\n\
\t\t -t <s|d>: data type (s: single; d:double)\n\
\t\t -n <ndim>: total number of dimensions\n\
\t\t\t [dim1]: slowest dimention\n\
\t\t\t [dim2]: 2nd slowest dimention\n\
\t\t\t  ...\n\
\t\t\t [dimN]: fastest dimention\n\
\t\t -u <path to coordinate file>\n\
\t\t -m <abs|rel>: error bound mode (abs: abolute; rel: relative)\n\
\t\t -e <error>: error bound\n\
\t\t -s <smoothness>: smoothness parameter\n\
\t\t -l choose lossless compressor (0:Huffman 1:Huffman+LZ4 2:Huffman+Zstd)\n\
\t\t -d <auto|serial|openmp|cuda|hip|sycl>: device type\n\
\t\t -v enable verbose (show timing and statistics)\n\
\n\
\t -x: decompress data\n\
\t\t -c <path to compressed file>\n\
\t\t -o <path to decompressed file>\n\
\t\t -d <auto|serial|cuda|hip>: device type\n\
\t\t -v enable verbose (show timing and statistics)\n");
  exit(0);
}

bool has_arg(int argc, char *argv[], std::string option) {
  for (int i = 0; i < argc; i++) {
    if (option.compare(std::string(argv[i])) == 0) {
      return true;
    }
  }
  return false;
}

bool require_arg(int argc, char *argv[], std::string option) {
  for (int i = 0; i < argc; i++) {
    if (option.compare(std::string(argv[i])) == 0) {
      return true;
    }
  }
  print_usage_message("missing option: " + option + ".");
  return false;
}

std::string get_arg(int argc, char *argv[], std::string option) {
  if (require_arg(argc, argv, option)) {
    for (int i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        return std::string(argv[i + 1]);
      }
    }
  }
  return std::string("");
}

int get_arg_int(int argc, char *argv[], std::string option) {
  if (require_arg(argc, argv, option)) {
    std::string arg;
    int i;
    for (i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        arg = std::string(argv[i + 1]);
      }
    }
    try {
      int d = std::stoi(arg);
      return d;
    } catch (std::invalid_argument const &e) {
      print_usage_message("illegal argument for option " + option + ".");
      return 0;
    }
  }
  return 0;
}

std::vector<mgard_x::SIZE> get_arg_dims(int argc, char *argv[],
                                        std::string option) {
  std::vector<mgard_x::SIZE> shape;
  if (require_arg(argc, argv, option)) {
    std::string arg;
    int arg_idx = 0, i;
    for (i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        arg = std::string(argv[i + 1]);
        arg_idx = i + 1;
      }
    }
    try {
      int d = std::stoi(arg);
      for (int i = 0; i < d; i++) {
        shape.push_back(std::stoi(argv[arg_idx + 1 + i]));
      }
      return shape;
    } catch (std::invalid_argument const &e) {
      print_usage_message("illegal argument for option " + option + ".");
      return shape;
    }
  }
  return shape;
}

double get_arg_double(int argc, char *argv[], std::string option) {
  if (require_arg(argc, argv, option)) {
    std::string arg;
    int i;
    for (i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        arg = std::string(argv[i + 1]);
      }
    }
    try {
      double d = std::stod(arg);
      return d;
    } catch (std::invalid_argument const &e) {
      print_usage_message("illegal argument for option " + option + ".");
    }
  }
  return 0;
}

template <typename T> void min_max(size_t n, T *in_buff) {
  T min = std::numeric_limits<T>::infinity();
  T max = 0;
  for (size_t i = 0; i < n; i++) {
    if (min > in_buff[i]) {
      min = in_buff[i];
    }
    if (max < in_buff[i]) {
      max = in_buff[i];
    }
  }
  printf("Min: %f, Max: %f\n", min, max);
}

template <typename T> size_t readfile(const char *input_file, T *&in_buff) {
  std::cout << mgard_x::log::log_info << "Loading file: " << input_file << "\n";

  FILE *pFile;
  pFile = fopen(input_file, "rb");
  if (pFile == NULL) {
    std::cout << mgard_x::log::log_err << "file open error!\n";
    exit(1);
  }
  fseek(pFile, 0, SEEK_END);
  size_t lSize = ftell(pFile);
  rewind(pFile);
  in_buff = (T *)malloc(lSize);
  lSize = fread(in_buff, 1, lSize, pFile);
  fclose(pFile);
  // min_max(lSize/sizeof(T), in_buff);
  return lSize;
}

template <typename T>
std::vector<T *> readcoords(const char *input_file, mgard_x::DIM D,
                            std::vector<mgard_x::SIZE> shape) {
  std::cout << mgard_x::log::log_info
            << "Loading coordinate file: " << input_file << "\n";
  FILE *pFile;
  pFile = fopen(input_file, "rb");
  if (pFile == NULL) {
    std::cout << mgard_x::log::log_err << "coordinate file open error!\n";
    exit(1);
  }
  fseek(pFile, 0, SEEK_END);
  size_t lSize = ftell(pFile);
  size_t expected_size = 0;
  for (mgard_x::DIM d = 0; d < D; d++) {
    expected_size += sizeof(T) * shape[d];
  }
  if (lSize < expected_size) {
    std::cout << mgard_x::log::log_err << "coordinate file read error!\n";
    exit(-1);
  }
  rewind(pFile);
  std::vector<T *> coords(D);
  for (mgard_x::DIM d = 0; d < D; d++) {
    coords[d] = (T *)malloc(shape[d]);
    lSize = fread(coords[d], sizeof(T), shape[d], pFile);
  }
  fclose(pFile);
  return coords;
}

template <typename T>
void writefile(const char *output_file, size_t num_bytes, T *out_buff) {
  FILE *file = fopen(output_file, "w");
  fwrite(out_buff, 1, num_bytes, file);
  fclose(file);
}

template <typename T>
void print_statistics(double s, std::string eb_mode,
                      std::vector<mgard_x::SIZE> shape, T *original_data,
                      T *decompressed_data, double tol,
                      bool normalize_coordinates) {
  mgard_x::SIZE n = 1;
  for (mgard_x::DIM d = 0; d < shape.size(); d++)
    n *= shape[d];
  T actual_error = 0.0;
  std::cout << std::scientific;
  if (s == std::numeric_limits<T>::infinity()) {
    if (eb_mode.compare("abs") == 0) {
      actual_error = mgard_x::L_inf_error(n, original_data, decompressed_data,
                                          mgard_x::error_bound_type::ABS);
      std::cout << mgard_x::log::log_info
                << "Absoluate L_inf error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    } else if (eb_mode.compare("rel") == 0) {
      actual_error = mgard_x::L_inf_error(n, original_data, decompressed_data,
                                          mgard_x::error_bound_type::REL);
      std::cout << mgard_x::log::log_info
                << "Relative L_inf error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    }
  } else {
    actual_error = mgard_x::L_2_error(shape, original_data, decompressed_data,
                                      mgard_x::error_bound_type::ABS,
                                      normalize_coordinates);
    if (eb_mode.compare("abs") == 0) {
      std::cout << mgard_x::log::log_info
                << "Absoluate L_2 error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    } else if (eb_mode.compare("rel") == 0) {
      actual_error = mgard_x::L_2_error(shape, original_data, decompressed_data,
                                        mgard_x::error_bound_type::REL,
                                        normalize_coordinates);
      std::cout << mgard_x::log::log_info
                << "Relative L_2 error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    }
  }

  std::cout << mgard_x::log::log_info
            << "MSE: " << mgard_x::MSE(n, original_data, decompressed_data)
            << "\n";
  std::cout << std::defaultfloat;
  std::cout << mgard_x::log::log_info
            << "PSNR: " << mgard_x::PSNR(n, original_data, decompressed_data)
            << "\n";

  if (actual_error > tol)
    exit(-1);
}

void print_shape(adios2::Dims shape) {
  std::cout << "(";
  for (int i = 0; i < shape.size(); i++) {
    std::cout << shape[i];
    if (i < shape.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << ")";
}

template <typename T>
int launch_compress(mgard_x::DIM D, enum mgard_x::data_type dtype,
                    std::string input_file, std::string output_file,
                    std::string var_name, int step_start, int step_end,
                    std::vector<size_t> shape, double tol, double s,
                    std::string eb_mode) {

  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  adios2::ADIOS adios(MPI_COMM_WORLD);
  adios2::IO read_io = adios.DeclareIO("Input Data");
  adios2::IO write_io = adios.DeclareIO("Output Data");
  read_io.SetEngine("BP5");
  write_io.SetEngine("BP4");
  adios2::Engine reader = read_io.Open(input_file, adios2::Mode::Read);
  adios2::Engine writer = write_io.Open(output_file, adios2::Mode::Write);

  adios2::Variable<T> org_var;
  adios2::Variable<T> cmp_var;

  adios2::Dims var_shape = shape;
  int largest_idx = 0;
  size_t largest_dim = var_shape[largest_idx];
  for (int i = 1; i < var_shape.size(); i++) {
    if (var_shape[i] > largest_dim) {
      largest_idx = i;
      largest_dim = var_shape[i];
    }
  }
  size_t block_size = var_shape[largest_idx] / comm_size;
  size_t leftover_size = var_shape[largest_idx] - block_size * rank;
  size_t var_size = std::min(block_size, leftover_size);

  adios2::Dims var_count_local = var_shape;
  var_count_local[largest_idx] = var_size;
  adios2::Dims var_start_local(var_shape.size(), 0);
  var_start_local[largest_idx] = block_size * rank;

  bool first = true;

  for (int sim_iter = 0; sim_iter <= step_end; sim_iter++) {
    std::vector<T> var_data_vec;
    reader.BeginStep();
    writer.BeginStep();

    // Input variable
    org_var = read_io.InquireVariable<T>(var_name);
    adios2::Box<adios2::Dims> sel(var_start_local, var_count_local);
    org_var.SetSelection(sel);
    std::cout << "rank " << rank << ": ";
    print_shape(var_start_local);
    print_shape(var_count_local);
    std::cout << "\n";

    if (first) {
      // Output variable
      cmp_var = write_io.DefineVariable<T>(var_name, var_shape, var_start_local,
                                           var_count_local);
      string eb_mode_adios;
      if (eb_mode.compare("abs") == 0)
        eb_mode_adios = "ABS";
      else
        eb_mode_adios = "REL";

      cmp_var.AddOperation("mgard", {{"tolerance", std::to_string(tol)},
                                     {"mode", eb_mode_adios},
                                     {"s", std::to_string(s)}});
      first = false;
    }
    if (sim_iter >= step_start) {
      reader.Get<T>(org_var, var_data_vec, adios2::Mode::Sync);
      writer.Put<T>(cmp_var, var_data_vec.data(), adios2::Mode::Sync);

      // std::fstream myfile;
      // myfile.open("wrf"+std::to_string(sim_iter) + ".dat", std::ios::out |
      // std::ios::binary); if (!myfile) {
      //   printf("Error: cannot open file\n");
      // }
      // myfile.write((char *)var_data_vec.data(), 1200*1500 * sizeof(T));
      // myfile.close();
    }

    reader.EndStep();
    writer.EndStep();
  }
  reader.Close();
  writer.Close();

  return 0;
}

template <typename T>
int launch_decompress(mgard_x::DIM D, enum mgard_x::data_type dtype,
                      std::string org_file, std::string cmp_file,
                      std::string dec_file, std::string var_name,
                      int step_start, int step_end, std::vector<size_t> shape,
                      double tol, double s, std::string eb_mode) {

  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  adios2::ADIOS adios(MPI_COMM_WORLD);
  adios2::IO org_io = adios.DeclareIO("Original Data");
  adios2::IO cmp_io = adios.DeclareIO("Compressed Data");
  adios2::IO dec_io = adios.DeclareIO("Decompressed Data");
  org_io.SetEngine("BP5");
  cmp_io.SetEngine("BP4");
  dec_io.SetEngine("BP4");
  adios2::Engine org_reader = org_io.Open(org_file, adios2::Mode::Read);
  adios2::Engine cmp_reader = cmp_io.Open(cmp_file, adios2::Mode::Read);
  adios2::Engine dec_writer = dec_io.Open(dec_file, adios2::Mode::Write);

  adios2::Variable<T> org_var;
  adios2::Variable<T> cmp_var;
  adios2::Variable<T> dec_var;

  adios2::Dims var_shape = shape;
  int largest_idx = 0;
  size_t largest_dim = var_shape[largest_idx];
  for (int i = 1; i < var_shape.size(); i++) {
    if (var_shape[i] > largest_dim) {
      largest_idx = i;
      largest_dim = var_shape[i];
    }
  }
  size_t block_size = var_shape[largest_idx] / comm_size;
  size_t leftover_size = var_shape[largest_idx] - block_size * rank;
  size_t var_size = std::min(block_size, leftover_size);

  adios2::Dims var_count_local = var_shape;
  var_count_local[largest_idx] = var_size;
  adios2::Dims var_start_local(var_shape.size(), 0);
  var_start_local[largest_idx] = block_size * rank;

  bool first = true;

  for (int sim_iter = 0; sim_iter <= step_end; sim_iter++) {
    std::vector<T> org_vec, dec_vec;
    org_reader.BeginStep();
    cmp_reader.BeginStep();
    dec_writer.BeginStep();

    // Original variable
    org_var = org_io.InquireVariable<T>(var_name);
    adios2::Box<adios2::Dims> sel(var_start_local, var_count_local);
    org_var.SetSelection(sel);
    std::cout << "rank " << rank << ": ";
    print_shape(var_start_local);
    print_shape(var_count_local);
    std::cout << "\n";

    // Ccompressed variable
    cmp_var = cmp_io.InquireVariable<T>(var_name);
    cmp_var.SetSelection(sel);

    if (first) {
      // Output variable
      dec_var = dec_io.DefineVariable<T>(var_name, var_shape, var_start_local,
                                         var_count_local);
      first = false;
    }

    if (sim_iter >= step_start) {
      org_reader.Get<T>(org_var, org_vec, adios2::Mode::Sync);
      cmp_reader.Get<T>(cmp_var, dec_vec, adios2::Mode::Sync);
      dec_writer.Put<T>(dec_var, dec_vec.data(), adios2::Mode::Sync);
    }

    org_reader.EndStep();
    cmp_reader.EndStep();
    dec_writer.EndStep();
    print_statistics(s, eb_mode, var_count_local, org_vec.data(),
                     dec_vec.data(), tol, true);
  }
  org_reader.Close();
  cmp_reader.Close();
  dec_writer.Close();
  return 0;
}

bool try_exec(int argc, char *argv[]) {
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int compress_or_decompress;
  if (has_arg(argc, argv, "-z")) {
    compress_or_decompress = 0;
    if (!rank)
      std::cout << mgard_x::log::log_info << "mode: compression\n";
  } else if (has_arg(argc, argv, "-x")) {
    compress_or_decompress = 1;
    if (!rank)
      std::cout << mgard_x::log::log_info << "mode: decompress\n";
  } else {
    return false;
  }

  std::string input_file = get_arg(argc, argv, "-i");
  std::string output_file = get_arg(argc, argv, "-c");
  std::string decompressed_file = get_arg(argc, argv, "-o");

  if (!rank)
    std::cout << mgard_x::log::log_info << "input data: " << input_file << "\n";
  if (!rank)
    std::cout << mgard_x::log::log_info << "output data: " << output_file
              << "\n";
  if (!rank)
    std::cout << mgard_x::log::log_info
              << "decompressed data: " << decompressed_file << "\n";

  enum mgard_x::data_type dtype;
  std::string dt = get_arg(argc, argv, "-t");
  if (dt.compare("s") == 0) {
    dtype = mgard_x::data_type::Float;
    if (!rank)
      std::cout << mgard_x::log::log_info << "data type: Single precision\n";
  } else if (dt.compare("d") == 0) {
    dtype = mgard_x::data_type::Double;
    if (!rank)
      std::cout << mgard_x::log::log_info << "data type: Double precision\n";
  } else {
    if (!rank)
      print_usage_message("wrong data type.");
  }

  std::string var_name = get_arg(argc, argv, "-v");
  if (!rank)
    std::cout << mgard_x::log::log_info << "Variable name: " << var_name
              << "\n";

  mgard_x::DIM D = get_arg_int(argc, argv, "-n");
  std::vector<mgard_x::SIZE> shape = get_arg_dims(argc, argv, "-n");
  if (!rank) {
    std::string shape_string = "shape (";
    for (mgard_x::DIM d = 0; d < shape.size(); d++)
      shape_string = shape_string + std::to_string(shape[d]) + " ";
    shape_string = shape_string + ")";
  }

  std::string eb_mode = get_arg(argc, argv, "-m");
  if (eb_mode.compare("rel") == 0) {
    if (!rank)
      std::cout << mgard_x::log::log_info << "error bound mode: Relative\n";
  } else if (eb_mode.compare("abs") == 0) {
    if (!rank)
      std::cout << mgard_x::log::log_info << "error bound mode: Absolute\n";
  } else {
    if (!rank)
      print_usage_message("wrong error bound mode.");
  }

  double tol = get_arg_double(argc, argv, "-e");
  double s = get_arg_double(argc, argv, "-s");

  std::cout << std::scientific;
  if (!rank)
    std::cout << mgard_x::log::log_info << "error bound: " << tol << "\n";
  std::cout << std::defaultfloat;
  if (!rank)
    std::cout << mgard_x::log::log_info << "s: " << s << "\n";

  int step_start = get_arg_int(argc, argv, "-b");
  int step_end = get_arg_int(argc, argv, "-d");

  if (!rank) {
    std::cout << mgard_x::log::log_info << "step: " << step_start << " - "
              << step_end << "\n";
  }

  if (compress_or_decompress == 0) {
    if (dtype == mgard_x::data_type::Double) {
      launch_compress<double>(D, dtype, input_file, output_file, var_name,
                              step_start, step_end, shape, tol, s, eb_mode);
    } else if (dtype == mgard_x::data_type::Float) {
      launch_compress<float>(D, dtype, input_file, output_file, var_name,
                             step_start, step_end, shape, tol, s, eb_mode);
    }
  } else {
    if (dtype == mgard_x::data_type::Double) {
      launch_decompress<double>(D, dtype, input_file, output_file,
                                decompressed_file, var_name, step_start,
                                step_end, shape, tol, s, eb_mode);
    } else if (dtype == mgard_x::data_type::Float) {
      launch_decompress<float>(D, dtype, input_file, output_file,
                               decompressed_file, var_name, step_start,
                               step_end, shape, tol, s, eb_mode);
    }
  }
  return true;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  if (!try_exec(argc, argv)) {
    print_usage_message("");
  }
  MPI_Finalize();
  return 0;
}
