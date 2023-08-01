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

#include "compress_x.hpp"
#include "mgard-x/Utilities/ErrorCalculator.h"
// #include "compress_cuda.hpp"

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
void print_statistics(double s, enum mgard_x::error_bound_type mode,
                      std::vector<mgard_x::SIZE> shape, T *original_data,
                      T *decompressed_data, T tol, bool normalize_coordinates) {
  mgard_x::SIZE n = 1;
  for (mgard_x::DIM d = 0; d < shape.size(); d++)
    n *= shape[d];
  T actual_error = 0.0;
  std::cout << std::scientific;
  if (s == std::numeric_limits<T>::infinity()) {
    actual_error =
        mgard_x::L_inf_error(n, original_data, decompressed_data, mode);
    if (mode == mgard_x::error_bound_type::ABS) {
      std::cout << mgard_x::log::log_info
                << "Absoluate L_inf error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    } else if (mode == mgard_x::error_bound_type::REL) {
      std::cout << mgard_x::log::log_info
                << "Relative L_inf error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    }
  } else {
    actual_error = mgard_x::L_2_error(shape, original_data, decompressed_data,
                                      mode, normalize_coordinates);
    if (mode == mgard_x::error_bound_type::ABS) {
      std::cout << mgard_x::log::log_info
                << "Absoluate L_2 error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    } else if (mode == mgard_x::error_bound_type::REL) {
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

int verbose_to_log_level(int verbose) {
  if (verbose == 0) {
    return mgard_x::log::ERR;
  } else if (verbose == 1) {
    return mgard_x::log::ERR | mgard_x::log::INFO;
  } else if (verbose == 2) {
    return mgard_x::log::ERR | mgard_x::log::TIME;
  } else if (verbose == 3) {
    return mgard_x::log::ERR | mgard_x::log::INFO | mgard_x::log::TIME;
  }
}

template <typename T>
int launch_compress(mgard_x::DIM D, enum mgard_x::data_type dtype,
                    const char *input_file, const char *output_file,
                    std::vector<mgard_x::SIZE> shape, bool non_uniform,
                    const char *coords_file, double tol, double s,
                    enum mgard_x::error_bound_type mode, int reorder,
                    int lossless, int domain_decomposition,
                    int hybrid_decomposition,
                    enum mgard_x::device_type dev_type, int verbose,
                    bool prefetch, mgard_x::SIZE max_memory_footprint) {

  mgard_x::Config config;
  config.log_level = verbose_to_log_level(verbose);
  if (hybrid_decomposition == 0) {
    config.decomposition = mgard_x::decomposition_type::MultiDim;
  } else {
    config.decomposition = mgard_x::decomposition_type::Hybrid;
    config.num_local_refactoring_level = 1;
    // config.max_larget_level = 0;
  }

  if (domain_decomposition == 0) {
    config.domain_decomposition = mgard_x::domain_decomposition_type::MaxDim;
  } else {
    config.domain_decomposition = mgard_x::domain_decomposition_type::Block;
  }
  config.dev_type = dev_type;
  config.reorder = reorder;
  config.prefetch = prefetch;
  config.max_memory_footprint = max_memory_footprint;
  config.huff_dict_size = 8192;
  config.adjust_shape = false;
  config.cache_compressor = true;

  if (lossless == 0) {
    config.lossless = mgard_x::lossless_type::Huffman;
  } else if (lossless == 1) {
    config.lossless = mgard_x::lossless_type::Huffman_LZ4;
  } else if (lossless == 2) {
    config.lossless = mgard_x::lossless_type::Huffman_Zstd;
  } else if (lossless == 3) {
    config.lossless = mgard_x::lossless_type::CPU_Lossless;
  }

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  T *original_data;
  size_t in_size = 0;
  if (std::string(input_file).compare("random") == 0) {
    in_size = original_size * sizeof(T);
    original_data = (T *)malloc(original_size * sizeof(T));
    srand(7117);
    T c = 0;
    for (size_t i = 0; i < original_size; i++) {
      original_data[i] = rand() % 10 + 1;
    }
  } else {
    in_size = readfile(input_file, original_data);
  }
  if (in_size != original_size * sizeof(T)) {
    std::cout << mgard_x::log::log_warn << "input file size mismatch "
              << in_size << " vs. " << original_size * sizeof(T) << "!\n";
  }

  size_t compressed_size = original_size * sizeof(T) * 2;
  void *compressed_data = (void *)malloc(compressed_size);
  mgard_x::pin_memory(original_data, original_size * sizeof(T), config);
  mgard_x::pin_memory(compressed_data, compressed_size, config);
  std::vector<const mgard_x::Byte *> coords_byte;
  mgard_x::compress_status_type ret;
  if (!non_uniform) {
    ret = mgard_x::compress(D, dtype, shape, tol, s, mode, original_data,
                            compressed_data, compressed_size, config, true);
  } else {
    std::vector<T *> coords;
    if (non_uniform) {
      coords = readcoords<T>(coords_file, D, shape);
    }
    for (auto &coord : coords) {
      coords_byte.push_back((const mgard_x::Byte *)coord);
    }
    ret = mgard_x::compress(D, dtype, shape, tol, s, mode, original_data,
                            compressed_data, compressed_size, coords_byte,
                            config, true);
  }

  if (ret != mgard_x::compress_status_type::Success) {
    std::cout << mgard_x::log::log_err << "Compression failed\n";
    exit(-1);
  }

  writefile(output_file, compressed_size, compressed_data);

  std::cout << mgard_x::log::log_info << "Compression ratio: "
            << (double)original_size * sizeof(T) / compressed_size << "\n";
  // printf("In size:  %10ld  Out size: %10ld  Compression ratio: %f \n",
  //        original_size * sizeof(T), compressed_size,
  //        (double)original_size * sizeof(T) / compressed_size);

  void *decompressed_data = malloc(original_size * sizeof(T));
  mgard_x::pin_memory(decompressed_data, original_size * sizeof(T), config);
  mgard_x::decompress(compressed_data, compressed_size, decompressed_data,
                      config, true);

  print_statistics<T>(s, mode, shape, original_data, (T *)decompressed_data,
                      tol, config.normalize_coordinates);

  mgard_x::unpin_memory(decompressed_data, config);
  free(decompressed_data);

  mgard_x::unpin_memory(original_data, config);
  mgard_x::unpin_memory(compressed_data, config);
  free(original_data);
  free(compressed_data);

  return 0;
}

int launch_decompress(const char *input_file, const char *output_file,
                      enum mgard_x::device_type dev_type, int verbose,
                      bool prefetch) {

  mgard_x::Config config;
  config.log_level = verbose_to_log_level(verbose);
  config.dev_type = dev_type;
  config.prefetch = prefetch;
  config.cache_compressor = true;

  mgard_x::SERIALIZED_TYPE *compressed_data;
  size_t compressed_size = readfile(input_file, compressed_data);
  std::vector<mgard_x::SIZE> shape;
  mgard_x::data_type dtype;
  void *decompressed_data;

  mgard_x::decompress(compressed_data, compressed_size, decompressed_data,
                      shape, dtype, config, false);

  int elem_size = 0;
  if (dtype == mgard_x::data_type::Double) {
    elem_size = 8;
  } else if (dtype == mgard_x::data_type::Float) {
    elem_size = 4;
  }

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < shape.size(); i++) {
    original_size *= shape[i];
  }

  writefile(output_file, original_size * elem_size, decompressed_data);

  delete[] compressed_data;
  return 0;
}

bool try_compression(int argc, char *argv[]) {
  if (!has_arg(argc, argv, "-z"))
    return false;
  std::cout << mgard_x::log::log_info << "mode: compression\n";
  std::string input_file = get_arg(argc, argv, "-i");
  std::string output_file = get_arg(argc, argv, "-c");

  std::cout << mgard_x::log::log_info << "original data: " << input_file
            << "\n";
  std::cout << mgard_x::log::log_info << "compressed data: " << output_file
            << "\n";

  enum mgard_x::data_type dtype;
  std::string dt = get_arg(argc, argv, "-t");
  if (dt.compare("s") == 0) {
    dtype = mgard_x::data_type::Float;
    std::cout << mgard_x::log::log_info << "data type: Single precision\n";
  } else if (dt.compare("d") == 0) {
    dtype = mgard_x::data_type::Double;
    std::cout << mgard_x::log::log_info << "data type: Double precision\n";
  } else
    print_usage_message("wrong data type.");

  mgard_x::DIM D = get_arg_int(argc, argv, "-n");
  std::vector<mgard_x::SIZE> shape = get_arg_dims(argc, argv, "-n");
  std::string shape_string = "shape (";
  for (mgard_x::DIM d = 0; d < shape.size(); d++)
    shape_string = shape_string + std::to_string(shape[d]) + " ";
  shape_string = shape_string + ")";

  bool non_uniform = false;
  std::string non_uniform_coords_file;
  if (has_arg(argc, argv, "-u")) {
    non_uniform = true;
    non_uniform_coords_file = get_arg(argc, argv, "-u");
    std::cout << mgard_x::log::log_info
              << "non-uniform coordinate file: " << non_uniform_coords_file
              << "\n";
  }

  enum mgard_x::error_bound_type mode; // REL or ABS
  std::string em = get_arg(argc, argv, "-m");
  if (em.compare("rel") == 0) {
    mode = mgard_x::error_bound_type::REL;
    std::cout << mgard_x::log::log_info << "error bound mode: Relative\n";
  } else if (em.compare("abs") == 0) {
    mode = mgard_x::error_bound_type::ABS;
    std::cout << mgard_x::log::log_info << "error bound mode: Absolute\n";
  } else
    print_usage_message("wrong error bound mode.");

  double tol = get_arg_double(argc, argv, "-e");
  double s = get_arg_double(argc, argv, "-s");

  std::cout << std::scientific;
  std::cout << mgard_x::log::log_info << "error bound: " << tol << "\n";
  std::cout << std::defaultfloat;
  std::cout << mgard_x::log::log_info << "s: " << s << "\n";

  int reorder = 0;
  if (has_arg(argc, argv, "-r")) {
    reorder = get_arg_int(argc, argv, "-r");
  }

  int lossless_level = get_arg_int(argc, argv, "-l");
  if (lossless_level == 0) {
    std::cout << mgard_x::log::log_info << "lossless: Huffman\n";
  } else if (lossless_level == 1) {
    std::cout << mgard_x::log::log_info << "lossless: Huffman + LZ4\n";
  } else if (lossless_level == 2) {
    std::cout << mgard_x::log::log_info << "lossless: Huffman + Zstd\n";
  }

  enum mgard_x::device_type dev_type;
  std::string dev = get_arg(argc, argv, "-d");
  if (dev.compare("auto") == 0) {
    dev_type = mgard_x::device_type::AUTO;
    std::cout << mgard_x::log::log_info << "device type: AUTO\n";
  } else if (dev.compare("serial") == 0) {
    dev_type = mgard_x::device_type::SERIAL;
    std::cout << mgard_x::log::log_info << "device type: SERIAL\n";
  } else if (dev.compare("openmp") == 0) {
    dev_type = mgard_x::device_type::OPENMP;
    std::cout << mgard_x::log::log_info << "device type: OPENMP\n";
  } else if (dev.compare("cuda") == 0) {
    dev_type = mgard_x::device_type::CUDA;
    std::cout << mgard_x::log::log_info << "device type: CUDA\n";
  } else if (dev.compare("hip") == 0) {
    dev_type = mgard_x::device_type::HIP;
    std::cout << mgard_x::log::log_info << "device type: HIP\n";
  } else if (dev.compare("sycl") == 0) {
    dev_type = mgard_x::device_type::SYCL;
    std::cout << mgard_x::log::log_info << "device type: SYCL\n";
  } else {
    print_usage_message("wrong device type.");
  }

  int verbose = 0;
  if (has_arg(argc, argv, "-v")) {
    verbose = get_arg_int(argc, argv, "-v");
  }

  int repeat = 1;
  if (has_arg(argc, argv, "-p")) {
    repeat = get_arg_int(argc, argv, "-p");
  }

  bool prefetch = true;
  if (has_arg(argc, argv, "-h")) {
    prefetch = get_arg_int(argc, argv, "-h") == 1 ? true : false;
  }

  mgard_x::SIZE max_memory_footprint =
      std::numeric_limits<mgard_x::SIZE>::max();
  if (has_arg(argc, argv, "-f")) {
    max_memory_footprint = (mgard_x::SIZE)get_arg_double(argc, argv, "-f");
  }

  int domain_decomposition = 0;
  if (has_arg(argc, argv, "-b")) {
    domain_decomposition = get_arg_int(argc, argv, "-b");
  }

  int hybrid_decomposition = 0;
  if (has_arg(argc, argv, "-y")) {
    hybrid_decomposition = get_arg_int(argc, argv, "-y");
  }

  if (verbose)
    std::cout << mgard_x::log::log_info << "Verbose: enabled\n";
  for (int repeat_iter = 0; repeat_iter < repeat; repeat_iter++) {
    if (dtype == mgard_x::data_type::Double) {
      launch_compress<double>(
          D, dtype, input_file.c_str(), output_file.c_str(), shape, non_uniform,
          non_uniform_coords_file.c_str(), tol, s, mode, reorder,
          lossless_level, domain_decomposition, hybrid_decomposition, dev_type,
          verbose, prefetch, max_memory_footprint);
    } else if (dtype == mgard_x::data_type::Float) {
      launch_compress<float>(
          D, dtype, input_file.c_str(), output_file.c_str(), shape, non_uniform,
          non_uniform_coords_file.c_str(), tol, s, mode, reorder,
          lossless_level, domain_decomposition, hybrid_decomposition, dev_type,
          verbose, prefetch, max_memory_footprint);
    }
  }
  mgard_x::release_cache(mgard_x::Config());
  return true;
}

bool try_decompression(int argc, char *argv[]) {
  if (!has_arg(argc, argv, "-x"))
    return false;
  std::cout << mgard_x::log::log_info << "mode: decompress\n";
  std::string input_file = get_arg(argc, argv, "-c");
  std::string output_file = get_arg(argc, argv, "-o");
  std::cout << mgard_x::log::log_info << "compressed data: " << input_file
            << "\n";
  std::cout << mgard_x::log::log_info << "decompressed data: " << output_file
            << "\n";

  enum mgard_x::device_type dev_type;
  std::string dev = get_arg(argc, argv, "-d");
  if (dev.compare("auto") == 0) {
    dev_type = mgard_x::device_type::AUTO;
    std::cout << mgard_x::log::log_info << "device type: AUTO\n";
  } else if (dev.compare("serial") == 0) {
    dev_type = mgard_x::device_type::SERIAL;
    std::cout << mgard_x::log::log_info << "device type: SERIAL\n";
  } else if (dev.compare("openmp") == 0) {
    dev_type = mgard_x::device_type::OPENMP;
    std::cout << mgard_x::log::log_info << "device type: OPENMP\n";
  } else if (dev.compare("cuda") == 0) {
    dev_type = mgard_x::device_type::CUDA;
    std::cout << mgard_x::log::log_info << "device type: CUDA\n";
  } else if (dev.compare("hip") == 0) {
    dev_type = mgard_x::device_type::HIP;
    std::cout << mgard_x::log::log_info << "device type: HIP\n";
  } else if (dev.compare("sycl") == 0) {
    dev_type = mgard_x::device_type::HIP;
    std::cout << mgard_x::log::log_info << "device type: SYCL\n";
  } else {
    print_usage_message("wrong device type.");
  }

  int verbose = 0;
  if (has_arg(argc, argv, "-v")) {
    verbose = get_arg_int(argc, argv, "-v");
  }

  int repeat = 1;
  if (has_arg(argc, argv, "-p")) {
    repeat = get_arg_int(argc, argv, "-p");
  }

  bool prefetch = true;
  if (has_arg(argc, argv, "-h")) {
    prefetch = get_arg_int(argc, argv, "-h") == 1 ? true : false;
  }

  if (verbose)
    std::cout << mgard_x::log::log_info << "verbose: enabled.\n";
  for (int repeat_iter = 0; repeat_iter < repeat; repeat_iter++) {
    launch_decompress(input_file.c_str(), output_file.c_str(), dev_type,
                      verbose, prefetch);
  }
  mgard_x::release_cache(mgard_x::Config());
  return true;
}

int main(int argc, char *argv[]) {

  if (!try_compression(argc, argv) && !try_decompression(argc, argv)) {
    print_usage_message("");
  }
  return 0;
}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 