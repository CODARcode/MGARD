/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#include <chrono>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compress.hpp"
// #include "compress_cuda.hpp"

using namespace std::chrono;

void print_usage_message(std::string error) {
  if (error.compare("") != 0) {
    std::cout << mgard_cuda::log::log_err << error << std::endl;
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
\t\t -l choose lossless compressor (0:ZSTD@CPU 1:Huffman@GPU 2:Huffman@GPU+LZ4@GPU)\n\
\t\t -v enable verbose (show timing and statistics)\n\
\n\
\t -x: decompress data\n\
\t\t -c <path to compressed file>\n\
\t\t -d <path to decompressed file>\n");
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

std::vector<mgard_cuda::SIZE> get_arg_dims(int argc, char *argv[],
                                           std::string option) {
  std::vector<mgard_cuda::SIZE> shape;
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
  std::cout << mgard_cuda::log::log_info << "Loading file: " << input_file
            << "\n";

  FILE *pFile;
  pFile = fopen(input_file, "rb");
  if (pFile == NULL) {
    std::cout << mgard_cuda::log::log_err << "file open error!\n";
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
std::vector<T *> readcoords(const char *input_file, mgard_cuda::DIM D,
                            std::vector<mgard_cuda::SIZE> shape) {
  std::cout << mgard_cuda::log::log_info
            << "Loading coordinate file: " << input_file << "\n";
  FILE *pFile;
  pFile = fopen(input_file, "rb");
  if (pFile == NULL) {
    std::cout << mgard_cuda::log::log_err << "coordinate file open error!\n";
    exit(1);
  }
  fseek(pFile, 0, SEEK_END);
  size_t lSize = ftell(pFile);
  size_t expected_size = 0;
  for (mgard_cuda::DIM d = 0; d < D; d++) {
    expected_size += sizeof(T) * shape[d];
  }
  if (lSize < expected_size) {
    std::cout << mgard_cuda::log::log_err << "coordinate file read error!\n";
    exit(-1);
  }
  rewind(pFile);
  std::vector<T *> coords(D);
  for (mgard_cuda::DIM d = 0; d < D; d++) {
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
void print_statistics(double s, enum mgard_cuda::error_bound_type mode,
                      size_t n, T *original_data, T *decompressed_data) {
  std::cout << std::scientific;
  if (s == std::numeric_limits<T>::infinity()) {
    if (mode == mgard_cuda::error_bound_type::ABS) {
      std::cout << mgard_cuda::log::log_info << "Absoluate L_inf error: "
                << mgard_cuda::L_inf_error(n, original_data, decompressed_data,
                                           mode)
                << "\n";
    } else if (mode == mgard_cuda::error_bound_type::REL) {
      std::cout << mgard_cuda::log::log_info << "Relative L_inf error: "
                << mgard_cuda::L_inf_error(n, original_data, decompressed_data,
                                           mode)
                << "\n";
    }
  } else {
    if (mode == mgard_cuda::error_bound_type::ABS) {
      std::cout << mgard_cuda::log::log_info << "Absoluate L_2 error: "
                << mgard_cuda::L_2_error(n, original_data, decompressed_data,
                                         mode)
                << "\n";
    } else if (mode == mgard_cuda::error_bound_type::REL) {
      std::cout << mgard_cuda::log::log_info << "Relative L_2 error: "
                << mgard_cuda::L_2_error(n, original_data, decompressed_data,
                                         mode)
                << "\n";
    }
  }
  // std::cout << mgard_cuda::log::log_info << "L_2 error: " <<
  // mgard_cuda::L_2_error(n, original_data, decompressed_data) << "\n";
  std::cout << mgard_cuda::log::log_info
            << "MSE: " << mgard_cuda::MSE(n, original_data, decompressed_data)
            << "\n";
  std::cout << std::defaultfloat;
  std::cout << mgard_cuda::log::log_info
            << "PSNR: " << mgard_cuda::PSNR(n, original_data, decompressed_data)
            << "\n";
}

template <typename T>
int launch_compress(mgard_cuda::DIM D, enum mgard_cuda::data_type dtype,
                    const char *input_file, const char *output_file,
                    std::vector<mgard_cuda::SIZE> shape, bool non_uniform,
                    const char *coords_file, double tol, double s,
                    enum mgard_cuda::error_bound_type mode, int lossless,
                    bool verbose) {

  mgard_cuda::Config config;
  config.timing = verbose;

  if (lossless == 0) {
    config.lossless = mgard_cuda::lossless_type::CPU_Lossless;
  } else if (lossless == 1) {
    config.lossless = mgard_cuda::lossless_type::GPU_Huffman;
  } else if (lossless == 2) {
    config.lossless = mgard_cuda::lossless_type::GPU_Huffman_LZ4;
  }

  size_t original_size = 1;
  for (mgard_cuda::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  T *original_data;
  size_t in_size = 0;
  if (std::string(input_file).compare("random") == 0) {
    in_size = original_size * sizeof(T);
    original_data = new T[original_size];
    for (size_t i = 0; i < original_size; i++)
      original_data[i] = rand() % 10 + 1;
  } else {
    in_size = readfile(input_file, original_data);
  }
  if (in_size != original_size * sizeof(T)) {
    std::cout << mgard_cuda::log::log_err << "input file size mismatch!\n";
  }

  void *compressed_data = NULL;
  size_t compressed_size = 0;
  void *decompressed_data = NULL;
  std::vector<const mgard_cuda::Byte *> coords_byte;
  if (!non_uniform) {
    mgard_cuda::compress(D, dtype, shape, tol, s, mode, original_data,
                         compressed_data, compressed_size, config);
  } else {
    std::vector<T *> coords;
    if (non_uniform) {
      coords = readcoords<T>(coords_file, D, shape);
    }
    for (auto &coord : coords) {
      coords_byte.push_back((const mgard_cuda::Byte *)coord);
    }
    mgard_cuda::compress(D, dtype, shape, tol, s, mode, original_data,
                         compressed_data, compressed_size, coords_byte, config);
  }

  writefile(output_file, compressed_size, compressed_data);

  printf("In size:  %10ld  Out size: %10ld  Compression ratio: %f \n",
         original_size * sizeof(T), compressed_size,
         (double)original_size * sizeof(T) / compressed_size);

  if (verbose) {
    config.timing = verbose;

    mgard_cuda::decompress(compressed_data, compressed_size, decompressed_data,
                           config);

    print_statistics<T>(s, mode, original_size, original_data,
                        (T *)decompressed_data);
  }

  delete[](T *) original_data;
  return 0;
}

int launch_decompress(const char *input_file, const char *output_file,
                      bool verbose) {

  mgard_cuda::Config config;
  config.timing = verbose;

  mgard_cuda::SERIALIZED_TYPE *compressed_data;
  size_t compressed_size = readfile(input_file, compressed_data);
  std::vector<mgard_cuda::SIZE> shape =
      mgard_cuda::infer_shape(compressed_data, compressed_size);
  mgard_cuda::data_type dtype =
      mgard_cuda::infer_data_type(compressed_data, compressed_size);

  size_t original_size = 1;
  for (mgard_cuda::DIM i = 0; i < shape.size(); i++) {
    original_size *= shape[i];
  }

  void *decompressed_data;

  mgard_cuda::decompress(compressed_data, compressed_size, decompressed_data,
                         config);

  int elem_size = 0;
  if (dtype == mgard_cuda::data_type::Double) {
    elem_size = 8;
  } else if (dtype == mgard_cuda::data_type::Float) {
    elem_size = 4;
  }
  writefile(output_file, original_size * elem_size, decompressed_data);

  delete[] compressed_data;
  return 0;
}

bool try_compression(int argc, char *argv[]) {
  if (!has_arg(argc, argv, "-z"))
    return false;
  std::cout << mgard_cuda::log::log_info << "mode: compression\n";
  std::string input_file = get_arg(argc, argv, "-i");
  std::string output_file = get_arg(argc, argv, "-c");

  std::cout << mgard_cuda::log::log_info << "original data: " << input_file
            << "\n";
  std::cout << mgard_cuda::log::log_info << "compressed data: " << output_file
            << "\n";

  enum mgard_cuda::data_type dtype;
  std::string dt = get_arg(argc, argv, "-t");
  if (dt.compare("s") == 0) {
    dtype = mgard_cuda::data_type::Float;
    std::cout << mgard_cuda::log::log_info << "data type: Single precision\n";
  } else if (dt.compare("d") == 0) {
    dtype = mgard_cuda::data_type::Double;
    std::cout << mgard_cuda::log::log_info << "data type: Double precision\n";
  } else
    print_usage_message("wrong data type.");

  mgard_cuda::DIM D = get_arg_int(argc, argv, "-n");
  std::vector<mgard_cuda::SIZE> shape = get_arg_dims(argc, argv, "-n");
  std::string shape_string = "shape (";
  for (mgard_cuda::DIM d = 0; d < shape.size(); d++)
    shape_string = shape_string + std::to_string(shape[d]) + " ";
  shape_string = shape_string + ")";

  bool non_uniform = false;
  std::string non_uniform_coords_file;
  if (has_arg(argc, argv, "-u")) {
    non_uniform = true;
    non_uniform_coords_file = get_arg(argc, argv, "-u");
    std::cout << mgard_cuda::log::log_info
              << "non-uniform coordinate file: " << non_uniform_coords_file
              << "\n";
  }

  enum mgard_cuda::error_bound_type mode; // REL or ABS
  std::string em = get_arg(argc, argv, "-m");
  if (em.compare("rel") == 0) {
    mode = mgard_cuda::error_bound_type::REL;
    std::cout << mgard_cuda::log::log_info << "error bound mode: Relative\n";
  } else if (em.compare("abs") == 0) {
    mode = mgard_cuda::error_bound_type::ABS;
    std::cout << mgard_cuda::log::log_info << "error bound mode: Absolute\n";
  } else
    print_usage_message("wrong error bound mode.");

  double tol = get_arg_double(argc, argv, "-e");
  double s = get_arg_double(argc, argv, "-s");

  std::cout << std::scientific;
  std::cout << mgard_cuda::log::log_info << "error bound: " << tol << "\n";
  std::cout << std::defaultfloat;
  std::cout << mgard_cuda::log::log_info << "s: " << s << "\n";

  int lossless_level = get_arg_int(argc, argv, "-l");
  if (lossless_level == 0) {
    std::cout << mgard_cuda::log::log_info << "lossless: ZSTD@CPU\n";
  } else if (lossless_level == 1) {
    std::cout << mgard_cuda::log::log_info << "lossless: Huffman@GPU\n";
  } else if (lossless_level == 2) {
    std::cout << mgard_cuda::log::log_info
              << "lossless: Huffman@GPU + LZ4@GPU\n";
  }
  bool verbose = has_arg(argc, argv, "-v");
  if (verbose)
    std::cout << mgard_cuda::log::log_info << "Verbose: enabled\n";
  if (dtype == mgard_cuda::data_type::Double) {
    launch_compress<double>(D, dtype, input_file.c_str(), output_file.c_str(),
                            shape, non_uniform, non_uniform_coords_file.c_str(),
                            tol, s, mode, lossless_level, verbose);
  } else if (dtype == mgard_cuda::data_type::Float) {
    launch_compress<float>(D, dtype, input_file.c_str(), output_file.c_str(),
                           shape, non_uniform, non_uniform_coords_file.c_str(),
                           tol, s, mode, lossless_level, verbose);
  }
  return true;
}

bool try_decompression(int argc, char *argv[]) {
  if (!has_arg(argc, argv, "-x"))
    return false;
  std::cout << mgard_cuda::log::log_info << "mode: decompress\n";
  std::string input_file = get_arg(argc, argv, "-c");
  std::string output_file = get_arg(argc, argv, "-d");
  std::cout << mgard_cuda::log::log_info << "compressed data: " << input_file
            << "\n";
  std::cout << mgard_cuda::log::log_info << "decompressed data: " << output_file
            << "\n";
  bool verbose = has_arg(argc, argv, "-v");
  if (verbose)
    std::cout << mgard_cuda::log::log_info << "verbose: enabled.\n";
  launch_decompress(input_file.c_str(), output_file.c_str(), verbose);
  return true;
}

int main(int argc, char *argv[]) {

  if (!try_compression(argc, argv) && !try_decompression(argc, argv)) {
    print_usage_message("");
  }
  return 0;
}
