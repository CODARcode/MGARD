/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include <chrono>
#include <cstring>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "compress_x.hpp"
#include "mgard-x/Utilities/ErrorCalculator.h"

#include "ArgumentParser.h"

#define OUTPUT_SAFTY_OVERHEAD 1e6

using namespace std::chrono;

void print_usage_message(std::string error) {
  if (error.compare("") != 0) {
    std::cout << mgard_x::log::log_err << error << std::endl;
  }
  printf("Options\n\
\t -z / --compress: compress mode\n\
\t\t -i / --input <path to original data>\n\
\t\t -o / --output <path to compressed data>\n\
\t\t -dt / --data-type <s/single|d/double>: data type (s: single; d:double)\n\
\t\t -dim / --dimension <int>: total number of dimensions\n\
\t\t\t [int]: slowest dimention\n\
\t\t\t [int]: 2nd slowest dimention\n\
\t\t\t  ...\n\
\t\t\t [int]: fastest dimention\n\
\t\t -em / --error-bound-mode <abs|rel>: error bound mode (abs: abolute; rel: relative)\n\
\t\t -e / --error-bound <float>: error bound\n\
\t\t -s / --smoothness <float>: smoothness parameter\n\
\t\t -l / --lossless <huffman|huffman-lz4|huffman-zstd>: lossless compression\n\
\t\t -d / --device <auto|serial|cuda|hip>: device type\n\
\t\t (optional) -v / --verbose <0|1|2|3> 0: error; 1: error+info; 2: error+timing; 3: all\n\
\n\
\t -x / --decompress: decompress mode\n\
\t\t -i / --input <path to compressed data>\n\
\t\t -o / --output <path to decompressed data>\n\
\t\t -d / --device <auto|serial|cuda|hip>: device type\n\
\t\t (optional) -v / --verbose <0|1|2|3> 0: error; 1: error+info; 2: error+timing; 3: all\n");
  exit(0);
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

  // if (actual_error > tol)
  // exit(-1);
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
  } else if (verbose == 4) {
    return mgard_x::log::ERR | mgard_x::log::INFO | mgard_x::log::TIME |
           mgard_x::log::DBG;
  }
}

template <typename T>
int launch_compress(mgard_x::DIM D, enum mgard_x::data_type dtype,
                    const char *input_file, const char *output_file,
                    std::vector<mgard_x::SIZE> shape, double tol, double s,
                    enum mgard_x::error_bound_type mode, std::string lossless,
                    std::string domain_decomposition, mgard_x::SIZE block_size,
                    enum mgard_x::device_type dev_type, int verbose,
                    mgard_x::SIZE max_memory_footprint) {

  mgard_x::Config config;
  config.log_level = verbose_to_log_level(verbose);
  config.decomposition = mgard_x::decomposition_type::MultiDim;
  // config.decomposition = mgard_x::decomposition_type::Hybrid;
  // config.num_local_refactoring_level = 1;

  // config.max_larget_level = 1;

  // config.compressor = mgard_x::compressor_type::ZFP;

  if (domain_decomposition == "block") {
    config.domain_decomposition = mgard_x::domain_decomposition_type::Block;
    config.block_size = block_size;
  } else {
    config.domain_decomposition = mgard_x::domain_decomposition_type::MaxDim;
  }

  config.cpu_mode = mgard_x::cpu_parallelization_mode::INTRA_BLOCK;

  // config.domain_decomposition = mgard_x::domain_decomposition_type::Variable;
  config.domain_decomposition_dim = 0;
  // NYX
  // config.domain_d  ecomposition_sizes = {512, 512};

  // config.domain_decomposition_sizes = {512, 512, 512, 512};
  // config.domain_decomposition_sizes = {2048};
  // config.domain_decomposition_sizes = {128, 248, 315, 348, 384, 424, 201};
  // config.domain_decomposition_sizes = std::vector<mgard_x::SIZE>(128, 16);
  // config.domain_decomposition_sizes = std::vector<mgard_x::SIZE>(128, 16);

  // XGC
  // config.domain_decomposition_sizes = {312, 312, 312, 312};
  // config.domain_decomposition_sizes = {1248};
  // config.domain_decomposition_sizes = {156, 283, 514, 295};
  // config.domain_decomposition_sizes = std::vector<mgard_x::SIZE>(96, 13);

  // E3SM
  // config.domain_decomposition_sizes = {720, 720, 720, 720};
  // config.domain_decomposition_sizes = {180, 368, 463, 529, 605, 692, 43};
  // config.domain_decomposition_sizes = std::vector<mgard_x::SIZE>(192, 15);

  config.estimate_outlier_ratio = 0.3;

  config.dev_type = dev_type;
  config.reorder = 0;
  config.auto_pin_host_buffers = true;
  config.max_memory_footprint = max_memory_footprint;
  config.huff_dict_size = 8192;
  config.adjust_shape = false;
  config.auto_cache_release = false;

  if (lossless == "huffman") {
    config.lossless = mgard_x::lossless_type::Huffman;
  } else if (lossless == "huffman-lz4") {
    config.lossless = mgard_x::lossless_type::Huffman_LZ4;
  } else if (lossless == "huffman-zstd") {
    config.lossless = mgard_x::lossless_type::Huffman_Zstd;
  }

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  T *original_data = (T *)malloc(original_size * sizeof(T));
  size_t in_size = 0;
  if (std::string(input_file).compare("random") == 0) {
    in_size = original_size * sizeof(T);
    srand(7117);
    T c = 0;
    for (size_t i = 0; i < original_size; i++) {
      original_data[i] = rand() % 10 + 1;
    }
  } else {
    T *file_data;
    in_size = readfile(input_file, file_data);

    size_t loaded_size = 0;
    while (loaded_size < original_size) {
      // std::cout << "copy input\n";
      std::memcpy(original_data + loaded_size, file_data,
                  std::min(in_size / sizeof(T), original_size - loaded_size) *
                      sizeof(T));
      loaded_size += std::min(in_size / sizeof(T), original_size - loaded_size);
    }
    in_size = loaded_size * sizeof(T);
  }
  if (in_size != original_size * sizeof(T)) {
    std::cout << mgard_x::log::log_warn << "input file size mismatch "
              << in_size << " vs. " << original_size * sizeof(T) << "!\n";
  }

  size_t compressed_size = original_size * sizeof(T) * 2;
  void *compressed_data = (void *)malloc(compressed_size);
  mgard_x::pin_memory(original_data, original_size * sizeof(T), config);
  mgard_x::pin_memory(compressed_data, compressed_size, config);
  mgard_x::compress_status_type ret;
  ret = mgard_x::compress(D, dtype, shape, tol, s, mode, original_data,
                          compressed_data, compressed_size, config, true);
  if (ret != mgard_x::compress_status_type::Success) {
    std::cout << mgard_x::log::log_err << "Compression failed\n";
    exit(-1);
  }
  writefile(output_file, compressed_size, compressed_data);
  std::cout << mgard_x::log::log_info << "Compression ratio: "
            << (double)original_size * sizeof(T) / compressed_size << "\n";

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
                      enum mgard_x::device_type dev_type, int verbose) {
  mgard_x::Config config;
  config.log_level = verbose_to_log_level(verbose);
  config.dev_type = dev_type;
  config.auto_pin_host_buffers = true;
  config.auto_cache_release = true;

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
  if (!has_arg(argc, argv, "-z", "--compress"))
    return false;
  mgard_x::log::info("mode: compress", true);
  std::string input_file =
      get_arg<std::string>(argc, argv, "Original data", "-i", "--input");
  std::string output_file =
      get_arg<std::string>(argc, argv, "Compressed data", "-o", "--output");
  enum mgard_x::data_type dtype = get_data_type(argc, argv);
  std::vector<mgard_x::SIZE> shape =
      get_args<mgard_x::SIZE>(argc, argv, "Dimensions", "-dim", "--dimension");
  enum mgard_x::error_bound_type mode =
      get_error_bound_mode(argc, argv); // REL or ABS
  double tol =
      get_arg<double>(argc, argv, "Error bound", "-e", "--error-bound");
  double s = get_arg<double>(argc, argv, "Smoothness", "-s", "--smoothness");
  std::string lossless =
      get_arg<std::string>(argc, argv, "Lossless", "-l", "--lossless");
  enum mgard_x::device_type dev_type = get_device_type(argc, argv);
  int verbose = 0;
  if (has_arg(argc, argv, "-v", "--verbose")) {
    verbose = get_arg<int>(argc, argv, "Verbose", "-v", "--verbose");
  }
  mgard_x::SIZE max_memory_footprint =
      std::numeric_limits<mgard_x::SIZE>::max();
  if (has_arg(argc, argv, "-m", "--max-memory")) {
    max_memory_footprint = (mgard_x::SIZE)get_arg<double>(
        argc, argv, "Max memory", "-m", "--max-memory");
  }
  std::string domain_decomposition = "max-dim";
  mgard_x::SIZE block_size = 0;
  if (has_arg(argc, argv, "-dd", "--domain-decomposition")) {
    domain_decomposition = get_arg<std::string>(
        argc, argv, "Domain decomposition", "-dd", "--domain-decomposition");
    if (domain_decomposition == "block") {
      block_size = get_arg<mgard_x::SIZE>(argc, argv, "Block size", "-dd-size",
                                          "--domain-decomposition-size");
    }
  }

  if (dtype == mgard_x::data_type::Double) {
    launch_compress<double>(shape.size(), dtype, input_file.c_str(),
                            output_file.c_str(), shape, tol, s, mode, lossless,
                            domain_decomposition, block_size, dev_type, verbose,
                            max_memory_footprint);
  } else if (dtype == mgard_x::data_type::Float) {
    launch_compress<float>(shape.size(), dtype, input_file.c_str(),
                           output_file.c_str(), shape, tol, s, mode, lossless,
                           domain_decomposition, block_size, dev_type, verbose,
                           max_memory_footprint);
  }
  mgard_x::release_cache(mgard_x::Config());
  return true;
}

bool try_decompression(int argc, char *argv[]) {
  if (!has_arg(argc, argv, "-x", "--decompress"))
    return false;
  mgard_x::log::info("mode: decompress", true);
  std::string input_file =
      get_arg<std::string>(argc, argv, "Compressed data", "-i", "--input");
  std::string output_file =
      get_arg<std::string>(argc, argv, "Decompressed data", "-o", "--output");
  enum mgard_x::device_type dev_type = get_device_type(argc, argv);
  int verbose = 0;
  if (has_arg(argc, argv, "-v", "--verbose")) {
    verbose = get_arg<int>(argc, argv, "Verbose", "-v", "--verbose");
  }
  launch_decompress(input_file.c_str(), output_file.c_str(), dev_type, verbose);
  mgard_x::release_cache(mgard_x::Config());
  return true;
}

int main(int argc, char *argv[]) {

  if (!try_compression(argc, argv) && !try_decompression(argc, argv)) {
    print_usage_message("");
  }
  return 0;
}