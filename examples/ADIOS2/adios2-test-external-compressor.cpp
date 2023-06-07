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
#include <hdf5.h>

#include <mpi.h>

#include "mgard/compress_x.hpp"
#include "mgard/mgard-x/Utilities/ErrorCalculator.h"

#include "cusz/compressor.hh"
#include "cusz/cusz.h"
#include "cusz/utils/io.hh"

#include "zfp.h"

#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

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

  // if (actual_error > tol)
  //   exit(-1);
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
int no_compress(mgard_x::DIM D, T *original_data, void *compressed_data,
                size_t &compressed_size, std::vector<mgard_x::SIZE> shape,
                double tol, double s, std::string eb_mode) {
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  memcpy(compressed_data, original_data, original_size * sizeof(T));
  compressed_size = original_size * sizeof(T);
  return 0;
}

int no_decompress(void *compressed_data, size_t compressed_size,
                  void *decompressed_data) {

  memcpy(decompressed_data, compressed_data, compressed_size);
  return 0;
}

template <typename T>
double mgard_compress(mgard_x::DIM D, T *original_data, void *compressed_data,
                      size_t &compressed_size, std::vector<mgard_x::SIZE> shape,
                      double tol, double s, std::string eb_mode) {
  mgard_x::Timer timer;
  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();
  mgard_x::error_bound_type mode;
  if (eb_mode.compare("abs") == 0) {
    mode = mgard_x::error_bound_type::ABS;
  } else if (eb_mode.compare("rel") == 0) {
    mode = mgard_x::error_bound_type::REL;
  } else {
    std::cout << "wrong eb_mode\n";
    exit(-1);
  }

  mgard_x::data_type dtype;
  if (std::is_same<T, double>::value) {
    dtype = mgard_x::data_type::Double;
  } else if (std::is_same<T, float>::value) {
    dtype = mgard_x::data_type::Float;
  } else {
    std::cout << "wrong dtype\n";
    exit(-1);
  }

  mgard_x::Config config;
  // config.log_level =
  //     mgard_x::log::ERR | mgard_x::log::INFO | mgard_x::log::TIME;
  config.lossless = mgard_x::lossless_type::Huffman_LZ4;
  // config.lossless = mgard_x::lossless_type::Huffman;
  // config.adjust_shape = true;
  // config.max_memory_footprint = 16e9;
  mgard_x::compress_status_type ret;
  ret = mgard_x::compress(D, dtype, shape, tol, s, mode, original_data,
                          compressed_data, compressed_size, config, true);

  if (ret != mgard_x::compress_status_type::Success) {
    std::cout << mgard_x::log::log_err << "Compression failed\n";
    exit(-1);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  timer.end();
  return timer.get();
}

double mgard_decompress(void *compressed_data, size_t compressed_size,
                        void *decompressed_data) {

  mgard_x::Timer timer;
  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();
  mgard_x::Config config;
  // config.log_level = mgard_x::log::ERR | mgard_x::log::INFO |
  // mgard_x::log::TIME;
  config.lossless = mgard_x::lossless_type::Huffman_LZ4;

  mgard_x::decompress(compressed_data, compressed_size, decompressed_data,
                      config, true);
  MPI_Barrier(MPI_COMM_WORLD);
  timer.end();
  return timer.get();
}

cusz_header *header;

template <typename T>
double sz_compress(cusz_header *header, mgard_x::DIM D, T *original_data,
                   uint8_t *compressed_data, size_t &compressed_size,
                   std::vector<mgard_x::SIZE> shape, double tol, double s,
                   std::string eb_mode) {
  mgard_x::Timer timer;

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];

  T *d_original_data;
  uint8_t *d_compressed_data;
  cudaMalloc(&d_original_data, sizeof(T) * original_size);
  cudaMemcpy(d_original_data, original_data, sizeof(T) * original_size,
             cudaMemcpyHostToDevice);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cusz_framework *framework = new cusz_custom_framework{
      .pipeline = Auto,
      .predictor = cusz_custom_predictor{.type = LorenzoI},
      .quantization = cusz_custom_quantization{.radius = 512},
      .codec = cusz_custom_codec{.type = Huffman}};

  cusz_compressor *comp;
  if (std::is_same<T, double>::value) {
    comp = cusz_create(framework, FP64);
  } else if (std::is_same<T, float>::value) {
    comp = cusz_create(framework, FP32);
  } else {
    std::cout << "wrong dtype\n";
    exit(-1);
  }

  cusz_config *config;
  if (eb_mode.compare("abs") == 0) {
    config = new cusz_config{.eb = tol, .mode = Abs};
  } else if (eb_mode.compare("rel") == 0) {
    config = new cusz_config{.eb = tol, .mode = Rel};
  } else {
    std::cout << "wrong eb_mode\n";
    exit(-1);
  }

  cusz_len uncomp_len;
  if (D == 1) {
    uncomp_len = cusz_len{shape[0], 1, 1, 1};
  } else if (D == 2) {
    uncomp_len = cusz_len{shape[1], shape[0], 1, 1};
  } else if (D == 3) {
    uncomp_len = cusz_len{shape[2], shape[1], shape[0], 1};
  } else if (D == 4) {
    uncomp_len = cusz_len{shape[3], shape[2], shape[1], shape[0]};
  } else {
    std::cout << "wrong D\n";
    exit(-1);
  }

  cusz::TimeRecord compress_timerecord;

  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();
  cusz_compress(comp, config, d_original_data, uncomp_len, &d_compressed_data,
                &compressed_size, header, &compress_timerecord, stream);
  cudaStreamSynchronize(stream);
  MPI_Barrier(MPI_COMM_WORLD);
  timer.end();

  cudaMemcpy(compressed_data, d_compressed_data, compressed_size,
             cudaMemcpyDeviceToHost);

  cudaFree(d_original_data);
  cudaFree(d_compressed_data);
  cusz_release(comp);
  cudaStreamDestroy(stream);

  return timer.get();
}

template <typename T>
double sz_compress(mgard_x::DIM D, T *original_data, uint8_t *compressed_data,
                   size_t &compressed_size, std::vector<mgard_x::SIZE> shape,
                   double tol, double s, std::string eb_mode,
                   int input_multiplier) {

  header = new cusz_header[input_multiplier];

  shape[0] /= input_multiplier;
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];

  size_t total_compressed_buffer_size = compressed_size;
  size_t total_compressed_size = 0;
  double time = 0;
  for (int i = 0; i < input_multiplier; i++) {
    compressed_size = total_compressed_buffer_size - total_compressed_size;
    time += sz_compress(&header[i], D, original_data,
                        compressed_data + sizeof(size_t), compressed_size,
                        shape, tol, s, eb_mode);
    cudaDeviceReset();
    cudaFree(0);
    *(size_t *)compressed_data = compressed_size;
    // std::cout << "compressed_size: " << compressed_size << "\n";
    total_compressed_size += compressed_size + sizeof(size_t);
    original_data += original_size;
    compressed_data += compressed_size + sizeof(size_t);
  }
  compressed_size = total_compressed_size;
  return time;
}

template <typename T>
double sz_compress2(mgard_x::DIM D, T *original_data, uint8_t *compressed_data,
                    size_t &compressed_size, std::vector<mgard_x::SIZE> shape,
                    double tol, double s, std::string eb_mode,
                    int input_multiplier) {
  header = new cusz_header[input_multiplier];
  shape[0] /= input_multiplier;
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];

  size_t total_compressed_buffer_size = compressed_size;
  size_t total_compressed_size = 0;
  uint8_t *first_compressed_data = NULL;
  size_t first_compressed_size = 0;
  double time = 0;
  for (int i = 0; i < input_multiplier; i++) {
    compressed_size = total_compressed_buffer_size - total_compressed_size;
    if (!first_compressed_data) {
      mgard_x::Timer timer;
      time = sz_compress(&header[i], D, original_data, compressed_data,
                         compressed_size, shape, tol, s, eb_mode);
      first_compressed_data = compressed_data;
      first_compressed_size = compressed_size;
    } else {
      memcpy(compressed_data, first_compressed_data, first_compressed_size);
    }
    total_compressed_size += first_compressed_size;
    original_data += original_size;
    compressed_data += first_compressed_size;
  }
  compressed_size = total_compressed_size;
  return time * input_multiplier;
}

template <typename T>
double sz_decompress(cusz_header *header, mgard_x::DIM D,
                     uint8_t *compressed_data, size_t compressed_size,
                     T *decompressed_data, std::vector<mgard_x::SIZE> shape) {
  mgard_x::Timer timer;
  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];

  cusz_framework *framework = cusz_default_framework();
  cusz_compressor *comp;
  if (std::is_same<T, double>::value) {
    comp = cusz_create(framework, FP64);
  } else if (std::is_same<T, float>::value) {
    comp = cusz_create(framework, FP32);
  } else {
    std::cout << "wrong dtype\n";
    exit(-1);
  }

  uint8_t *d_compressed_data;
  cudaMalloc(&d_compressed_data, compressed_size);
  cudaMemcpy(d_compressed_data, compressed_data, compressed_size,
             cudaMemcpyHostToDevice);

  T *d_decompressed_data;
  cudaMalloc(&d_decompressed_data, sizeof(T) * original_size);

  cusz_len decomp_len;
  if (D == 1) {
    decomp_len = cusz_len{shape[0]};
  } else if (D == 2) {
    decomp_len = cusz_len{shape[1], shape[0]};
  } else if (D == 3) {
    decomp_len = cusz_len{shape[2], shape[1], shape[0]};
  } else if (D == 4) {
    decomp_len = cusz_len{shape[3], shape[2], shape[1], shape[0]};
  } else {
    std::cout << "wrong D\n";
    exit(-1);
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cusz::TimeRecord compress_timerecord;

  cusz_decompress(comp, header, d_compressed_data, compressed_size,
                  d_decompressed_data, decomp_len, &compress_timerecord,
                  stream);
  cudaStreamSynchronize(stream);

  cudaMemcpy(decompressed_data, d_decompressed_data, sizeof(T) * original_size,
             cudaMemcpyDeviceToHost);

  cudaFree(d_decompressed_data);
  cudaFree(d_compressed_data);
  cusz_release(comp);
  cudaStreamDestroy(stream);
  MPI_Barrier(MPI_COMM_WORLD);
  timer.end();
  return timer.get();
}

template <typename T>
double sz_decompress(mgard_x::DIM D, uint8_t *compressed_data,
                     size_t compressed_size, T *decompressed_data,
                     std::vector<mgard_x::SIZE> shape, int input_multiplier) {
  shape[0] /= input_multiplier;
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  double time = 0;
  for (int i = 0; i < input_multiplier; i++) {
    compressed_size = *(size_t *)compressed_data;
    // std::cout << "compressed_size: " << compressed_size << "\n";
    time += sz_decompress(&header[i], D, compressed_data + sizeof(size_t),
                          compressed_size, decompressed_data, shape);
    cudaDeviceReset();
    cudaFree(0);
    decompressed_data += original_size;
    compressed_data += compressed_size + sizeof(size_t);
  }
  delete header;
  return time;
}

template <typename T>
double sz_decompress2(mgard_x::DIM D, uint8_t *compressed_data,
                      size_t compressed_size, T *decompressed_data,
                      std::vector<mgard_x::SIZE> shape, int input_multiplier) {
  compressed_size /= input_multiplier;
  shape[0] /= input_multiplier;
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  double time = 0;
  T *first_decompressed_data = NULL;

  for (int i = 0; i < input_multiplier; i++) {
    if (!first_decompressed_data) {
      time = sz_decompress(&header[i], D, compressed_data, compressed_size,
                           decompressed_data, shape);
      first_decompressed_data = decompressed_data;
    } else {
      memcpy(decompressed_data, first_decompressed_data,
             original_size * sizeof(T));
    }
    decompressed_data += original_size;
    compressed_data += compressed_size;
  }
  delete header;
  return time * input_multiplier;
}

template <typename T>
double zfp_compress(mgard_x::DIM D, T *original_data, uint8_t *compressed_data,
                    size_t &compressed_size, std::vector<mgard_x::SIZE> shape,
                    double tol, double s, std::string eb_mode) {
  mgard_x::Timer timer;
  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];

  zfp_type type;
  if (std::is_same<T, double>::value) {
    type = zfp_type_double;
  } else if (std::is_same<T, float>::value) {
    type = zfp_type_float;
  } else {
    std::cout << "wrong dtype\n";
    exit(-1);
  }

  zfp_field *field;
  if (D == 1) {
    field = zfp_field_1d(original_data, type, shape[0]);
  } else if (D == 2) {
    field = zfp_field_2d(original_data, type, shape[1], shape[0]);
  } else if (D == 3) {
    field = zfp_field_3d(original_data, type, shape[2], shape[1], shape[0]);
  } else if (D == 4) {
    field = zfp_field_4d(original_data, type, shape[3], shape[2], shape[1],
                         shape[0]);
  } else {
    std::cout << "wrong D\n";
    exit(-1);
  }

  zfp_stream *zfp = zfp_stream_open(NULL);
  zfp_stream_set_rate(zfp, tol, type, zfp_field_dimensionality(field),
                      zfp_false);

  size_t bufsize = zfp_stream_maximum_size(zfp, field);
  if (bufsize > compressed_size) {
    std::cout << "not enough buffer space\n";
    std::cout << bufsize << " vs. " << compressed_size << "\n";
    exit(-1);
  }

  bitstream *stream = stream_open(compressed_data, compressed_size);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  if (!zfp_stream_set_execution(zfp, zfp_exec_cuda)) {
    std::cout << "zfp-cuda not available\n";
    exit(-1);
  }

  // std::cout << "orginal: ";
  // for (int i = 0; i < 10; i++) std::cout << original_data[i] << " ";
  // std::cout << "\n";

  compressed_size = zfp_compress(zfp, field);

  if (compressed_size == 0) {
    std::cout << "zfp-cuda compress error\n";
    exit(-1);
  }

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(stream);
  MPI_Barrier(MPI_COMM_WORLD);
  timer.end();
  return timer.get();
}

template <typename T>
double zfp_compress(mgard_x::DIM D, T *original_data, uint8_t *compressed_data,
                    size_t &compressed_size, std::vector<mgard_x::SIZE> shape,
                    double tol, double s, std::string eb_mode,
                    int input_multiplier) {
  double time = 0;
  shape[0] /= input_multiplier;
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];

  size_t total_compressed_buffer_size = compressed_size;
  size_t total_compressed_size = 0;
  for (int i = 0; i < input_multiplier; i++) {
    compressed_size = total_compressed_buffer_size - total_compressed_size;
    time += zfp_compress(D, original_data, compressed_data + sizeof(size_t),
                         compressed_size, shape, tol, s, eb_mode);
    *(size_t *)compressed_data = compressed_size;
    total_compressed_size += compressed_size + sizeof(size_t);
    original_data += original_size;
    compressed_data += compressed_size + sizeof(size_t);
  }
  compressed_size = total_compressed_size;
  return time;
}

template <typename T>
double zfp_decompress(mgard_x::DIM D, uint8_t *compressed_data,
                      size_t &compressed_size, T *decompressed_data,
                      std::vector<mgard_x::SIZE> shape, double tol, double s,
                      std::string eb_mode) {
  mgard_x::Timer timer;
  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];

  zfp_type type;
  if (std::is_same<T, double>::value) {
    type = zfp_type_double;
  } else if (std::is_same<T, float>::value) {
    type = zfp_type_float;
  } else {
    std::cout << "wrong dtype\n";
    exit(-1);
  }

  zfp_field *field;
  if (D == 1) {
    field = zfp_field_1d(decompressed_data, type, shape[0]);
  } else if (D == 2) {
    field = zfp_field_2d(decompressed_data, type, shape[1], shape[0]);
  } else if (D == 3) {
    field = zfp_field_3d(decompressed_data, type, shape[2], shape[1], shape[0]);
  } else if (D == 4) {
    field = zfp_field_4d(decompressed_data, type, shape[3], shape[2], shape[1],
                         shape[0]);
  } else {
    std::cout << "wrong D\n";
    exit(-1);
  }

  zfp_stream *zfp = zfp_stream_open(NULL);
  zfp_stream_set_rate(zfp, tol, type, zfp_field_dimensionality(field),
                      zfp_false);

  bitstream *stream = stream_open(compressed_data, compressed_size);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  if (!zfp_stream_set_execution(zfp, zfp_exec_cuda)) {
    std::cout << "zfp-cuda not available\n";
    exit(-1);
  }

  int status = zfp_decompress(zfp, field);

  if (!status) {
    std::cout << "zfp-cuda decompress error\n";
    exit(-1);
  }

  // std::cout << "decompressed: ";
  // for (int i = 0; i < 10; i++) std::cout << decompressed_data[i] << " ";
  // std::cout << "\n";

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(stream);
  MPI_Barrier(MPI_COMM_WORLD);
  timer.end();
  return timer.get();
}

template <typename T>
double zfp_decompress(mgard_x::DIM D, uint8_t *compressed_data,
                      size_t compressed_size, T *decompressed_data,
                      std::vector<mgard_x::SIZE> shape, double tol, double s,
                      std::string eb_mode, int input_multiplier) {
  double time = 0;
  shape[0] /= input_multiplier;
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  for (int i = 0; i < input_multiplier; i++) {
    compressed_size = *(size_t *)compressed_data;
    time += zfp_decompress(D, compressed_data + sizeof(size_t), compressed_size,
                           decompressed_data, shape, tol, s, eb_mode);
    decompressed_data += original_size;
    compressed_data += compressed_size + sizeof(size_t);
  }
  return time;
}

template <typename T>
double nvcomp_compress(mgard_x::DIM D, T *original_data,
                       uint8_t *compressed_data, size_t &compressed_size,
                       std::vector<mgard_x::SIZE> shape, double tol, double s,
                       std::string eb_mode) {
  mgard_x::Timer timer;
  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];

  T *d_original_data;
  uint8_t *d_compressed_data;
  cudaMalloc(&d_original_data, sizeof(T) * original_size);
  cudaMalloc(&d_compressed_data, compressed_size);
  cudaMemcpy(d_original_data, original_data, sizeof(T) * original_size,
             cudaMemcpyHostToDevice);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t chunk_size = 1 << 15;
  nvcompType_t dtype = NVCOMP_TYPE_UCHAR;
  nvcomp::LZ4Manager nvcomp_manager{chunk_size, dtype, stream};
  size_t input_count = original_size * sizeof(T);
  nvcomp::CompressionConfig comp_config =
      nvcomp_manager.configure_compression(input_count);
  nvcomp_manager.compress((uint8_t *)d_original_data, d_compressed_data,
                          comp_config);
  compressed_size =
      nvcomp_manager.get_compressed_output_size(d_compressed_data);
  cudaStreamSynchronize(stream);

  cudaMemcpy(compressed_data, d_compressed_data, compressed_size,
             cudaMemcpyDeviceToHost);

  cudaFree(d_original_data);
  cudaFree(d_compressed_data);
  cudaStreamDestroy(stream);
  MPI_Barrier(MPI_COMM_WORLD);
  timer.end();
  return timer.get();
}

template <typename T>
double nvcomp_compress(mgard_x::DIM D, T *original_data,
                       uint8_t *compressed_data, size_t &compressed_size,
                       std::vector<mgard_x::SIZE> shape, double tol, double s,
                       std::string eb_mode, int input_multiplier) {

  shape[0] /= input_multiplier;
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];

  size_t total_compressed_buffer_size = compressed_size;
  size_t total_compressed_size = 0;
  double time = 0;
  for (int i = 0; i < input_multiplier; i++) {
    compressed_size = total_compressed_buffer_size - total_compressed_size;
    time += nvcomp_compress(D, original_data, compressed_data + sizeof(size_t),
                            compressed_size, shape, tol, s, eb_mode);
    *(size_t *)compressed_data = compressed_size;
    total_compressed_size += compressed_size + sizeof(size_t);
    original_data += original_size;
    compressed_data += compressed_size + sizeof(size_t);
  }
  compressed_size = total_compressed_size;
  return time;
}

template <typename T>
double nvcomp_decompress(mgard_x::DIM D, uint8_t *compressed_data,
                         size_t compressed_size, T *decompressed_data,
                         std::vector<mgard_x::SIZE> shape) {
  mgard_x::Timer timer;
  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];

  uint8_t *d_compressed_data;
  cudaMalloc(&d_compressed_data, compressed_size);
  cudaMemcpy(d_compressed_data, compressed_data, compressed_size,
             cudaMemcpyHostToDevice);

  T *d_decompressed_data;
  cudaMalloc(&d_decompressed_data, sizeof(T) * original_size);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto decomp_nvcomp_manager =
      nvcomp::create_manager(d_compressed_data, stream);

  nvcomp::DecompressionConfig decomp_config =
      decomp_nvcomp_manager->configure_decompression(d_compressed_data);

  decomp_nvcomp_manager->decompress((uint8_t *)d_decompressed_data,
                                    d_compressed_data, decomp_config);
  cudaStreamSynchronize(stream);
  cudaMemcpy(decompressed_data, d_decompressed_data, sizeof(T) * original_size,
             cudaMemcpyDeviceToHost);

  cudaFree(d_decompressed_data);
  cudaFree(d_compressed_data);
  cudaStreamDestroy(stream);
  MPI_Barrier(MPI_COMM_WORLD);
  timer.end();
  return timer.get();
}

template <typename T>
double nvcomp_decompress(mgard_x::DIM D, uint8_t *compressed_data,
                         size_t compressed_size, T *decompressed_data,
                         std::vector<mgard_x::SIZE> shape,
                         int input_multiplier) {
  shape[0] /= input_multiplier;
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  double time = 0;
  for (int i = 0; i < input_multiplier; i++) {
    compressed_size = *(size_t *)compressed_data;
    // std::cout << "compressed_size: " << compressed_size << "\n";
    time += nvcomp_decompress(D, compressed_data + sizeof(size_t),
                              compressed_size, decompressed_data, shape);
    decompressed_data += original_size;
    compressed_data += compressed_size + sizeof(size_t);
  }
  delete header;
  return time;
}

template <typename T>
int launch_compress(mgard_x::DIM D, enum mgard_x::data_type dtype,
                    std::string input_file, std::string output_file,
                    std::string log_file, std::string var_name, int step_start,
                    int step_end, std::vector<size_t> shape, double tol,
                    double s, std::string eb_mode, int compressor_type,
                    int input_file_type, int input_multiplier,
                    int compress_block) {

  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  adios2::ADIOS adios(MPI_COMM_WORLD);
  adios2::IO read_io = adios.DeclareIO("Input Data");
  adios2::IO write_io = adios.DeclareIO("Output Data");
  read_io.SetEngine("BP4");
  write_io.SetEngine("BP5");
  adios2::Engine reader;
  if (input_file_type) {
    reader = read_io.Open(input_file, adios2::Mode::Read);
  }
  adios2::Engine writer = write_io.Open(output_file, adios2::Mode::Write);

  using C = unsigned char;

  adios2::Variable<T> org_var;
  adios2::Variable<C> cmp_var;

  adios2::Dims cmp_shape;
  adios2::Dims cmp_start_local;
  adios2::Dims cmp_count_local;

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];

  size_t original_size_multiplied = original_size * input_multiplier;

  size_t compressed_size = original_size_multiplied * sizeof(T) * 1.5;
  void *compressed_data = (void *)malloc(compressed_size);

  bool first = true;

  double compress_time = 0, write_time = 0;

  for (int sim_iter = 0; sim_iter <= step_end; sim_iter++) {
    std::vector<T> var_data_vec;

    if (input_file_type) {
      reader.BeginStep();
      // Input variable
      org_var = read_io.InquireVariable<T>(var_name);
      if (sim_iter >= step_start) {
        reader.Get<T>(org_var, var_data_vec, adios2::Mode::Sync);
        var_data_vec.resize(original_size_multiplied);
      }
      reader.EndStep();
    } else {
      std::fstream myfile;
      myfile.open(input_file, std::ios::in | std::ios::binary);
      if (!myfile) {
        printf("Error: cannot open file\n");
      }
      var_data_vec.resize(original_size_multiplied);
      myfile.read((char *)var_data_vec.data(), original_size * sizeof(T));
      myfile.close();
    }

    for (int i = 1; i < input_multiplier; i++) {
      memcpy(var_data_vec.data() + i * original_size, var_data_vec.data(),
             original_size * sizeof(T));
    }
    shape[0] *= input_multiplier;

    {
      // std::fstream myfile;
      // myfile.open("QMCPACK.dat", std::ios::out | std::ios::binary);
      // if (!myfile) {
      //   printf("Error: cannot open file\n");
      // }
      // myfile.write((char *)var_data_vec.data(),
      //              original_size_multiplied * sizeof(T));
      // myfile.close();
    }

    if (compressor_type != 3) {
      mgard_x::pin_memory(var_data_vec.data(),
                          original_size_multiplied * sizeof(T),
                          mgard_x::Config());
      mgard_x::pin_memory(compressed_data, compressed_size, mgard_x::Config());
    }

    if (eb_mode.compare("rel") == 0 && compressor_type != 3) {
      T norm =
          mgard_x::L_inf_norm(original_size_multiplied, var_data_vec.data());
      std::cout << "abs error bound: " << norm * tol;
    }

    // call externl compressors
    if (compressor_type == 0) {
      no_compress(D, var_data_vec.data(), compressed_data, compressed_size,
                  shape, tol, s, eb_mode);
    } else if (compressor_type == 1) {
      compress_time = mgard_compress(D, var_data_vec.data(), compressed_data,
                                     compressed_size, shape, tol, s, eb_mode);
    } else if (compressor_type == 2) {
      compress_time = sz_compress2(D, var_data_vec.data(), (C *)compressed_data,
                                   compressed_size, shape, tol, s, eb_mode,
                                   input_multiplier * compress_block);
    } else if (compressor_type == 3) {
      compress_time = zfp_compress(D, var_data_vec.data(), (C *)compressed_data,
                                   compressed_size, shape, tol, s, eb_mode,
                                   input_multiplier * compress_block);
    } else if (compressor_type == 4) {
      compress_time = nvcomp_compress(
          D, var_data_vec.data(), (C *)compressed_data, compressed_size, shape,
          tol, s, eb_mode, input_multiplier * compress_block);
    }

    if (compressor_type != 3) {
      mgard_x::unpin_memory(var_data_vec.data(), mgard_x::Config());
      mgard_x::unpin_memory(compressed_data, mgard_x::Config());
    }

    cmp_shape = std::vector<size_t>({compressed_size * comm_size});
    cmp_start_local = std::vector<size_t>({compressed_size * rank});
    cmp_count_local = std::vector<size_t>({compressed_size});
    // std::cout << "rank " << rank << " compressed: ";
    // print_shape(cmp_start_local);
    // print_shape(cmp_count_local);
    // std::cout << "\n";

    mgard_x::Timer timer;
    MPI_Barrier(MPI_COMM_WORLD);
    timer.start();
    writer.BeginStep();
    if (first) {
      // Output variable
      cmp_var = write_io.DefineVariable<C>(var_name, cmp_shape, cmp_start_local,
                                           cmp_count_local);
      first = false;
    }
    if (sim_iter >= step_start) {
      writer.Put<C>(cmp_var, (C *)compressed_data, adios2::Mode::Sync);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    writer.EndStep();
    timer.end();
    write_time = timer.get();
    timer.clear();

    if (!rank) {
      double compress_throughput = (double)original_size_multiplied *
                                   sizeof(T) * comm_size / compress_time / 1e9;
      double write_throughput = (double)original_size_multiplied * sizeof(T) *
                                comm_size / (compress_time + write_time) / 1e9;

      std::cout << mgard_x::log::log_info << "Compression ratio: "
                << (double)original_size_multiplied * sizeof(T) /
                       compressed_size
                << "\n";
      std::cout << mgard_x::log::log_info
                << "Compression time: " << compress_time << " s.\n";
      std::cout << mgard_x::log::log_info
                << "Compression throughput: " << compress_throughput
                << " GB/s.\n";
      std::cout << mgard_x::log::log_info << "Write time: " << write_time
                << " s.\n";
      std::cout << mgard_x::log::log_info
                << "Write throughput: " << write_throughput << " GB/s.\n";

      std::fstream myfile;
      myfile.open(log_file, std::ios::out | std::ios::trunc);
      myfile << compress_time << ",";
      myfile << compress_throughput << ",";
      myfile << write_time << ",";
      myfile << write_throughput << ",";
      myfile.close();
    }
  }
  if (input_file_type)
    reader.Close();
  writer.Close();

  free(compressed_data);

  return 0;
}

template <typename T>
int launch_decompress(mgard_x::DIM D, enum mgard_x::data_type dtype,
                      std::string org_file, std::string cmp_file,
                      std::string log_file, std::string var_name,
                      int step_start, int step_end, std::vector<size_t> shape,
                      double tol, double s, std::string eb_mode,
                      int compressor_type, int input_file_type,
                      int input_multiplier, int compress_block) {

  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  adios2::ADIOS adios(MPI_COMM_WORLD);
  adios2::IO org_io = adios.DeclareIO("Original Data");
  adios2::IO cmp_io = adios.DeclareIO("Compressed Data");
  org_io.SetEngine("BP4");
  cmp_io.SetEngine("BP5");
  adios2::Engine org_reader;
  if (input_file_type) {
    org_reader = org_io.Open(org_file, adios2::Mode::Read);
  }
  adios2::Engine cmp_reader = cmp_io.Open(cmp_file, adios2::Mode::Read);

  using C = unsigned char;

  adios2::Variable<T> org_var;
  adios2::Variable<C> cmp_var;

  adios2::Dims cmp_shape;
  adios2::Dims cmp_start_local;
  adios2::Dims cmp_count_local;

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  size_t original_size_multiplied = original_size * input_multiplier;

  size_t decompressed_size = original_size_multiplied * sizeof(T);
  void *decompressed_data = (void *)malloc(decompressed_size);
  size_t compressed_size;

  double decompress_time = 0, read_time = 0;

  for (int sim_iter = 0; sim_iter <= step_end; sim_iter++) {
    std::vector<T> org_vec;
    std::vector<C> cmp_vec;

    if (input_file_type) {
      org_reader.BeginStep();
      if (sim_iter >= step_start) {
        org_var = org_io.InquireVariable<T>(var_name);
        org_reader.Get<T>(org_var, org_vec, adios2::Mode::Sync);
        org_vec.resize(original_size_multiplied);
      }
      org_reader.EndStep();
    } else {
      std::fstream myfile;
      myfile.open(org_file, std::ios::in | std::ios::binary);
      if (!myfile) {
        printf("Error: cannot open file\n");
      }
      org_vec.resize(original_size_multiplied);
      myfile.read((char *)org_vec.data(), original_size * sizeof(T));
      myfile.close();
    }

    for (int i = 1; i < input_multiplier; i++) {
      memcpy(org_vec.data() + i * original_size, org_vec.data(),
             original_size * sizeof(T));
    }
    shape[0] *= input_multiplier;

    mgard_x::Timer timer;

    MPI_Barrier(MPI_COMM_WORLD);
    timer.start();
    cmp_reader.BeginStep();
    // Compressed variable
    cmp_var = cmp_io.InquireVariable<C>(var_name);

    cmp_shape = cmp_var.Shape();
    compressed_size = cmp_shape[0] / comm_size;
    cmp_start_local = std::vector<size_t>({compressed_size * rank});
    cmp_count_local = std::vector<size_t>({compressed_size});
    adios2::Box<adios2::Dims> cmp_sel(cmp_start_local, cmp_count_local);
    cmp_var.SetSelection(cmp_sel);
    // std::cout << "rank " << rank << " compressed: ";
    // print_shape(cmp_start_local);
    // print_shape(cmp_count_local);
    // std::cout << "\n";

    if (sim_iter >= step_start) {
      cmp_reader.Get<C>(cmp_var, cmp_vec, adios2::Mode::Sync);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    cmp_reader.EndStep();
    timer.end();
    read_time = timer.get();
    timer.clear();

    if (compressor_type != 3) {
      mgard_x::pin_memory(decompressed_data,
                          original_size_multiplied * sizeof(T),
                          mgard_x::Config());
      mgard_x::pin_memory(cmp_vec.data(), compressed_size, mgard_x::Config());
    }

    if (compressor_type == 0) {
      no_decompress((void *)cmp_vec.data(), compressed_size, decompressed_data);
    } else if (compressor_type == 1) {
      decompress_time = mgard_decompress((void *)cmp_vec.data(),
                                         compressed_size, decompressed_data);
    } else if (compressor_type == 2) {
      decompress_time = sz_decompress2(D, cmp_vec.data(), compressed_size,
                                       (T *)decompressed_data, shape,
                                       input_multiplier * compress_block);
    } else if (compressor_type == 3) {
      decompress_time = zfp_decompress(
          D, cmp_vec.data(), compressed_size, (T *)decompressed_data, shape,
          tol, s, eb_mode, input_multiplier * compress_block);
    } else if (compressor_type == 4) {
      decompress_time = nvcomp_decompress(D, cmp_vec.data(), compressed_size,
                                          (T *)decompressed_data, shape,
                                          input_multiplier * compress_block);
    }

    if (compressor_type != 3) {
      mgard_x::unpin_memory(decompressed_data, mgard_x::Config());
      mgard_x::unpin_memory(cmp_vec.data(), mgard_x::Config());
    }

    if (!rank) {
      double decompress_throughput = (double)original_size_multiplied *
                                     sizeof(T) * comm_size / decompress_time /
                                     1e9;
      double read_throughput = (double)original_size_multiplied * sizeof(T) *
                               comm_size / (decompress_time + read_time) / 1e9;
      std::cout << mgard_x::log::log_info
                << "Decompression time: " << decompress_time << " s.\n";
      std::cout << mgard_x::log::log_info
                << "Decompression throughput: " << decompress_throughput
                << " GB/s.\n";
      std::cout << mgard_x::log::log_info << "Read time: " << read_time
                << " s.\n";
      std::cout << mgard_x::log::log_info
                << "Read throughput: " << read_throughput << " GB/s.\n";

      print_statistics(s, eb_mode, shape, org_vec.data(),
                       (T *)decompressed_data, tol, true);

      std::fstream myfile;
      myfile.open(log_file, std::ios::out | std::ios::app);
      myfile << decompress_time << ",";
      myfile << decompress_throughput << ",";
      myfile << read_time << ",";
      myfile << read_throughput << ",";
      myfile.close();
    }
  }

  if (input_file_type)
    org_reader.Close();
  cmp_reader.Close();

  free(decompressed_data);
  return 0;
}

bool try_exec(int argc, char *argv[]) {
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string input_file = get_arg(argc, argv, "-i");
  std::string output_file = get_arg(argc, argv, "-c");
  std::string log_file = get_arg(argc, argv, "-o");

  if (!rank)
    std::cout << mgard_x::log::log_info << "input data: " << input_file << "\n";
  if (!rank)
    std::cout << mgard_x::log::log_info << "output data: " << output_file
              << "\n";
  if (!rank)
    std::cout << mgard_x::log::log_info << "log_file: " << log_file << "\n";

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

  int compressor_type = get_arg_int(argc, argv, "-p");

  int input_file_type = get_arg_int(argc, argv, "-u");

  int input_multiplier = get_arg_int(argc, argv, "-r");

  int compress_block = get_arg_int(argc, argv, "-k");

  // if (compress_or_decompress == 0) {
  if (dtype == mgard_x::data_type::Double) {
    launch_compress<double>(D, dtype, input_file, output_file, log_file,
                            var_name, step_start, step_end, shape, tol, s,
                            eb_mode, compressor_type, input_file_type,
                            input_multiplier, compress_block);
    launch_decompress<double>(D, dtype, input_file, output_file, log_file,
                              var_name, step_start, step_end, shape, tol, s,
                              eb_mode, compressor_type, input_file_type,
                              input_multiplier, compress_block);
  } else if (dtype == mgard_x::data_type::Float) {
    launch_compress<float>(D, dtype, input_file, output_file, log_file,
                           var_name, step_start, step_end, shape, tol, s,
                           eb_mode, compressor_type, input_file_type,
                           input_multiplier, compress_block);
    launch_decompress<float>(D, dtype, input_file, output_file, log_file,
                             var_name, step_start, step_end, shape, tol, s,
                             eb_mode, compressor_type, input_file_type,
                             input_multiplier, compress_block);
  }
  // } else {
  //   if (dtype == mgard_x::data_type::Double) {
  //     launch_decompress<double>(D, dtype, input_file, output_file,
  //                               decompressed_file, var_name, step_start,
  //                               step_end, shape, tol, s, eb_mode);
  //   } else if (dtype == mgard_x::data_type::Float) {
  //     launch_decompress<float>(D, dtype, input_file, output_file,
  //                              decompressed_file, var_name, step_start,
  //                              step_end, shape, tol, s, eb_mode);
  //   }
  // }
  return true;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  if (!try_exec(argc, argv)) {
    print_usage_message("");
  }
  MPI_Finalize();
  exit(0);
  return 0;
}
