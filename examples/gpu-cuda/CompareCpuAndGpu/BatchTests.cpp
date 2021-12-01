/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgard/compress.hpp"
// #include "compress_cuda.hpp"
#include "mgard/mgard-x/Utilities/ErrorCalculator.h"

#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_RESET "\x1b[0m"

using namespace std::chrono;

enum device { CPU, GPU };
enum data_type { SINGLE, DOUBLE };

struct Result {
  double actual_error;
  double cr;
};

void print_usage_message(char *argv[], FILE *fp) {
  fprintf(fp,
          "Usage: %s [input file] [num. of dimensions] [1st dim.] [2nd dim.] "
          "[3rd. dim] ... [tolerance] [s]\n",
          argv[0]);
  exit(0);
}

template <typename T>
void readfile(char *input_file, size_t num_bytes, bool check_size, T *in_buff) {
  if (strcmp(input_file, "random") == 0) {
    srand(7117);
    for (mgard_x::LENGTH i = 0; i < num_bytes / sizeof(T); i++) {
      in_buff[i] = rand() % 10 + 1;
    }
  } else {
    fprintf(stdout, "Loading file: %s\n", input_file);
    FILE *pFile;
    pFile = fopen(input_file, "rb");
    if (pFile == NULL) {
      fputs("File error", stderr);
      exit(1);
    }
    fseek(pFile, 0, SEEK_END);
    size_t lSize = ftell(pFile);

    rewind(pFile);

    if (check_size && lSize != num_bytes) {
      fprintf(stderr,
              "%s contains %lu bytes when %lu were expected. Exiting.\n",
              input_file, lSize, num_bytes);
      exit(1);
    }

    size_t result = fread(in_buff, 1, num_bytes, pFile);
    if (result != num_bytes) {
      fputs("Reading error", stderr);
      exit(3);
    }
    fclose(pFile);
  }
}

template <mgard_x::DIM D, typename T>
void compression(std::vector<mgard_x::SIZE> shape, enum device dev, T tol,
                 T s, enum mgard_x::error_bound_type mode, T norm,
                 T *original_data, void *&compressed_data,
                 size_t &compressed_size, mgard_x::Config config,
                 enum mgard_x::device_type dev_type) {
  // printf("Start compressing\n");
  std::array<std::size_t, D> array_shape;
  std::copy(shape.begin(), shape.end(), array_shape.begin());
  if (dev == CPU) {
    if (mode == mgard_x::error_bound_type::REL)
      tol *= norm;
    const mgard::TensorMeshHierarchy<D, T> hierarchy(array_shape);
    mgard::CompressedDataset<D, T> compressed_dataset =
        mgard::compress(hierarchy, original_data, s, tol);
    compressed_size = compressed_dataset.size();
    compressed_data = (void *)malloc(compressed_size);
    memcpy(compressed_data, compressed_dataset.data(), compressed_size);
  } else {
    mgard_cuda::data_type dtype;
    if (std::is_same<T, double>::value) {
      dtype = mgard_cuda::data_type::Double;
    } else if (std::is_same<T, float>::value) {
      dtype = mgard_x::data_type::Float;
    }

    mgard_x::compress(D, dtype, shape, tol, s, mode, original_data,
             compressed_data, compressed_size, config, dev_type, false);
    // mgard_x::compress(D, dtype, shape, tol, s, mode, original_data,
    //          compressed_data, compressed_size);
  }
}

template <mgard_x::DIM D, typename T>
void decompression(std::vector<mgard_x::SIZE> shape, enum device dev, T tol,
                   T s, enum mgard_x::error_bound_type mode, T norm,
                   void *compressed_data, size_t compressed_size,
                   void *&decompressed_data, mgard_x::Config config,
                   enum mgard_x::device_type dev_type) {
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];

  if (dev == CPU) {
    decompressed_data = (T *)malloc(original_size * sizeof(T));
    if (mode == mgard_x::error_bound_type::REL) {
      tol *= norm;
    }
    const std::unique_ptr<unsigned char const[]> new_data_ =
        mgard::decompress((unsigned char *)compressed_data, compressed_size);
    const void *decompressed_data_void = new_data_.get();
    memcpy(decompressed_data, decompressed_data_void,
           original_size * sizeof(T));
  } else { // GPU
    mgard_x::decompress(compressed_data, compressed_size, decompressed_data, config, dev_type, false);
    // mgard_x::decompress(compressed_data, compressed_size, decompressed_data);
  }
}

template <typename T>
struct Result test(mgard_x::DIM D, T *original_data,
                   std::vector<mgard_x::SIZE> shape, enum device dev,
                   double tol, double s,
                   enum mgard_x::error_bound_type mode,
                   enum mgard_x::device_type dev_type) {

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  // T * original_data = (T*)malloc(original_size * sizeof(T));
  // readfile(input_file, original_size * sizeof(T), false, original_data);

  T norm;
  if (s == std::numeric_limits<T>::infinity()) {
    norm = mgard_x::L_inf_norm(original_size, original_data);
  } else {
    norm = mgard_x::L_2_norm(original_size, original_data);
  }

  mgard_x::Config config;
  config.lossless = mgard_x::lossless_type::GPU_Huffman_LZ4;
  config.uniform_coord_mode = 1;
  // config.huff_dict_size = 64;
  config.timing = false;

  void *compressed_data = NULL;
  size_t compressed_size = 0;
  void *decompressed_data = NULL;
  if (D == 1) {
    compression<1, T>(shape, dev, tol, s, mode, norm, original_data,
                      compressed_data, compressed_size, config, dev_type);
    decompression<1, T>(shape, dev, tol, s, mode, norm, compressed_data,
                        compressed_size, decompressed_data, config, dev_type);
  }
  if (D == 2) {
    compression<2, T>(shape, dev, tol, s, mode, norm, original_data,
                      compressed_data, compressed_size, config, dev_type);
    decompression<2, T>(shape, dev, tol, s, mode, norm, compressed_data,
                        compressed_size, decompressed_data, config, dev_type);
  }
  if (D == 3) {
    compression<3, T>(shape, dev, tol, s, mode, norm, original_data,
                      compressed_data, compressed_size, config, dev_type);
    decompression<3, T>(shape, dev, tol, s, mode, norm, compressed_data,
                        compressed_size, decompressed_data, config, dev_type);
  }
  if (D == 4) {
    compression<4, T>(shape, dev, tol, s, mode, norm, original_data,
                      compressed_data, compressed_size, config, dev_type);
    decompression<4, T>(shape, dev, tol, s, mode, norm, compressed_data,
                        compressed_size, decompressed_data, config, dev_type);
  }
  if (D == 5) {
    compression<5, T>(shape, dev, tol, s, mode, norm, original_data,
                      compressed_data, compressed_size, config, dev_type);
    decompression<5, T>(shape, dev, tol, s, mode, norm, compressed_data,
                        compressed_size, decompressed_data, config, dev_type);
  }

  // printf("In size:  %10ld  Out size: %10ld  Compression ratio: %10ld \n",
  // original_size * sizeof(T),
  //        compressed_size, original_size * sizeof(T) / compressed_size);

  T error;
  if (s == std::numeric_limits<T>::infinity()) {
    error = mgard_x::L_inf_error(original_size, original_data,
                                    (T*)decompressed_data, mode);
    // if (mode == mgard_x::REL) {
    //   error /= norm; printf("Rel. L^infty error: %10.5E \n", error);
    // }
    // if (mode ==  mgard_x::ABS) printf("Abs. L^infty error: %10.5E \n",
    // error);
  } else {
    error = mgard_x::L_2_error(original_size, original_data,
                                  (T*)decompressed_data, mode);
    // if (mode == mgard_x::REL) {
    //   error /= norm; printf("Rel. L^2 error: %10.5E \n", error);
    // }
    // if (mode ==  mgard_x::ABS) printf("Abs. L^2 error: %10.5E \n", error);
  }

  // if (error < tol) {
  //   printf(ANSI_GREEN "SUCCESS: Error tolerance met!" ANSI_RESET "\n");
  //   return 0;
  // } else {
  //   printf(ANSI_RED "FAILURE: Error tolerance NOT met!" ANSI_RESET "\n");
  //   return -1;
  // }

  struct Result result;
  result.actual_error = error;
  result.cr = original_size * sizeof(T) / compressed_size;
  return result;
}

void print_config(enum data_type dtype, std::vector<mgard_x::SIZE> shape,
                  double tol, double s,
                  enum mgard_x::error_bound_type mode) {
  mgard_x::DIM d = 0;
  for (d = 0; d < shape.size(); d++)
    std::cout << std::setw(5) << shape[d];
  for (; d < 5; d++)
    std::cout << std::setw(5) << "";
  if (dtype == DOUBLE)
    std::cout << std::setw(3) << "64";
  if (dtype == SINGLE)
    std::cout << std::setw(3) << "32";
  if (mode == mgard_x::error_bound_type::REL)
    std::cout << std::setw(4) << "Rel";
  if (mode == mgard_x::error_bound_type::ABS)
    std::cout << std::setw(4) << "Abs";
  std::cout << std::setw(6) << std::setprecision(0) << std::scientific << tol;
  std::cout << std::setw(6) << std::setprecision(1) << std::fixed << s;
  // if (dev == CPU) std::cout << std::setw(4) << "CPU\n";
  // if (dev == GPU) std::cout << std::setw(4) << "GPU\n";
}

int main(int argc, char *argv[]) {
  if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
    print_usage_message(argv, stdout);
  }

  int i = 1;

  char *input_file; //, *outfile;
  input_file = argv[i++];

  char *dev; //, *outfile;
  dev = argv[i++];



  enum mgard_x::device_type dev_type;

  if (strcmp (dev, "serial") == 0) {
    dev_type = mgard_x::device_type::Serial;
  } else if (strcmp (dev, "cuda") == 0) {
    dev_type = mgard_x::device_type::CUDA;
  } else if (strcmp (dev, "hip") == 0) {
    dev_type = mgard_x::device_type::HIP;
  }


  std::vector<std::vector<mgard_x::SIZE>> shapes;

  shapes.push_back({5});
  shapes.push_back({129});
  shapes.push_back({100});
  shapes.push_back({400});
  shapes.push_back({1000});

  shapes.push_back({5, 5});
  shapes.push_back({129, 129});
  shapes.push_back({100, 100});
  shapes.push_back({1000, 1000});
  shapes.push_back({100, 1000});
  shapes.push_back({1000, 100});
  shapes.push_back({10, 1000});
  shapes.push_back({1000, 10});

  shapes.push_back({5, 5, 5});
  shapes.push_back({129, 129, 129});
  shapes.push_back({100, 100, 100});
  shapes.push_back({200, 200, 200});
  shapes.push_back({1000, 100, 10});
  shapes.push_back({100, 10, 1000});
  shapes.push_back({10, 1000, 100});

  shapes.push_back({5, 5, 5, 5});
  shapes.push_back({3, 3, 3, 4});
  // shapes.push_back({65, 65, 65, 65});
  shapes.push_back({100, 10, 100, 10});
  shapes.push_back({10, 100, 10, 100});
  shapes.push_back({1000, 10, 10, 10});
  shapes.push_back({10, 1000, 10, 10});
  shapes.push_back({10, 10, 1000, 10});
  shapes.push_back({10, 10, 10, 1000});

  // XGC
  // shapes.push_back({8, 16395, 39, 39});

  shapes.push_back({5, 5, 5, 5, 5});
  shapes.push_back({17, 17, 17, 17, 17});
  // shapes.push_back({10, 10, 10, 10, 100});
  // shapes.push_back({10, 10, 10, 100, 10});
  // shapes.push_back({10, 10, 100, 10, 10});
  // shapes.push_back({10, 100, 10, 10, 10});
  // shapes.push_back({100, 10, 10, 10, 10});
  // shapes.push_back({10, 10, 10, 100, 100});
  // shapes.push_back({10, 10, 100, 10, 100});
  // shapes.push_back({10, 100, 10, 100, 10});
  // shapes.push_back({100, 10, 100, 10, 10});

  // std::vector<enum data_type> dtypes = {data_type::SINGLE,
  // data_type::DOUBLE};
  // std::vector<enum data_type> dtypes = {data_type::SINGLE};
  std::vector<enum data_type> dtypes = {data_type::DOUBLE};

  // std::vector<enum mgard_x::error_bound_type> ebtypes = {
  //     mgard_x::error_bound_type::ABS, mgard_x::error_bound_type::REL};
  std::vector<enum mgard_x::error_bound_type> ebtypes = {mgard_x::error_bound_type::REL};

  // std::vector<float> tols = {1e-2, 1e-3, 1e-4};
      std::vector<float> tols = {1e-4};

  std::vector<double> told = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};

  std::vector<float> ssf = {std::numeric_limits<float>::infinity(), 0, 1, -1};
  std::vector<double> ssd = {std::numeric_limits<double>::infinity(), 0, 1, -1};

  for (mgard_x::DIM sp = 0; sp < shapes.size(); sp++) {
    for (mgard_x::DIM dt = 0; dt < dtypes.size(); dt++) {
      for (mgard_x::DIM ebt = 0; ebt < ebtypes.size(); ebt++) {
        for (mgard_x::DIM s = 0; s < ssd.size(); s++) {
          for (mgard_x::DIM tol = 0; tol < tols.size(); tol++) {
            struct Result result_cpu, result_gpu;
            if (dtypes[dt] == SINGLE) {
              size_t original_size = 1;
              for (mgard_x::DIM i = 0; i < shapes[sp].size(); i++)
                original_size *= shapes[sp][i];
              float *original_data =
                  (float *)malloc(original_size * sizeof(float));
              readfile(input_file, original_size * sizeof(float), false,
                       original_data);
              if (dev_type == mgard_x::device_type::Serial) {
                result_cpu =
                    test<float>(shapes[sp].size(), original_data, shapes[sp],
                                 CPU, told[tol], ssd[s], ebtypes[ebt], dev_type);
              } else {
                result_cpu =
                    test<float>(shapes[sp].size(), original_data, shapes[sp],
                                 GPU, told[tol], ssd[s], ebtypes[ebt], mgard_x::device_type::Serial);
              }
              result_gpu =
                  test<float>(shapes[sp].size(), original_data, shapes[sp], GPU,
                              tols[tol], ssf[s], ebtypes[ebt], dev_type);
              free(original_data);
            } else {
              size_t original_size = 1;
              for (mgard_x::DIM i = 0; i < shapes[sp].size(); i++)
                original_size *= shapes[sp][i];
              double *original_data =
                  (double *)malloc(original_size * sizeof(double));
              readfile(input_file, original_size * sizeof(double), false,
                       original_data);

              if (dev_type == mgard_x::device_type::Serial) {
                result_cpu =
                    test<double>(shapes[sp].size(), original_data, shapes[sp],
                                 CPU, told[tol], ssd[s], ebtypes[ebt], dev_type);
              } else {
                result_cpu =
                    test<double>(shapes[sp].size(), original_data, shapes[sp],
                                 GPU, told[tol], ssd[s], ebtypes[ebt], mgard_x::device_type::Serial);
              }

              result_gpu =
                  test<double>(shapes[sp].size(), original_data, shapes[sp],
                               GPU, told[tol], ssd[s], ebtypes[ebt], dev_type);
              free(original_data);
            }

            print_config(dtypes[dt], shapes[sp], tols[tol], ssd[s],
                         ebtypes[ebt]);

            std::cout << std::setw(12) << std::setprecision(4)
                      << std::scientific << result_cpu.actual_error;
            std::cout << std::setw(10) << std::setprecision(2)
                      << std::scientific << result_cpu.cr;

            // std::cout << std::endl;
            // print_config(input_file, dtypes[dt], shapes[sp], GPU, tols[tol],
            // ssd[s], ebtypes[ebt]);
            // if (std::abs(result_cpu.actual_error-result_gpu.actual_error) >
            // 1e-4) std::cout << ANSI_RED;
            if (std::abs(log10(result_cpu.actual_error) -
                         log10(result_gpu.actual_error)) > 1)
              std::cout << ANSI_RED;
            std::cout << std::setw(12) << std::setprecision(4)
                      << std::scientific << result_gpu.actual_error;
            std::cout << ANSI_RESET;
            std::cout << std::setw(10) << std::setprecision(2)
                      << std::scientific << result_gpu.cr;
            std::cout << std::endl;
          }
        }
      }
    }
  }
}
