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
#include <vector>

#include "compress_x.hpp"

using namespace std::chrono;

namespace mgard_x {

//! Enable autotuning
void BeginAutoTuning(enum device_type dev_type);

//! Disable autotuning
void EndAutoTuning(enum device_type dev_type);

} // namespace mgard_x

template <typename T>
int launch_compress(mgard_x::DIM D, enum mgard_x::data_type dtype,
                    std::vector<mgard_x::SIZE> shape,
                    enum mgard_x::device_type dev_type) {

  mgard_x::Config config;
  config.uniform_coord_mode = 0;
  config.dev_type = dev_type;
  config.lossless = mgard_x::lossless_type::Huffman;

  enum mgard_x::error_bound_type mode = mgard_x::error_bound_type::REL;
  double tol = 0.1;
  double s = 0;

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  T *original_data;
  size_t in_size = 0;
  in_size = original_size * sizeof(T);
  original_data = new T[original_size];
  srand(7117);
  for (size_t i = 0; i < original_size; i++)
    original_data[i] = rand() % 10 + 1;

  void *compressed_data = NULL;
  size_t compressed_size = 0;
  void *decompressed_data = NULL;
  mgard_x::compress(D, dtype, shape, tol, s, mode, original_data,
                    compressed_data, compressed_size, config, false);
  mgard_x::decompress(compressed_data, compressed_size, decompressed_data,
                      config, false);

  delete[](T *) original_data;
  free(compressed_data);
  free(decompressed_data);
  return 0;
}

void autotuning(enum mgard_x::device_type dev_type,
                std::vector<mgard_x::SIZE> shape) {
  if (dev_type == mgard_x::device_type::Serial) {
    std::cout << mgard_x::log::log_info
              << "Start autotuning MGARD-X::Serial.\n";
  } else if (dev_type == mgard_x::device_type::CUDA) {
    std::cout << mgard_x::log::log_info << "Start auto tuning MGARD-X::CUDA.\n";
  } else if (dev_type == mgard_x::device_type::HIP) {
    std::cout << mgard_x::log::log_info << "Start auto tuning MGARD-X::HIP.\n";
  } else if (dev_type == mgard_x::device_type::SYCL) {
    std::cout << mgard_x::log::log_info << "Start auto tuning MGARD-X::SYCL.\n";
  }
  mgard_x::BeginAutoTuning(dev_type);
  std::cout << mgard_x::log::log_info
            << "Tuning for single precision data ... ";
  launch_compress<float>(shape.size(), mgard_x::data_type::Float, shape,
                         dev_type);
  std::cout << "Done.\n";
  std::cout << mgard_x::log::log_info
            << "Tuning for double precision data ... ";
  launch_compress<double>(shape.size(), mgard_x::data_type::Double, shape,
                          dev_type);
  std::cout << "Done.\n";
  mgard_x::EndAutoTuning(dev_type);
  if (dev_type == mgard_x::device_type::Serial) {
    std::cout << mgard_x::log::log_info
              << "Done auto tuning MGARD-X::Serial.\n";
  } else if (dev_type == mgard_x::device_type::CUDA) {
    std::cout << mgard_x::log::log_info << "Done auto tuning MGARD-X::CUDA.\n";
  } else if (dev_type == mgard_x::device_type::HIP) {
    std::cout << mgard_x::log::log_info << "Done auto tuning MGARD-X::HIP.\n";
  } else if (dev_type == mgard_x::device_type::SYCL) {
    std::cout << mgard_x::log::log_info << "Done auto tuning MGARD-X::SYCL.\n";
  }
  std::cout << mgard_x::log::log_info
            << "Please recompile MGARD-X to make the auto tuning effective.\n";
}

void print_usage_message(std::string error) {
  if (error.compare("") != 0) {
    std::cout << mgard_x::log::log_err << error << std::endl;
  }
  printf("* Full automatic mode: run 'mgard-x-autotuner' without arguments\n\
* For a specific backend: run 'mgard-x-autotuner -d <auto|serial|cuda|hip|sycl> '\n\
* For a specific input size on a specific backend: run 'mgard-x-autotuner -d <auto|serial|cuda|hip> -n <ndim> [dim1] [dim2] ... [dimN]'\n");
  exit(0);
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
  for (int i = 0; i < argc; i++) {
    if (option.compare(std::string(argv[i])) == 0) {
      return std::string(argv[i + 1]);
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

int main(int argc, char *argv[]) {
  enum mgard_x::device_type dev_type;
  if (argc > 3) {
    mgard_x::DIM D = get_arg_int(argc, argv, "-n");
    std::vector<mgard_x::SIZE> shape = get_arg_dims(argc, argv, "-n");
    std::cout << mgard_x::log::log_info << "Auto tuning for shape: ";
    for (int i = 0; i < D; i++)
      std::cout << shape[i] << " ";
    std::cout << "\n";
    std::string dev = get_arg(argc, argv, "-d");
    if (dev.compare("serial") == 0) {
      dev_type = mgard_x::device_type::Serial;
      std::cout << mgard_x::log::log_info << "device type: Serial\n";
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
      std::cout << "wrong device type.\n";
      exit(-1);
    }
    autotuning(dev_type, shape);
  } else if (argc == 3) {
    std::vector<mgard_x::SIZE> shape({513, 513, 513});
    std::string dev = get_arg(argc, argv, "-d");
    if (dev.compare("serial") == 0) {
      dev_type = mgard_x::device_type::Serial;
      std::cout << mgard_x::log::log_info << "device type: Serial\n";
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
      std::cout << "wrong device type.\n";
      exit(-1);
    }
    autotuning(dev_type, shape);
  } else {
    std::cout << mgard_x::log::log_info << "Full automatic mode\n";
    std::vector<mgard_x::SIZE> shape({513, 513, 513});
#ifdef MGARD_ENABLE_SERIAL
    autotuning(mgard_x::device_type::Serial, shape);
#endif
#ifdef MGARD_ENABLE_CUDA
    autotuning(mgard_x::device_type::CUDA, shape);
#endif
#ifdef MGARD_ENABLE_HIP
    autotuning(mgard_x::device_type::HIP, shape);
#endif
#ifdef MGARD_ENABLE_SYCL
    autotuning(mgard_x::device_type::SYCL, shape);
#endif
  }
  return 0;
}
