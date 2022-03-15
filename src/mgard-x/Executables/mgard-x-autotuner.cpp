/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
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

void autotuning(enum mgard_x::device_type dev_type) {
  if (dev_type == mgard_x::device_type::Serial) {
    std::cout << mgard_x::log::log_info
              << "Start autotuning MGARD-X::Serial.\n";
  } else if (dev_type == mgard_x::device_type::CUDA) {
    std::cout << mgard_x::log::log_info << "Start autotuning MGARD-X::CUDA.\n";
  } else if (dev_type == mgard_x::device_type::HIP) {
    std::cout << mgard_x::log::log_info << "Start autotuning MGARD-X::HIP.\n";
  }

  mgard_x::BeginAutoTuning(dev_type);

  std::vector<mgard_x::SIZE> shape = {512, 512, 512};
  std::cout << mgard_x::log::log_info << "Auto tuning 3D float.\n";
  launch_compress<float>(3, mgard_x::data_type::Float, shape, dev_type);
  std::cout << mgard_x::log::log_info << "Auto tuning 3D double.\n";
  launch_compress<double>(3, mgard_x::data_type::Double, shape, dev_type);

  std::vector<mgard_x::SIZE> shape4 = {8, 39, 16395, 39};
  std::cout << mgard_x::log::log_info << "Auto tuning 4D float.\n";
  launch_compress<float>(4, mgard_x::data_type::Float, shape4, dev_type);
  std::cout << mgard_x::log::log_info << "Auto tuning 4D double.\n";
  launch_compress<double>(4, mgard_x::data_type::Double, shape4, dev_type);

  if (dev_type == mgard_x::device_type::Serial) {
    std::cout << mgard_x::log::log_info << "Done autotuning MGARD-X::Serial.\n";
  } else if (dev_type == mgard_x::device_type::CUDA) {
    std::cout << mgard_x::log::log_info << "Done autotuning MGARD-X::CUDA.\n";
  } else if (dev_type == mgard_x::device_type::HIP) {
    std::cout << mgard_x::log::log_info << "Done autotuning MGARD-X::HIP.\n";
  }

  mgard_x::EndAutoTuning(dev_type);
}

std::string get_arg(int argc, char *argv[], std::string option) {
  for (int i = 0; i < argc; i++) {
    if (option.compare(std::string(argv[i])) == 0) {
      return std::string(argv[i + 1]);
    }
  }
  return std::string("");
}

int main(int argc, char *argv[]) {
  enum mgard_x::device_type dev_type;
  if (argc == 3) {
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
    } else {
      std::cout << "wrong device type.\n";
    }
    autotuning(dev_type);
  } else {
#ifdef MGARD_ENABLE_SERIAL
    autotuning(mgard_x::device_type::Serial);
#endif
#ifdef MGARD_ENABLE_CUDA
    autotuning(mgard_x::device_type::CUDA);
#endif
#ifdef MGARD_ENABLE_HIP
    autotuning(mgard_x::device_type::HIP);
#endif
  }
  return 0;
}
