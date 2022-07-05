/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "compress_x.hpp"
#include "mgard-x/CompressionHighLevel/CompressionHighLevel.h"
#include "mgard-x/CompressionHighLevel/Metadata.hpp"
#include "mgard-x/Hierarchy/Hierarchy.h"
#include "mgard-x/RuntimeX/RuntimeXPublic.h"

namespace mgard_x {

enum device_type auto_detect_device() {
  enum device_type dev_type = device_type::NONE;
#if MGARD_ENABLE_SERIAL
  dev_type = device_type::SERIAL;
#endif
#if MGARD_ENABLE_CUDA
  if (deviceAvailable<CUDA>()) {
    dev_type = device_type::CUDA;
  }
#endif
#if MGARD_ENABLE_HIP
  if (deviceAvailable<HIP>()) {
    dev_type = device_type::HIP;
  }
#endif
#if MGARD_ENABLE_SYCL
  if (deviceAvailable<SYCL>()) {
    dev_type = device_type::SYCL;
  }
#endif
  if (dev_type == device_type::NONE) {
    std::cout << log::log_err << "MGARD-X was not built with any backend.\n";
    exit(-1);
  }
  return dev_type;
}

void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size, Config config,
              bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    compress<SERIAL>(D, dtype, shape, tol, s, mode, original_data,
                     compressed_data, compressed_size, config,
                     output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SERIAL backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    compress<CUDA>(D, dtype, shape, tol, s, mode, original_data,
                   compressed_data, compressed_size, config,
                   output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    compress<HIP>(D, dtype, shape, tol, s, mode, original_data, compressed_data,
                  compressed_size, config, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    compress<SYCL>(D, dtype, shape, tol, s, mode, original_data,
                   compressed_data, compressed_size, config,
                   output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SYCL backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              bool output_pre_allocated) {

  enum device_type dev_type = auto_detect_device();

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    compress<SERIAL>(D, dtype, shape, tol, s, mode, original_data,
                     compressed_data, compressed_size, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SERIAL backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    compress<CUDA>(D, dtype, shape, tol, s, mode, original_data,
                   compressed_data, compressed_size, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    compress<HIP>(D, dtype, shape, tol, s, mode, original_data, compressed_data,
                  compressed_size, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    compress<SYCL>(D, dtype, shape, tol, s, mode, original_data,
                   compressed_data, compressed_size, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SYCL backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              std::vector<const Byte *> coords, Config config,
              bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    compress<SERIAL>(D, dtype, shape, tol, s, mode, original_data,
                     compressed_data, compressed_size, coords, config,
                     output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SERIAL backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    compress<CUDA>(D, dtype, shape, tol, s, mode, original_data,
                   compressed_data, compressed_size, coords, config,
                   output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    compress<HIP>(D, dtype, shape, tol, s, mode, original_data, compressed_data,
                  compressed_size, coords, config, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    compress<SYCL>(D, dtype, shape, tol, s, mode, original_data,
                   compressed_data, compressed_size, coords, config,
                   output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SYCL backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              std::vector<const Byte *> coords, bool output_pre_allocated) {

  enum device_type dev_type = auto_detect_device();

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    compress<SERIAL>(D, dtype, shape, tol, s, mode, original_data,
                     compressed_data, compressed_size, coords,
                     output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SERIAL backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    compress<CUDA>(D, dtype, shape, tol, s, mode, original_data,
                   compressed_data, compressed_size, coords,
                   output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    compress<HIP>(D, dtype, shape, tol, s, mode, original_data, compressed_data,
                  compressed_size, coords, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    compress<SYCL>(D, dtype, shape, tol, s, mode, original_data,
                   compressed_data, compressed_size, coords,
                   output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SYCL backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, Config config,
                bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    decompress<SERIAL>(compressed_data, compressed_size, decompressed_data,
                       config, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SERIAL backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    decompress<CUDA>(compressed_data, compressed_size, decompressed_data,
                     config, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    decompress<HIP>(compressed_data, compressed_size, decompressed_data, config,
                    output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    decompress<SYCL>(compressed_data, compressed_size, decompressed_data,
                     config, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SYCL backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, bool output_pre_allocated) {

  enum device_type dev_type = auto_detect_device();

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    decompress<SERIAL>(compressed_data, compressed_size, decompressed_data,
                       output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SERIAL backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    decompress<CUDA>(compressed_data, compressed_size, decompressed_data,
                     output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    decompress<HIP>(compressed_data, compressed_size, decompressed_data,
                    output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    decompress<SYCL>(compressed_data, compressed_size, decompressed_data,
                     output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SYCL backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, std::vector<mgard_x::SIZE> &shape,
                data_type &dtype, Config config, bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    decompress<SERIAL>(compressed_data, compressed_size, decompressed_data,
                       dtype, shape, config, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SERIAL backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    decompress<CUDA>(compressed_data, compressed_size, decompressed_data, dtype,
                     shape, config, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    decompress<HIP>(compressed_data, compressed_size, decompressed_data, dtype,
                    shape, config, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    decompress<SYCL>(compressed_data, compressed_size, decompressed_data, dtype,
                     shape, config, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SYCL backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, std::vector<mgard_x::SIZE> &shape,
                data_type &dtype, bool output_pre_allocated) {

  enum device_type dev_type = auto_detect_device();

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    decompress<SERIAL>(compressed_data, compressed_size, decompressed_data,
                       dtype, shape, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SERIAL backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    decompress<CUDA>(compressed_data, compressed_size, decompressed_data, dtype,
                     shape, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    decompress<HIP>(compressed_data, compressed_size, decompressed_data, dtype,
                    shape, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    decompress<SYCL>(compressed_data, compressed_size, decompressed_data, dtype,
                     shape, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with SYCL backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void BeginAutoTuning(enum device_type dev_type) {

  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    mgard_x::BeginAutoTuning<mgard_x::SERIAL>();
#else
    std::cout << log::log_err << "MGARD-X was not built with SERIAL backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    mgard_x::BeginAutoTuning<mgard_x::CUDA>();
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    mgard_x::BeginAutoTuning<mgard_x::HIP>();
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    mgard_x::BeginAutoTuning<mgard_x::SYCL>();
#else
    std::cout << log::log_err << "MGARD-X was not built with SYCL backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void EndAutoTuning(enum device_type dev_type) {

  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    mgard_x::EndAutoTuning<mgard_x::SERIAL>();
#else
    std::cout << log::log_err << "MGARD-X was not built with SERIAL backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    mgard_x::EndAutoTuning<mgard_x::CUDA>();
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    mgard_x::EndAutoTuning<mgard_x::HIP>();
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    mgard_x::EndAutoTuning<mgard_x::SYCL>();
#else
    std::cout << log::log_err << "MGARD-X was not built with SYCL backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

} // namespace mgard_x
