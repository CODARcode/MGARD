/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "compress_x.hpp"
#include "mgard-x/CompressionHighLevel/CompressionHighLevel.h"
#include "mgard-x/Config/Config.h"
#include "mgard-x/RuntimeX/AutoTuners/AutoTuner.h"
#include "mgard-x/RuntimeX/DataTypes.h"
#include "mgard-x/Utilities/Types.h"

namespace mgard_x {

enum device_type auto_detect_device() {
  enum device_type dev_type = device_type::NONE;
#if MGARD_ENABLE_SERIAL
  dev_type = device_type::SERIAL;
#endif
#if MGARD_ENABLE_OPENMP
  dev_type = device_type::OPENMP;
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
    log::err("MGARD-X was not built with any backend.");
    exit(-1);
  }
  return dev_type;
}

enum compress_status_type
compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol, double s,
         enum error_bound_type mode, const void *original_data,
         void *&compressed_data, size_t &compressed_size, Config config,
         bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    return compress<SERIAL>(D, dtype, shape, tol, s, mode, original_data,
                            compressed_data, compressed_size, config,
                            output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return compress<OPENMP>(D, dtype, shape, tol, s, mode, original_data,
                            compressed_data, compressed_size, config,
                            output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return compress<CUDA>(D, dtype, shape, tol, s, mode, original_data,
                          compressed_data, compressed_size, config,
                          output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return compress<HIP>(D, dtype, shape, tol, s, mode, original_data,
                         compressed_data, compressed_size, config,
                         output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return compress<SYCL>(D, dtype, shape, tol, s, mode, original_data,
                          compressed_data, compressed_size, config,
                          output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else {
    return compress_status_type::BackendNotAvailableFailure;
  }
}

enum compress_status_type
compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol, double s,
         enum error_bound_type mode, const void *original_data,
         void *&compressed_data, size_t &compressed_size,
         bool output_pre_allocated) {

  enum device_type dev_type = auto_detect_device();

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    return compress<SERIAL>(D, dtype, shape, tol, s, mode, original_data,
                            compressed_data, compressed_size,
                            output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return compress<OPENMP>(D, dtype, shape, tol, s, mode, original_data,
                            compressed_data, compressed_size,
                            output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return compress<CUDA>(D, dtype, shape, tol, s, mode, original_data,
                          compressed_data, compressed_size,
                          output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return compress<HIP>(D, dtype, shape, tol, s, mode, original_data,
                         compressed_data, compressed_size,
                         output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return compress<SYCL>(D, dtype, shape, tol, s, mode, original_data,
                          compressed_data, compressed_size,
                          output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else {
    return compress_status_type::BackendNotAvailableFailure;
  }
}

enum compress_status_type
compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol, double s,
         enum error_bound_type mode, const void *original_data,
         void *&compressed_data, size_t &compressed_size,
         std::vector<const Byte *> coords, Config config,
         bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    return compress<SERIAL>(D, dtype, shape, tol, s, mode, original_data,
                            compressed_data, compressed_size, coords, config,
                            output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return compress<OPENMP>(D, dtype, shape, tol, s, mode, original_data,
                            compressed_data, compressed_size, coords, config,
                            output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return compress<CUDA>(D, dtype, shape, tol, s, mode, original_data,
                          compressed_data, compressed_size, coords, config,
                          output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return compress<HIP>(D, dtype, shape, tol, s, mode, original_data,
                         compressed_data, compressed_size, coords, config,
                         output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return compress<SYCL>(D, dtype, shape, tol, s, mode, original_data,
                          compressed_data, compressed_size, coords, config,
                          output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else {
    return compress_status_type::BackendNotAvailableFailure;
  }
}

enum compress_status_type
compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol, double s,
         enum error_bound_type mode, const void *original_data,
         void *&compressed_data, size_t &compressed_size,
         std::vector<const Byte *> coords, bool output_pre_allocated) {

  enum device_type dev_type = auto_detect_device();

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    return compress<SERIAL>(D, dtype, shape, tol, s, mode, original_data,
                            compressed_data, compressed_size, coords,
                            output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return compress<OPENMP>(D, dtype, shape, tol, s, mode, original_data,
                            compressed_data, compressed_size, coords,
                            output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return compress<CUDA>(D, dtype, shape, tol, s, mode, original_data,
                          compressed_data, compressed_size, coords,
                          output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return compress<HIP>(D, dtype, shape, tol, s, mode, original_data,
                         compressed_data, compressed_size, coords,
                         output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return compress<SYCL>(D, dtype, shape, tol, s, mode, original_data,
                          compressed_data, compressed_size, coords,
                          output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else {
    return compress_status_type::BackendNotAvailableFailure;
  }
}

enum compress_status_type decompress(const void *compressed_data,
                                     size_t compressed_size,
                                     void *&decompressed_data, Config config,
                                     bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    return decompress<SERIAL>(compressed_data, compressed_size,
                              decompressed_data, config, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return decompress<OPENMP>(compressed_data, compressed_size,
                              decompressed_data, config, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return decompress<CUDA>(compressed_data, compressed_size, decompressed_data,
                            config, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return decompress<HIP>(compressed_data, compressed_size, decompressed_data,
                           config, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return decompress<SYCL>(compressed_data, compressed_size, decompressed_data,
                            config, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else {
    return compress_status_type::BackendNotAvailableFailure;
  }
}

enum compress_status_type decompress(const void *compressed_data,
                                     size_t compressed_size,
                                     void *&decompressed_data,
                                     bool output_pre_allocated) {

  enum device_type dev_type = auto_detect_device();

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    return decompress<SERIAL>(compressed_data, compressed_size,
                              decompressed_data, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return decompress<OPENMP>(compressed_data, compressed_size,
                              decompressed_data, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return decompress<CUDA>(compressed_data, compressed_size, decompressed_data,
                            output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return decompress<HIP>(compressed_data, compressed_size, decompressed_data,
                           output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return decompress<SYCL>(compressed_data, compressed_size, decompressed_data,
                            output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else {
    return compress_status_type::BackendNotAvailableFailure;
  }
}

enum compress_status_type
decompress(const void *compressed_data, size_t compressed_size,
           void *&decompressed_data, std::vector<mgard_x::SIZE> &shape,
           data_type &dtype, Config config, bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    return decompress<SERIAL>(compressed_data, compressed_size,
                              decompressed_data, dtype, shape, config,
                              output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return decompress<OPENMP>(compressed_data, compressed_size,
                              decompressed_data, dtype, shape, config,
                              output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return decompress<CUDA>(compressed_data, compressed_size, decompressed_data,
                            dtype, shape, config, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return decompress<HIP>(compressed_data, compressed_size, decompressed_data,
                           dtype, shape, config, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return decompress<SYCL>(compressed_data, compressed_size, decompressed_data,
                            dtype, shape, config, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else {
    return compress_status_type::BackendNotAvailableFailure;
  }
}

enum compress_status_type
decompress(const void *compressed_data, size_t compressed_size,
           void *&decompressed_data, std::vector<mgard_x::SIZE> &shape,
           data_type &dtype, bool output_pre_allocated) {

  enum device_type dev_type = auto_detect_device();

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    return decompress<SERIAL>(compressed_data, compressed_size,
                              decompressed_data, dtype, shape,
                              output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  }
  if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return decompress<OPENMP>(compressed_data, compressed_size,
                              decompressed_data, dtype, shape,
                              output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return decompress<CUDA>(compressed_data, compressed_size, decompressed_data,
                            dtype, shape, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return decompress<HIP>(compressed_data, compressed_size, decompressed_data,
                           dtype, shape, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return decompress<SYCL>(compressed_data, compressed_size, decompressed_data,
                            dtype, shape, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else {
    return compress_status_type::BackendNotAvailableFailure;
  }
}

enum compress_status_type release_cache(Config config) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    return release_cache<SERIAL>();
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return release_cache<OPENMP>();
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return release_cache<CUDA>();
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return release_cache<HIP>();
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return release_cache<SYCL>();
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else {
    return compress_status_type::BackendNotAvailableFailure;
  }
}

void BeginAutoTuning(enum device_type dev_type) {

  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    BeginAutoTuning<SERIAL>();
#else
    log::err("MGARD-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    BeginAutoTuning<OPENMP>();
#else
    log::err("MGARD-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    BeginAutoTuning<CUDA>();
#else
    log::err("MGARD-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    BeginAutoTuning<HIP>();
#else
    log::err("MGARD-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    BeginAutoTuning<SYCL>();
#else
    log::err("MGARD-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
  }
}

void EndAutoTuning(enum device_type dev_type) {

  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    EndAutoTuning<SERIAL>();
#else
    log::err("MGARD-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    EndAutoTuning<OPENMP>();
#else
    log::err("MGARD-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    EndAutoTuning<CUDA>();
#else
    log::err("MGARD-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    EndAutoTuning<HIP>();
#else
    log::err("MGARD-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    EndAutoTuning<SYCL>();
#else
    log::err("MGARD-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
  }
}

void pin_memory(void *ptr, SIZE num_bytes, Config config) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    pin_memory<SERIAL>(ptr, num_bytes);
#else
    log::err("MGARD-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    pin_memory<OPENMP>(ptr, num_bytes);
#else
    log::err("MGARD-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    pin_memory<CUDA>(ptr, num_bytes);
#else
    log::err("MGARD-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    pin_memory<HIP>(ptr, num_bytes);
#else
    log::err("MGARD-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    pin_memory<SYCL>(ptr, num_bytes);
#else
    log::err("MGARD-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
  }
}

bool check_memory_pinned(void *ptr, Config config) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    return check_memory_pinned<SERIAL>(ptr);
#else
    log::err("MGARD-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return check_memory_pinned<OPENMP>(ptr);
#else
    log::err("MGARD-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return check_memory_pinned<CUDA>(ptr);
#else
    log::err("MGARD-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return check_memory_pinned<HIP>(ptr);
#else
    log::err("MGARD-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return check_memory_pinned<SYCL>(ptr);
#else
    log::err("MGARD-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
  }
}

void unpin_memory(void *ptr, Config config) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    unpin_memory<SERIAL>(ptr);
#else
    log::err("MGARD-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    unpin_memory<OPENMP>(ptr);
#else
    log::err("MGARD-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    unpin_memory<CUDA>(ptr);
#else
    log::err("MGARD-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    unpin_memory<HIP>(ptr);
#else
    log::err("MGARD-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    unpin_memory<SYCL>(ptr);
#else
    log::err("MGARD-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
  }
}

} // namespace mgard_x
