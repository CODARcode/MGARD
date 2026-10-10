/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "compress_x.hpp"
#include "mgard-x/CompressionHighLevel/CompressionHighLevel.h"
#include "mgard-x/CompressionHighLevel/DataCompatibility.h"
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
    throw std::runtime_error("MGARD-X was not built with any backend.");
  }
  return dev_type;
}

// Decompression entry shared by the public overloads below: streams written by
// MGARD 1.6.x or earlier go to the separate 1.6.0-compatible pipeline
// (src/mgard-x-160), everything else to this version's.
namespace {

void report_v160_disabled() {
  log::err("this data was compressed by MGARD 1.6.x or earlier; decompressing "
           "it requires MGARD built with -DMGARD_ENABLE_DATA_COMPATIBILITY=ON.");
}

template <typename DeviceType>
enum compress_status_type
decompress_routed(const void *compressed_data, size_t compressed_size,
                  void *&decompressed_data, Config config,
                  bool output_pre_allocated) {
  if (compressed_by_v160<DeviceType>(compressed_data)) {
#if MGARD_ENABLE_DATA_COMPATIBILITY
    return v160::decompress<DeviceType>(compressed_data, compressed_size,
                                        decompressed_data, config,
                                        output_pre_allocated);
#else
    report_v160_disabled();
    return compress_status_type::DataCompatibilityNotEnabledFailure;
#endif
  }
  return decompress<DeviceType>(compressed_data, compressed_size,
                                decompressed_data, config,
                                output_pre_allocated);
}

template <typename DeviceType>
enum compress_status_type
decompress_routed(const void *compressed_data, size_t compressed_size,
                  void *&decompressed_data, bool output_pre_allocated) {
  return decompress_routed<DeviceType>(compressed_data, compressed_size,
                                       decompressed_data, Config(),
                                       output_pre_allocated);
}

template <typename DeviceType>
enum compress_status_type
decompress_routed(const void *compressed_data, size_t compressed_size,
                  void *&decompressed_data, data_type &dtype,
                  std::vector<SIZE> &shape, Config config,
                  bool output_pre_allocated) {
  if (compressed_by_v160<DeviceType>(compressed_data)) {
#if MGARD_ENABLE_DATA_COMPATIBILITY
    return v160::decompress<DeviceType>(compressed_data, compressed_size,
                                        decompressed_data, dtype, shape,
                                        config, output_pre_allocated);
#else
    report_v160_disabled();
    return compress_status_type::DataCompatibilityNotEnabledFailure;
#endif
  }
  return decompress<DeviceType>(compressed_data, compressed_size,
                                decompressed_data, dtype, shape, config,
                                output_pre_allocated);
}

template <typename DeviceType>
enum compress_status_type
decompress_routed(const void *compressed_data, size_t compressed_size,
                  void *&decompressed_data, data_type &dtype,
                  std::vector<SIZE> &shape, bool output_pre_allocated) {
  return decompress_routed<DeviceType>(compressed_data, compressed_size,
                                       decompressed_data, dtype, shape,
                                       Config(), output_pre_allocated);
}

template <typename DeviceType>
enum compress_status_type release_cache_routed() {
#if MGARD_ENABLE_DATA_COMPATIBILITY
  v160::release_cache<DeviceType>();
#endif
  return release_cache<DeviceType>();
}

} // namespace

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
    return decompress_routed<SERIAL>(compressed_data, compressed_size,
                              decompressed_data, config, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return decompress_routed<OPENMP>(compressed_data, compressed_size,
                              decompressed_data, config, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return decompress_routed<CUDA>(compressed_data, compressed_size, decompressed_data,
                            config, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return decompress_routed<HIP>(compressed_data, compressed_size, decompressed_data,
                           config, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return decompress_routed<SYCL>(compressed_data, compressed_size, decompressed_data,
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
    return decompress_routed<SERIAL>(compressed_data, compressed_size,
                              decompressed_data, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return decompress_routed<OPENMP>(compressed_data, compressed_size,
                              decompressed_data, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return decompress_routed<CUDA>(compressed_data, compressed_size, decompressed_data,
                            output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return decompress_routed<HIP>(compressed_data, compressed_size, decompressed_data,
                           output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return decompress_routed<SYCL>(compressed_data, compressed_size, decompressed_data,
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
    return decompress_routed<SERIAL>(compressed_data, compressed_size,
                              decompressed_data, dtype, shape, config,
                              output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return decompress_routed<OPENMP>(compressed_data, compressed_size,
                              decompressed_data, dtype, shape, config,
                              output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return decompress_routed<CUDA>(compressed_data, compressed_size, decompressed_data,
                            dtype, shape, config, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return decompress_routed<HIP>(compressed_data, compressed_size, decompressed_data,
                           dtype, shape, config, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return decompress_routed<SYCL>(compressed_data, compressed_size, decompressed_data,
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
    return decompress_routed<SERIAL>(compressed_data, compressed_size,
                              decompressed_data, dtype, shape,
                              output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  }
  if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return decompress_routed<OPENMP>(compressed_data, compressed_size,
                              decompressed_data, dtype, shape,
                              output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return decompress_routed<CUDA>(compressed_data, compressed_size, decompressed_data,
                            dtype, shape, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return decompress_routed<HIP>(compressed_data, compressed_size, decompressed_data,
                           dtype, shape, output_pre_allocated);
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return decompress_routed<SYCL>(compressed_data, compressed_size, decompressed_data,
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
    return release_cache_routed<SERIAL>();
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return release_cache_routed<OPENMP>();
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return release_cache_routed<CUDA>();
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return release_cache_routed<HIP>();
#else
    return compress_status_type::BackendNotAvailableFailure;
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return release_cache_routed<SYCL>();
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
    throw std::runtime_error("MGARD-X was not built with SERIAL backend.");
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    BeginAutoTuning<OPENMP>();
#else
    throw std::runtime_error("MGARD-X was not built with OPENMP backend.");
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    BeginAutoTuning<CUDA>();
#else
    throw std::runtime_error("MGARD-X was not built with CUDA backend.");
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    BeginAutoTuning<HIP>();
#else
    throw std::runtime_error("MGARD-X was not built with HIP backend.");
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    BeginAutoTuning<SYCL>();
#else
    throw std::runtime_error("MGARD-X was not built with SYCL backend.");
#endif
  } else {
    throw std::runtime_error("Unsupported backend.");
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
    throw std::runtime_error("MGARD-X was not built with SERIAL backend.");
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    EndAutoTuning<OPENMP>();
#else
    throw std::runtime_error("MGARD-X was not built with OPENMP backend.");
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    EndAutoTuning<CUDA>();
#else
    throw std::runtime_error("MGARD-X was not built with CUDA backend.");
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    EndAutoTuning<HIP>();
#else
    throw std::runtime_error("MGARD-X was not built with HIP backend.");
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    EndAutoTuning<SYCL>();
#else
    throw std::runtime_error("MGARD-X was not built with SYCL backend.");
#endif
  } else {
    throw std::runtime_error("Unsupported backend.");
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
    throw std::runtime_error("MGARD-X was not built with SERIAL backend.");
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    pin_memory<OPENMP>(ptr, num_bytes);
#else
    throw std::runtime_error("MGARD-X was not built with OPENMP backend.");
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    pin_memory<CUDA>(ptr, num_bytes);
#else
    throw std::runtime_error("MGARD-X was not built with CUDA backend.");
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    pin_memory<HIP>(ptr, num_bytes);
#else
    throw std::runtime_error("MGARD-X was not built with HIP backend.");
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    pin_memory<SYCL>(ptr, num_bytes);
#else
    throw std::runtime_error("MGARD-X was not built with SYCL backend.");
#endif
  } else {
    throw std::runtime_error("Unsupported backend.");
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
    throw std::runtime_error("MGARD-X was not built with SERIAL backend.");
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return check_memory_pinned<OPENMP>(ptr);
#else
    throw std::runtime_error("MGARD-X was not built with OPENMP backend.");
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return check_memory_pinned<CUDA>(ptr);
#else
    throw std::runtime_error("MGARD-X was not built with CUDA backend.");
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return check_memory_pinned<HIP>(ptr);
#else
    throw std::runtime_error("MGARD-X was not built with HIP backend.");
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return check_memory_pinned<SYCL>(ptr);
#else
    throw std::runtime_error("MGARD-X was not built with SYCL backend.");
#endif
  } else {
    throw std::runtime_error("Unsupported backend.");
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
    throw std::runtime_error("MGARD-X was not built with SERIAL backend.");
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    unpin_memory<OPENMP>(ptr);
#else
    throw std::runtime_error("MGARD-X was not built with OPENMP backend.");
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    unpin_memory<CUDA>(ptr);
#else
    throw std::runtime_error("MGARD-X was not built with CUDA backend.");
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    unpin_memory<HIP>(ptr);
#else
    throw std::runtime_error("MGARD-X was not built with HIP backend.");
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    unpin_memory<SYCL>(ptr);
#else
    throw std::runtime_error("MGARD-X was not built with SYCL backend.");
#endif
  } else {
    throw std::runtime_error("Unsupported backend.");
  }
}

} // namespace mgard_x
