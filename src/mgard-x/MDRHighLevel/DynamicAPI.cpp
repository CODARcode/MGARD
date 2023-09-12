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

#include "mgard-x/Config/Config.h"
#include "mgard-x/RuntimeX/AutoTuners/AutoTuner.h"
#include "mgard-x/RuntimeX/DataTypes.h"
#include "mgard-x/Utilities/Types.h"

#include "mgard-x/MDRHighLevel/MDRHighLevel.h"

namespace mgard_x {
namespace MDR {
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
    log::err("MDR-X was not built with any backend.");
    exit(-1);
  }
  return dev_type;
}

void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape,
                const void *original_data,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    MDRefactor<SERIAL>(D, dtype, shape, original_data, refactored_metadata,
                       refactored_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    MDRefactor<OPENMP>(D, dtype, shape, original_data, refactored_metadata,
                       refactored_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    MDRefactor<CUDA>(D, dtype, shape, original_data, refactored_metadata,
                     refactored_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    MDRefactor<HIP>(D, dtype, shape, original_data, refactored_metadata,
                    refactored_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    MDRefactor<SYCL>(D, dtype, shape, original_data, refactored_metadata,
                     refactored_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
  }
}

void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape,
                const void *original_data, std::vector<const Byte *> coords,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    MDRefactor<SERIAL>(D, dtype, shape, original_data, coords,
                       refactored_metadata, refactored_data, config,
                       output_pre_allocated);
#else
    log::err("MDR-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    MDRefactor<OPENMP>(D, dtype, shape, original_data, coords,
                       refactored_metadata, refactored_data, config,
                       output_pre_allocated);
#else
    log::err("MDR-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    MDRefactor<CUDA>(D, dtype, shape, original_data, coords,
                     refactored_metadata, refactored_data, config,
                     output_pre_allocated);
#else
    log::err("MDR-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    MDRefactor<HIP>(D, dtype, shape, original_data, coords, refactored_metadata,
                    refactored_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    MDRefactor<SYCL>(D, dtype, shape, original_data, coords,
                     refactored_metadata, refactored_data, config,
                     output_pre_allocated);
#else
    log::err("MDR-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
  }
}

void MDRequest(RefactoredMetadata &refactored_metadata, Config config) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    MDRequest<SERIAL>(refactored_metadata);
#else
    log::err("MDR-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    MDRequest<OPENMP>(refactored_metadata);
#else
    log::err("MDR-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    MDRequest<CUDA>(refactored_metadata);
#else
    log::err("MDR-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    MDRequest<HIP>(refactored_metadata);
#else
    log::err("MDR-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    MDRequest<SYCL>(refactored_metadata);
#else
    log::err("MDR-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
  }
}

SIZE MDRMaxOutputDataSize(DIM D, data_type dtype, std::vector<SIZE> shape,
                          Config config) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    return MDRMaxOutputDataSize<SERIAL>(D, dtype, shape, config);
#else
    log::err("MDR-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return MDRMaxOutputDataSize<OPENMP>(D, dtype, shape, config);
#else
    log::err("MDR-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return MDRMaxOutputDataSize<CUDA>(D, dtype, shape, config);
#else
    log::err("MDR-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return MDRMaxOutputDataSize<HIP>(D, dtype, shape, config);
#else
    log::err("MDR-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return MDRMaxOutputDataSize<SYCL>(D, dtype, shape, config);
#else
    log::err("MDR-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
  }
}

void MDReconstruct(RefactoredMetadata &refactored_metadata,
                   RefactoredData &refactored_data,
                   ReconstructedData &reconstructed_data, Config config,
                   bool output_pre_allocated, const void *original_data) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    MDReconstruct<SERIAL>(refactored_metadata, refactored_data,
                          reconstructed_data, config, output_pre_allocated,
                          original_data);
#else
    log::err("MDR-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    MDReconstruct<OPENMP>(refactored_metadata, refactored_data,
                          reconstructed_data, config, output_pre_allocated,
                          original_data);
#else
    log::err("MDR-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    MDReconstruct<CUDA>(refactored_metadata, refactored_data,
                        reconstructed_data, config, output_pre_allocated,
                        original_data);
#else
    log::err("MDR-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    MDReconstruct<HIP>(refactored_metadata, refactored_data, reconstructed_data,
                       config, output_pre_allocated, original_data);
#else
    log::err("MDR-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    MDReconstruct<SYCL>(refactored_metadata, refactored_data,
                        reconstructed_data, config, output_pre_allocated,
                        original_data);
#else
    log::err("MDR-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
  }
}

} // namespace MDR
} // namespace mgard_x
