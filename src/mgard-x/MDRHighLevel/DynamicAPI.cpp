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

void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape, const void *original_data,
                AggregatedMDRMetaData &refactored_metadata,
                AggregatedMDRData &refactored_data,
                Config config, bool output_pre_allocated) {

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

void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape, const void *original_data,
                AggregatedMDRMetaData &refactored_metadata,
                AggregatedMDRData &refactored_data,
                Config config, std::vector<const Byte *> coords, bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    MDRefactor<SERIAL>(D, dtype, shape, original_data, refactored_metadata, 
                      refactored_data, config, coords, output_pre_allocated);
#else
    log::err("MDR-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    MDRefactor<OPENMP>(D, dtype, shape, original_data, refactored_metadata, 
                      refactored_data, config, coords, output_pre_allocated);
#else
    log::err("MDR-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    MDRefactor<CUDA>(D, dtype, shape, original_data, refactored_metadata, 
                      refactored_data, config, coords, output_pre_allocated);
#else
    log::err("MDR-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    MDRefactor<HIP>(D, dtype, shape, original_data, refactored_metadata, 
                      refactored_data, config, coords, output_pre_allocated);
#else
    log::err("MDR-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    MDRefactor<SYCL>(D, dtype, shape, original_data, refactored_metadata, 
                      refactored_data, config, coords, output_pre_allocated);
#else
    log::err("MDR-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
  }
}

void MDRequest(DIM D, data_type dtype, std::vector<SIZE> shape, 
               AggregatedMDRMetaData &refactored_metadata, double tol, double s,
               enum error_bound_type ebtype, Config config) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    MDRequest<SERIAL>(D, dtype, shape, refactored_metadata, tol, s, ebtype);
#else
    log::err("MDR-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    MDRequest<OPENMP>(D, dtype, shape, refactored_metadata, tol, s, ebtype);
#else
    log::err("MDR-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    MDRequest<CUDA>(D, dtype, shape, refactored_metadata, tol, s, ebtype);
#else
    log::err("MDR-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    MDRequest<HIP>(D, dtype, shape, refactored_metadata, tol, s, ebtype);
#else
    log::err("MDR-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    MDRequest<SYCL>(D, dtype, shape, refactored_metadata, tol, s, ebtype);
#else
    log::err("MDR-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
  }
}

void MDRconstruct(DIM D, data_type dtype, std::vector<SIZE> shape,
                  AggregatedMDRMetaData &refactored_metadata,
                  AggregatedMDRData &refactored_data,
                  ReconstructuredData &reconstructed_data, Config config,
                  bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    MDRconstruct<SERIAL>(D, dtype, shape, refactored_metadata, refactored_data, reconstructed_data,
                                         config, output_pre_allocated);
#else
    log::err("MDR-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    MDRconstruct<OPENMP>(D, dtype, shape, refactored_metadata, refactored_data, reconstructed_data,
                                         config, output_pre_allocated);
#else
    log::err("MDR-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    MDRconstruct<CUDA>(D, dtype, shape, refactored_metadata, refactored_data, reconstructed_data,
                                         config, output_pre_allocated);
#else
    log::err("MDR-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    MDRconstruct<HIP>(D, dtype, shape, refactored_metadata, refactored_data, reconstructed_data,
                                         config, output_pre_allocated);
#else
    log::err("MDR-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    MDRconstruct<SYCL>(D, dtype, shape, refactored_metadata, refactored_data, reconstructed_data,
                                         config, output_pre_allocated);
#else
    log::err("MDR-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
  }
}

}
} // namespace mgard_x
