/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "mgard-x/Config/Config.h"
#include "mgard-x/MDRHighLevel/MDRDataHighLevel.hpp"

namespace mgard_x {
namespace MDR {

enum device_type auto_detect_device();

void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape, const void *original_data,
                AggregatedMDRMetaData &refactored_metadata,
                AggregatedMDRData &refactored_data,
                Config config, bool output_pre_allocated);

void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape, const void *original_data,
                std::vector<const Byte *> coords, 
                AggregatedMDRMetaData &refactored_metadata,
                AggregatedMDRData &refactored_data,
                Config config, bool output_pre_allocated);

void MDRequest(AggregatedMDRMetaData &refactored_metadata, double tol, double s,
               enum error_bound_type ebtype, Config config);

void MDReconstruct(AggregatedMDRMetaData &refactored_metadata,
                  AggregatedMDRData &refactored_data,
                  ReconstructuredData &reconstructed_data, Config config,
                  bool output_pre_allocated);
}
}
