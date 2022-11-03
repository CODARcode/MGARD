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

// #include "../Hierarchy/Hierarchy.h"
// #include "../RuntimeX/RuntimeXPublic.h"
// #include "Metadata.hpp"

#ifndef MGARD_X_MDR_HIGH_LEVEL_API_H
#define MGARD_X_MDR_HIGH_LEVEL_API_H

#include "../Config/Config.h"
#include "../Hierarchy/Hierarchy.h"
#include "../RuntimeX/RuntimeX.h"
#include "../RuntimeX/DataStructures/MDRMetaData.hpp"

namespace mgard_x {
namespace MDR {

class AggregatedMDRData {
public:
  std::vector<std::vector<Byte *>> data;
};

class AggregatedMDRMetaData {
public:
  Byte * header;
  std::vector<MDRMetaData> metadata;
};

class ReconstructuredData {
public:
  std::vector<Byte *> data;
};

template <typename DeviceType>
void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape, const void *original_data,
          AggregatedMDRMetaData &refactored_metadata,
          AggregatedMDRData &refactored_data,
          Config config, bool output_pre_allocated);

template <typename DeviceType>
void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape, const void *original_data,
          AggregatedMDRMetaData &refactored_metadata,
          AggregatedMDRData &refactored_data,
          Config config, std::vector<const Byte *> coords, bool output_pre_allocated);

template <typename DeviceType>
void MDRequest(AggregatedMDRMetaData &refactored_metadata, double tol, double s,
               enum error_bound_type ebtype);

template <typename DeviceType>
void MDRconstruct(AggregatedMDRMetaData &refactored_metadata,
                  AggregatedMDRData &refactored_data,
                  ReconstructuredData &reconstructed_data, Config config,
                  bool output_pre_allocated);
}
}

#endif