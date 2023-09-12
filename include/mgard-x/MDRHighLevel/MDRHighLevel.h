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
#include "MDRDataHighLevel.hpp"

namespace mgard_x {
namespace MDR {

template <typename DeviceType>
void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape,
                const void *original_data,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated);

template <typename DeviceType>
void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape,
                const void *original_data, std::vector<const Byte *> coords,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated);

template <typename DeviceType>
void MDRequest(RefactoredMetadata &refactored_metadata);

template <typename DeviceType>
SIZE MDRMaxOutputDataSize(DIM D, data_type dtype, std::vector<SIZE> shape,
                          Config config);

template <typename DeviceType>
void MDReconstruct(RefactoredMetadata &refactored_metadata,
                   RefactoredData &refactored_data,
                   ReconstructedData &reconstructed_data, Config config,
                   bool output_pre_allocated, const void *original_data);
} // namespace MDR
} // namespace mgard_x

#endif