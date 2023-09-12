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

void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape,
                const void *original_data,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated);

void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape,
                const void *original_data, std::vector<const Byte *> coords,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated);

void MDRequest(RefactoredMetadata &refactored_metadata, Config config);

SIZE MDRMaxOutputDataSize(DIM D, data_type dtype, std::vector<SIZE> shape,
                          Config config);

void MDReconstruct(RefactoredMetadata &refactored_metadata,
                   RefactoredData &refactored_data,
                   ReconstructedData &reconstructed_data, Config config,
                   bool output_pre_allocated,
                   const void *original_data = nullptr);
} // namespace MDR
} // namespace mgard_x
