/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MDR_X_MDR_METADATA_HPP
#define MDR_X_MDR_METADATA_HPP

namespace mgard_x {
namespace MDR {

class MDRMetaData {
public:
  MDRMetaData(): num_levels(0), num_bitplanes(0) {}
  MDRMetaData(SIZE num_levels, SIZE num_bitplanes): num_levels(num_levels), num_bitplanes(num_bitplanes) {}

  using T_error = double;
  // Metadata
  std::vector<T_error> level_error_bounds;
  std::vector<std::vector<T_error>> level_squared_errors;
  std::vector<std::vector<SIZE>> level_sizes;

  // For progressive reconstruction
  std::vector<uint8_t> loaded_level_num_bitplanes;
  std::vector<uint8_t> requested_level_num_bitplanes;
  std::vector<uint8_t> prev_used_level_num_bitplanes;

  void InitializeForReconstruction() {
    loaded_level_num_bitplanes = std::vector<uint8_t>(num_levels, 0);
    requested_level_num_bitplanes = std::vector<uint8_t>(num_levels, 0);
    prev_used_level_num_bitplanes = std::vector<uint8_t>(num_levels, 0);
  }

  void PrintStatus() {
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      printf("Request %d (%d more) bitplans from level %d\n",
              requested_level_num_bitplanes[level_idx],
              requested_level_num_bitplanes[level_idx]-loaded_level_num_bitplanes[level_idx],
              level_idx);
    }
  }

  void DoneLoadingBitplans() {
    // TODO: load
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      loaded_level_num_bitplanes[level_idx] = requested_level_num_bitplanes[level_idx];
    }
  }

  void DoneReconstruct() {
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      prev_used_level_num_bitplanes[level_idx] = loaded_level_num_bitplanes[level_idx];
    }
  }

  SIZE num_levels;
  SIZE num_bitplanes;

};

}
}


#endif