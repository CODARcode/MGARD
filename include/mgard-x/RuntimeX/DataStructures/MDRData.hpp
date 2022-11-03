/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MDR_X_MDR_DATA_HPP
#define MDR_X_MDR_DATA_HPP

namespace mgard_x {
namespace MDR {

#include "MDRMetaData.hpp"

template <typename DeviceType>
class MDRData {
public:

  MDRData() {}
  // Data
  std::vector<std::vector<Array<1, Byte, DeviceType>>> compressed_bitplanes;

  void Resize(MDRMetaData &mdr_metadata) {
    compressed_bitplanes.resize(mdr_metadata.num_levels);
    for (int level_idx = 0; level_idx < mdr_metadata.num_levels; level_idx++) {
      compressed_bitplanes[level_idx].resize(mdr_metadata.num_bitplanes);
      for (int bitplane_idx = 0; bitplane_idx < mdr_metadata.num_bitplanes; bitplane_idx++) {
        compressed_bitplanes[level_idx][bitplane_idx].resize({mdr_metadata.level_sizes[level_idx][bitplane_idx]});
      }
    }
  }

  void CopyFromAggregatedMDRData(MDRMetaData &mdr_metadata, std::vector<Byte*> &aggregated_mdr_data, int queue_idx) {
    int linearized_idx = 0;
    for (int level_idx = 0; level_idx < mdr_metadata.num_levels; level_idx++) {
      for (int bitplane_idx = mdr_metadata.loaded_level_num_bitplanes[level_idx]; 
               bitplane_idx < mdr_metadata.requested_level_num_bitplanes[level_idx]; bitplane_idx++) {
        MemoryManager<DeviceType>::Copy1D(compressed_bitplanes[level_idx][bitplane_idx].data(),
                                          aggregated_mdr_data[linearized_idx],
                                          mdr_metadata.level_sizes[level_idx][bitplane_idx], queue_idx);
        linearized_idx++;
      }
    }
  }

  void CopyToAggregatedMDRData(MDRMetaData &mdr_metadata, std::vector<Byte*> &aggregated_mdr_data, int queue_idx) {
    int linearized_idx = 0;
    for (int level_idx = 0; level_idx < mdr_metadata.num_levels; level_idx++) {
      for (int bitplane_idx = 0; bitplane_idx < mdr_metadata.num_bitplanes; bitplane_idx++) {
        MemoryManager<DeviceType>::Copy1D(aggregated_mdr_data[linearized_idx],
                                          compressed_bitplanes[level_idx][bitplane_idx].data(),
                                          mdr_metadata.level_sizes[level_idx][bitplane_idx], queue_idx);
        linearized_idx++;
      }
    }
  }

  void VerifyLoadedBitplans(MDRMetaData &mdr_metadata) {
    // TODO: load
    for (int level_idx = 0; level_idx < mdr_metadata.num_levels; level_idx++) {
      for (int bitplane_idx = mdr_metadata.prev_used_level_num_bitplanes[level_idx];
            bitplane_idx < mdr_metadata.loaded_level_num_bitplanes[level_idx]; bitplane_idx++) {
        if (!compressed_bitplanes[level_idx][bitplane_idx].hasDeviceAllocation() ||
            compressed_bitplanes[level_idx][bitplane_idx].shape(0) != mdr_metadata.level_sizes[level_idx][bitplane_idx]) {
          log::err("Bitplane verification failed.\n");
          exit(-1);
        }
      }
    }
  }
};

}
}


#endif