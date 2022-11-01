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

template <DIM D, typename T, typename DeviceType>
class MDRMetaData {
public:

  MDRMetaData(SIZE num_levels, SIZE num_bitplanes): num_levels(num_levels), num_bitplanes(num_bitplanes) {}

  using T_error = double;
  // Metadata
  std::vector<T> level_error_bounds;
  std::vector<std::vector<T_error>> level_squared_errors;
  std::vector<std::vector<SIZE>> level_sizes;

  // For progressive reconstruction
  std::vector<uint8_t> loaded_level_num_bitplanes;
  std::vector<uint8_t> requested_level_num_bitplanes;
  std::vector<uint8_t> prev_used_level_num_bitplanes;

  InitializeForReconstruction() {
    SIZE num_levels = level_error_bounds.size();
    loaded_level_num_bitplanes = std::vector<uint8_t>(num_levels, 0);
    requested_level_num_bitplanes = std::vector<uint8_t>(num_levels, 0);
    prev_used_level_num_bitplanes = std::vector<uint8_t>(num_levels, 0);
  }

  void PrintStatus() {
    SIZE num_levels = level_error_bounds.size();
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      printf("Request %d (%d more) bitplans from level %d\n",
              requested_level_num_bitplanes[level_idx],
              requested_level_num_bitplanes[level_idx]-loaded_level_num_bitplanes[level_idx],
              level_idx);
    }
  }

  void LoadBitplans() {
    // TODO: load
    SIZE num_levels = level_error_bounds.size();
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      loaded_level_num_bitplanes[level_idx] = requested_level_num_bitplanes[level_idx];
    }
  }

  void DoneReconstruct() {
    SIZE num_levels = level_error_bounds.size();
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      prev_used_level_num_bitplanes[level_idx] = loaded_level_num_bitplanes[level_idx];
    }
  }

  SIZE num_levels;
  SIZE num_bitplanes;

};

template <DIM D, typename T, typename DeviceType>
class MDRData {
public:

  MDRData(SIZE num_levels, SIZE num_bitplanes): num_levels(num_levels), num_bitplanes(num_bitplanes) {}
  // Data
  std::vector<std::vector<Array<1, Byte, DeviceType>>> compressed_bitplanes;

  void Resize(MDRMetaData<D, T, DeviceType> &mdr_metadata) {
    compressed_bitplanes.resize(mdr_metadata.num_levels);
    for (int level_idx = 0; level_idx < mdr_metadata.num_levels; level_idx++) {
      compressed_bitplanes[level_idx].resize(mdr_metadata.num_bitplanes);
      for (int bitplane_idx = 0; bitplane_idx < mdr_metadata.num_bitplanes; bitplane_idx++) {
        compressed_bitplanes[level_idx][bitplane_idx].resize(mdr_metadata.level_sizes[level_idx][bitplane_idx]);
      }
    }
  }

  void CopyFromAggregatedMDRData(MDRMetaData<D, T, DeviceType> &mdr_metadata, std::vector<Byte*> &aggregated_mdr_data, int queue_idx) {
    int linearized_idx = 0;
    for (int level_idx = 0; level_idx < mdr_metadata.num_levels; level_idx++) {
      for (int bitplane_idx = mdr_metadata.loaded_level_num_bitplanes[level_idx]; 
               bitplane_idx < mdr_metadata.requested_level_num_bitplanes[level_idx]; bitplane_idx++) {
        MemoryManager<DeviceType>::Copy1D(compressed_bitplanes[level_idx][bitplane_idx],
                                          aggregated_mdr_data[linearized_idx],
                                          mdr_metadata.level_sizes[level_idx][bitplane_idx], queue_idx);
        linearized_idx++;
      }
    }
  }

  void CopyToAggregatedMDRData(MDRMetaData<D, T, DeviceType> &mdr_metadata, std::vector<Byte*> &aggregated_mdr_data, int queue_idx) {
    int linearized_idx = 0;
    for (int level_idx = 0; level_idx < mdr_metadata.num_levels; level_idx++) {
      for (int bitplane_idx = 0; bitplane_idx < num_bitplanes; bitplane_idx++) {
        MemoryManager<DeviceType>::Copy1D(aggregated_mdr_data[linearized_idx],
                                          compressed_bitplanes[level_idx][bitplane_idx],
                                          mdr_metadata.level_sizes[level_idx][bitplane_idx], queue_idx);
        linearized_idx++;
      }
    }
  }

  void VerifyLoadedBitplans(MDRMetaData<D, T, DeviceType> &mdr_metadata) {
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

  SIZE num_levels;
  SIZE num_bitplanes;
};

template <DIM D, typename T, typename DeviceType>
class AggregatedMDRData {
  std::vector<std::vector<Byte *>> data;
};

template <DIM D, typename T, typename DeviceType>
class AggregatedMDRMetaData {
  std::vector<MDRMetaData<D, T, DeviceType>> metadata;
};

}
}


#endif