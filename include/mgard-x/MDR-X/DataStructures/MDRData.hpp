#ifndef MDR_X_MDR_DATA_HPP
#define MDR_X_MDR_DATA_HPP
namespace mgard_x {
namespace MDR {

template <DIM D, typename T_data, typename DeviceType>
class MDRData {
public:
  using T_error = double;
  // Metadata
  std::vector<T_data> level_error_bounds;
  std::vector<std::vector<T_error>> level_squared_errors;
  std::vector<std::vector<SIZE>> level_sizes;

  // Data
  std::vector<std::vector<Array<1, Byte, DeviceType>>> compressed_bitplanes;

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

  void VerifyLoadedBitplans() {
    // TODO: load
    SIZE num_levels = level_error_bounds.size();
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      for (int bitplane_idx = prev_used_level_num_bitplanes[level_idx];
            bitplane_idx < loaded_level_num_bitplanes[level_idx]; bitplane_idx++) {
        if (!compressed_bitplanes[level_idx][bitplane_idx].hasDeviceAllocation() ||
            compressed_bitplanes[level_idx][bitplane_idx].shape(0) != level_sizes[level_idx][bitplane_idx]) {
          log::err("Bitplane verification failed.\n");
          exit(-1);
        }
      }
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

};

}
}


#endif