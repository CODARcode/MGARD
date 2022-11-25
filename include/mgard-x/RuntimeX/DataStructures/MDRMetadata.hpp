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

class MDRMetadata {
public:
  MDRMetadata() : num_levels(0), num_bitplanes(0) {}

  void Initialize(SIZE num_levels, SIZE num_bitplanes) {
    this->num_levels = num_levels;
    this->num_bitplanes = num_bitplanes;
    level_error_bounds.resize(num_levels);
    level_squared_errors.resize(num_levels);
    level_sizes.resize(num_levels);
    for (int i = 0; i < num_levels; i++) {
      level_squared_errors[i].resize(num_bitplanes + 1);
      level_sizes[i].resize(num_bitplanes);
    }
  }
  MDRMetadata(SIZE num_levels, SIZE num_bitplanes)
      : num_levels(num_levels), num_bitplanes(num_bitplanes) {
    Initialize(num_levels, num_bitplanes);
  }

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

  void PrintLevelSizes() {
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      for (int bitplane_idx = 0; bitplane_idx < num_bitplanes; bitplane_idx++) {
        std::cout << level_sizes[level_idx][bitplane_idx] << " ";
      }
      std::cout << "\n";
    }
  }

  void PrintStatus() {
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      printf("Request %d (%d more) bitplans from level %d\n",
             requested_level_num_bitplanes[level_idx],
             requested_level_num_bitplanes[level_idx] -
                 loaded_level_num_bitplanes[level_idx],
             level_idx);
    }
  }

  void DoneLoadingBitplans() {
    // TODO: load
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      loaded_level_num_bitplanes[level_idx] =
          requested_level_num_bitplanes[level_idx];
    }
  }

  void DoneReconstruct() {
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      prev_used_level_num_bitplanes[level_idx] =
          loaded_level_num_bitplanes[level_idx];
    }
  }

  SIZE MetadataSize() {
    SIZE metadata_size = 0;
    metadata_size += sizeof(SIZE) * 2;
    metadata_size += sizeof(T_error) * num_levels;
    metadata_size += sizeof(T_error) * num_levels * (num_bitplanes + 1);
    metadata_size += sizeof(SIZE) * num_levels * num_bitplanes;
    return metadata_size;
  }

  template <typename T> void Serialize(Byte *&ptr, T *data, SIZE bytes) {
    memcpy(ptr, (Byte *)data, bytes);
    ptr += bytes;
  }

  template <typename T> void Deserialize(Byte *&ptr, T *data, SIZE bytes) {
    memcpy((Byte *)data, ptr, bytes);
    ptr += bytes;
  }

  std::vector<Byte> Serialize() {
    std::vector<Byte> serialize_metadata(MetadataSize());
    Byte *ptr = serialize_metadata.data();
    Serialize(ptr, &num_levels, sizeof(SIZE));
    Serialize(ptr, &num_bitplanes, sizeof(SIZE));
    Serialize(ptr, level_error_bounds.data(), sizeof(T_error) * num_levels);
    for (int i = 0; i < num_levels; i++) {
      Serialize(ptr, level_squared_errors[i].data(),
                sizeof(T_error) * (num_bitplanes + 1));
    }
    for (int i = 0; i < num_levels; i++) {
      Serialize(ptr, level_sizes[i].data(), sizeof(SIZE) * (num_bitplanes));
    }
    return serialize_metadata;
  }

  void Deserialize(std::vector<Byte> serialize_metadata) {
    Byte *ptr = serialize_metadata.data();
    Deserialize(ptr, &num_levels, sizeof(SIZE));
    Deserialize(ptr, &num_bitplanes, sizeof(SIZE));
    Initialize(num_levels, num_bitplanes);
    Deserialize(ptr, level_error_bounds.data(), sizeof(T_error) * num_levels);
    for (int i = 0; i < num_levels; i++) {
      Deserialize(ptr, level_squared_errors[i].data(),
                  sizeof(T_error) * (num_bitplanes + 1));
    }
    for (int i = 0; i < num_levels; i++) {
      Deserialize(ptr, level_sizes[i].data(), sizeof(SIZE) * (num_bitplanes));
    }
  }

  SIZE num_levels;
  SIZE num_bitplanes;
};

} // namespace MDR
} // namespace mgard_x

#endif