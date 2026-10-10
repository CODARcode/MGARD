/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MDR_X_MDR_METADATA_HPP
#define MDR_X_MDR_METADATA_HPP

#include <cstring>

namespace mgard_x_160 {
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
    level_num_elems.resize(num_levels);
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
  SIZE num_levels;
  SIZE num_bitplanes;
  std::vector<T_error> level_error_bounds;
  std::vector<std::vector<T_error>> level_squared_errors;
  std::vector<std::vector<SIZE>> level_sizes;
  std::vector<SIZE> level_num_elems;
  bool segmented = false;
  bool corresponding_error_return = false;
  size_t retrieved_size = 0;

  // For progressive reconstruction
  T_error loaded_tol, loaded_s;
  T_error requested_tol, requested_s;
  T_error prev_tol, prev_s;
  T_error tau;
  uint32_t requested_size;
  size_t num_elements;
  double corresponding_error;
  std::vector<uint8_t> loaded_level_num_bitplanes;
  std::vector<uint8_t> requested_level_num_bitplanes;
  std::vector<uint8_t> prev_used_level_num_bitplanes;

  void InitializeForReconstruction() {
    loaded_tol = 0, loaded_s = 0;
    requested_tol = 0, requested_s = 0;
    prev_tol = 0, prev_s = 0;
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

  uint32_t GetLoadedBitPlaneSizes() {
    uint32_t bitplanes_size = 0;
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      // std::cout << "level[" << level_idx << "]" << ", loaded bitplanes: " <<
      // (int)loaded_level_num_bitplanes[level_idx] <<  ":" << std::endl;
      for (int bitplane_idx = 0;
           bitplane_idx < loaded_level_num_bitplanes[level_idx];
           bitplane_idx++) {
        // std::cout << (int)level_sizes[level_idx][bitplane_idx] << " ";
        bitplanes_size += level_sizes[level_idx][bitplane_idx];
      }
      // std::cout << "\n";
    }
    return bitplanes_size;
  }

  void PrintStatus() {
    printf("Request size: %u, s: %f\n", requested_size, requested_s);
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      printf("Level %d bitplanes: used [%2d] loaded [%2d] requested [%2d]\n",
             level_idx, prev_used_level_num_bitplanes[level_idx],
             loaded_level_num_bitplanes[level_idx],
             requested_level_num_bitplanes[level_idx]);
    }
    // printf("level_num_elems: ");
    // for (int level_idx = 0; level_idx < num_levels; level_idx++) {
    //   printf("%llu ", level_num_elems[level_idx]);
    // }
    // printf("\n");
  }

  void DoneLoadingBitplans() {
    // TODO: load
    loaded_tol = requested_tol;
    loaded_s = requested_s;
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      loaded_level_num_bitplanes[level_idx] =
          requested_level_num_bitplanes[level_idx];
    }
  }

  void DoneReconstruct() {
    prev_tol = loaded_tol;
    prev_s = loaded_s;
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      prev_used_level_num_bitplanes[level_idx] =
          loaded_level_num_bitplanes[level_idx];
    }
  }

  int PrevFinalLevel() {
    int final_level = 0;
    for (int level_idx = num_levels - 1; level_idx >= 0; level_idx--) {
      SIZE num_bitplanes = prev_used_level_num_bitplanes[level_idx];
      if (num_bitplanes != 0) {
        final_level = level_idx;
        break;
      }
    }
    return final_level;
  }

  int CurrFinalLevel() {
    int final_level = 0;
    for (int level_idx = num_levels - 1; level_idx >= 0; level_idx--) {
      SIZE num_bitplanes = loaded_level_num_bitplanes[level_idx];
      if (num_bitplanes != 0) {
        final_level = level_idx;
        break;
      }
    }
    return final_level;
  }

  SIZE MetadataSize() {
    SIZE metadata_size = 0;
    metadata_size += sizeof(SIZE) * 2;
    metadata_size += sizeof(T_error) * num_levels;
    metadata_size += sizeof(T_error) * num_levels * (num_bitplanes + 1);
    metadata_size += sizeof(SIZE) * num_levels * num_bitplanes;
    metadata_size += sizeof(SIZE) * num_levels;
    return metadata_size;
  }

  template <typename T> void Serialize(Byte *&ptr, T *data, SIZE bytes) {
    std::memcpy(ptr, (Byte *)data, bytes);
    ptr += bytes;
  }

  template <typename T> void Deserialize(Byte *&ptr, T *data, SIZE bytes) {
    std::memcpy((Byte *)data, ptr, bytes);
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
    Serialize(ptr, level_num_elems.data(), sizeof(SIZE) * num_levels);
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
    Deserialize(ptr, level_num_elems.data(), sizeof(SIZE) * num_levels);
  }
};

} // namespace MDR
} // namespace mgard_x_160

#endif