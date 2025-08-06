/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_MDR_HIGH_LEVEL_DATA_HPP
#define MGARD_X_MDR_HIGH_LEVEL_DATA_HPP

#include "../RuntimeX/DataStructures/MDRMetadata.hpp"

namespace mgard_x {
namespace MDR {

class RefactoredMetadata {
public:
  void InitializeForRefactor(SIZE num_subdomains) {
    this->num_subdomains = num_subdomains;
    metadata.resize(num_subdomains);
  }

  void InitializeForReconstruction() {
    for (SIZE subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
      metadata[subdomain_id].InitializeForReconstruction();
    }
  }

  std::vector<Byte> header;
  std::vector<MDRMetadata> metadata;
  SIZE num_subdomains;

  template <typename T> void Serialize(Byte *&ptr, T *data, SIZE bytes) {
    memcpy(ptr, (Byte *)data, bytes);
    ptr += bytes;
  }

  template <typename T> void Deserialize(Byte *&ptr, T *data, SIZE bytes) {
    memcpy((Byte *)data, ptr, bytes);
    ptr += bytes;
  }

  std::vector<Byte> Serialize() {
    SIZE num_subdomains = metadata.size();
    size_t metadata_size = 0;
    metadata_size += sizeof(SIZE);
    for (SIZE subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
      metadata_size += sizeof(SIZE);
      metadata_size += metadata[subdomain_id].MetadataSize();
    }
    std::vector<Byte> serialize_metadata(metadata_size);
    Byte *ptr = serialize_metadata.data();
    Serialize(ptr, &num_subdomains, sizeof(SIZE));
    for (SIZE subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
      std::vector<Byte> serialized_MDRMetadata =
          metadata[subdomain_id].Serialize();
      SIZE serialized_MDRMetadata_size = serialized_MDRMetadata.size();
      Serialize(ptr, &serialized_MDRMetadata_size, sizeof(SIZE));
      Serialize(ptr, serialized_MDRMetadata.data(),
                serialized_MDRMetadata.size());
    }
    return serialize_metadata;
  }

  void Deserialize(std::vector<Byte> serialize_metadata) {
    Byte *ptr = serialize_metadata.data();
    Deserialize(ptr, &num_subdomains, sizeof(SIZE));
    this->num_subdomains = num_subdomains;
    metadata.resize(num_subdomains);
    for (SIZE subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
      SIZE serialized_MDRMetadata_size;
      Deserialize(ptr, &serialized_MDRMetadata_size, sizeof(SIZE));
      std::vector<Byte> serialized_MDRMetadata(serialized_MDRMetadata_size);
      Deserialize(ptr, serialized_MDRMetadata.data(),
                  serialized_MDRMetadata.size());
      metadata[subdomain_id].Deserialize(serialized_MDRMetadata);
    }
  }
};

class RefactoredData {
public:
  void InitializeForRefactor(SIZE num_subdomains) {
    this->num_subdomains = num_subdomains;
    data.resize(num_subdomains);
  }
  void InitializeForReconstruction(RefactoredMetadata &refactored_metadata) {
    int num_subdomains = refactored_metadata.metadata.size();
    this->num_subdomains = num_subdomains;
    data.resize(num_subdomains);
    level_signs.resize(num_subdomains);
    for (int subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
      MDRMetadata metadata = refactored_metadata.metadata[subdomain_id];
      int num_levels = metadata.level_sizes.size();
      data[subdomain_id].resize(num_levels);
      level_signs[subdomain_id].resize(num_levels);
      for (int level_idx = 0; level_idx < num_levels; level_idx++) {
        int num_bitplanes = metadata.level_sizes[level_idx].size();
        data[subdomain_id][level_idx].resize(num_bitplanes);
      }
    }
  }

  std::vector<std::vector<std::vector<Byte *>>> data;
  std::vector<std::vector<bool *>> level_signs;
  SIZE num_subdomains;
};

class ReconstructedData {
public:
  void Initialize(SIZE num_subdomains) {
    this->num_subdomains = num_subdomains;
    offset.resize(num_subdomains);
    shape.resize(num_subdomains);
    data.resize(num_subdomains);
    initialized = true;
  }
  bool IsInitialized() { return initialized; }
  std::vector<std::vector<SIZE>> offset;
  std::vector<std::vector<SIZE>> shape;
  std::vector<Byte *> data;
  SIZE num_subdomains;
  bool initialized = false;
};

} // namespace MDR
} // namespace mgard_x

#endif