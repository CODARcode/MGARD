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

class RefactoredData {
public:
  void Initialize(SIZE num_subdomains) {
    this->num_subdomains = num_subdomains;
    data.resize(num_subdomains);
  }
  std::vector<std::vector<std::vector<Byte *>>> data;
  SIZE num_subdomains;
};

class RefactoredMetadata {
public:
  void Initialize(SIZE num_subdomains) {
    this->num_subdomains = num_subdomains;
    metadata.resize(num_subdomains);
  }
  std::vector<Byte> header;
  std::vector<MDRMetadata> metadata;
  SIZE num_subdomains;
};

class ReconstructedData {
public:
  std::vector<Byte *> data;
};

}
}

#endif