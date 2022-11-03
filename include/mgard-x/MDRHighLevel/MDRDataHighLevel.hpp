/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_MDR_HIGH_LEVEL_DATA_HPP
#define MGARD_X_MDR_HIGH_LEVEL_DATA_HPP

#include "../RuntimeX/DataStructures/MDRMetaData.hpp"

namespace mgard_x {
namespace MDR {

class AggregatedMDRData {
public:
  void Initialize(SIZE num_subdomains) {
    this->num_subdomains = num_subdomains;
    data.resize(num_subdomains);
  }
  std::vector<std::vector<std::vector<Byte *>>> data;
  SIZE num_subdomains;
};

class AggregatedMDRMetaData {
public:
  void Initialize(SIZE num_subdomains) {
    this->num_subdomains = num_subdomains;
    metadata.resize(num_subdomains);
  }
  std::vector<Byte> header;
  std::vector<MDRMetaData> metadata;
  SIZE num_subdomains;
};

class ReconstructuredData {
public:
  std::vector<Byte *> data;
};

}
}

#endif