/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "../Utilities/Types.h"

#include "../Config/Config.h"
#include "../Hierarchy/Hierarchy.h"
#include "../RuntimeX/RuntimeX.h"

#ifndef MGARD_X_COMPRESSOR_CACHE_HPP
#define MGARD_X_COMPRESSOR_CACHE_HPP

namespace mgard_x {

template <DIM D, typename T, typename DeviceType, typename CompressorType>
class CompressorBundle {
public:
  using HierarchyType = typename CompressorType::HierarchyType;
  std::unordered_map<std::string, HierarchyType> *hierarchy_cache;
  CompressorType *compressor = nullptr;
  Array<D, T, DeviceType> *device_subdomain_buffer = nullptr;
  Array<1, Byte, DeviceType> *device_compressed_buffer = nullptr;

  bool initialized = false;

  std::string GenHierarchyKey(std::vector<SIZE> shape) {
    std::stringstream ss;
    for (int i = 0; i < shape.size(); i++) {
      ss << std::setfill('0') << std::setw(20) << shape[i] << std::fixed;
    }
    return ss.str();
  }

  bool InHierarchyCache(std::vector<SIZE> shape, bool uniform) {
    if (!initialized) {
      return false;
    } else if (!uniform) {
      return false;
    } else if (hierarchy_cache->find(GenHierarchyKey(shape)) ==
               hierarchy_cache->end()) {
      return false;
    } else if ((*hierarchy_cache)[GenHierarchyKey(shape)].data_structure() ==
               data_structure_type::Cartesian_Grid_Non_Uniform) {
      return false;
    } else {
      return true;
    }
  }

  void InsertHierarchyCache(HierarchyType hierarchy) {
    std::vector<SIZE> shape = hierarchy.level_shape(hierarchy.l_target());
    (*hierarchy_cache)[GenHierarchyKey(shape)] = hierarchy;
    std::stringstream ss;
    for (int i = 0; i < shape.size(); i++) {
      ss << shape[i] << " ";
    }
    log::info("Add hierarchy with shape ( " + ss.str() + ") into cache");
  }

  HierarchyType &GetHierarchyCache(std::vector<SIZE> shape) {
    std::stringstream ss;
    for (int i = 0; i < shape.size(); i++) {
      ss << shape[i] << " ";
    }
    log::info("Get hierarchy with shape ( " + ss.str() + ") from cache");
    return (*hierarchy_cache)[GenHierarchyKey(shape)];
  }

  void Release() {
    log::info("Releasing compressor cache");
    delete hierarchy_cache;
    delete compressor;
    delete[] device_subdomain_buffer;
    delete[] device_compressed_buffer;
    initialized = false;
  }

  void ClearHierarchyCache() {
    log::info("Clear hierarchy cache");
    if (hierarchy_cache) {
      delete hierarchy_cache;
    }
    hierarchy_cache = new std::unordered_map<std::string, HierarchyType>();
  }

  void Initialize() {
    log::info("Initializing compressor cache");
    hierarchy_cache = new std::unordered_map<std::string, HierarchyType>();
    compressor = new CompressorType();
    device_subdomain_buffer = new Array<D, T, DeviceType>[2];
    device_compressed_buffer = new Array<1, Byte, DeviceType>[2];
    initialized = true;
  }

  void SafeInitialize() {
    if (!initialized) {
      Initialize();
    }
  }

  void SafeRelease() {
    if (initialized) {
      Release();
    }
  }

  size_t CacheSize() {
    size_t size = 0;
    if (initialized) {
      for (auto &item : *hierarchy_cache) {
        HierarchyType &_hierarchy = item.second;
        size += _hierarchy.EstimateMemoryFootprint(
            _hierarchy.level_shape(_hierarchy.l_target()));
      }
      if (compressor->initialized) {
        size += CompressorType::EstimateMemoryFootprint(
            compressor->hierarchy->level_shape(
                compressor->hierarchy->l_target()),
            compressor->config);
        size += compressor->hierarchy->total_num_elems() * sizeof(T) * 4;
      }
    }
    return size;
  }
};

template <DIM D, typename T, typename DeviceType, typename CompressorType>
class CompressorCache {
public:
  static inline CompressorBundle<D, T, DeviceType, CompressorType> cache;
};

} // namespace mgard_x

#endif
