/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */
#ifndef MGARD_X_MDR_X_RECONSTRUCTOR_CACHE_HPP
#define MGARD_X_MDR_X_RECONSTRUCTOR_CACHE_HPP

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace mgard_x {
namespace MDR {

template <DIM D, typename T, typename DeviceType, typename ReconstructorType>
class ReconstructorBundle {
public:
  using HierarchyType = typename ReconstructorType::HierarchyType;
  std::unordered_map<std::string, HierarchyType> *hierarchy_cache;
  ReconstructorType *reconstructor = nullptr;
  Array<D, T, DeviceType> *device_subdomain_buffer = nullptr;
  MDRData<DeviceType> *mdr_data = nullptr;

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
    log::info("Releasing reconstructor cache");
    delete hierarchy_cache;
    delete reconstructor;
    delete[] device_subdomain_buffer;
    delete[] mdr_data;
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
    log::info("Initializing reconstructor cache");
    hierarchy_cache = new std::unordered_map<std::string, HierarchyType>();
    reconstructor = new ReconstructorType();
    device_subdomain_buffer = new Array<D, T, DeviceType>[2];
    mdr_data = new MDRData<DeviceType>[2];
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
      if (reconstructor->initialized) {
        size += ReconstructorType::EstimateMemoryFootprint(
            reconstructor->hierarchy->level_shape(
                reconstructor->hierarchy->l_target()),
            reconstructor->config);
        size += reconstructor->hierarchy->total_num_elems() * sizeof(T) * 4;
      }
    }
    return size;
  }
};

template <DIM D, typename T, typename DeviceType, typename ReconstructorType>
class ReconstructorCache {
public:
  static inline thread_local ReconstructorBundle<D, T, DeviceType,
                                                 ReconstructorType>
      cache;
};

} // namespace MDR
} // namespace mgard_x

#endif
