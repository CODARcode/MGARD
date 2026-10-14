/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */
#ifndef MGARD_X_MDR_X_REFACTOR_CACHE_HPP
#define MGARD_X_MDR_X_REFACTOR_CACHE_HPP

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace mgard_x {
namespace MDR {

template <DIM D, typename T, typename DeviceType, typename RefactorType>
class RefactorBundle {
public:
  using HierarchyType = typename RefactorType::HierarchyType;
  std::unordered_map<std::string, HierarchyType> *hierarchy_cache;
  RefactorType *refactor = nullptr;
  Array<D, T, DeviceType> *device_subdomain_buffer = nullptr;
  MDRData<DeviceType> *mdr_data = nullptr;

  bool initialized = false;

  // A hierarchy is determined by its shape together with the cap on its depth
  // (Config::max_larget_level): the same shape under a different cap has a
  // different number of levels. Both are part of the key; keying on the shape
  // alone let a decompress (or a later compress) silently reuse a hierarchy
  // that was built for another cap.
  std::string GenHierarchyKey(std::vector<SIZE> shape, SIZE max_larget_level) {
    std::stringstream ss;
    for (int i = 0; i < shape.size(); i++) {
      ss << std::setfill('0') << std::setw(20) << shape[i] << std::fixed;
    }
    ss << "|" << max_larget_level;
    return ss.str();
  }

  bool InHierarchyCache(std::vector<SIZE> shape, bool uniform,
                        SIZE max_larget_level) {
    if (!initialized) {
      return false;
    } else if (!uniform) {
      return false;
    } else if (hierarchy_cache->find(GenHierarchyKey(
                   shape, max_larget_level)) == hierarchy_cache->end()) {
      return false;
    } else if ((*hierarchy_cache)[GenHierarchyKey(shape, max_larget_level)]
                   .data_structure() ==
               data_structure_type::Cartesian_Grid_Non_Uniform) {
      return false;
    } else {
      return true;
    }
  }

  void InsertHierarchyCache(HierarchyType hierarchy, SIZE max_larget_level) {
    std::vector<SIZE> shape = hierarchy.level_shape(hierarchy.l_target());
    (*hierarchy_cache)[GenHierarchyKey(shape, max_larget_level)] = hierarchy;
    std::stringstream ss;
    for (int i = 0; i < shape.size(); i++) {
      ss << shape[i] << " ";
    }
    log::info("Add hierarchy with shape ( " + ss.str() + ") and max level " +
              std::to_string(max_larget_level) + " into cache");
  }

  HierarchyType &GetHierarchyCache(std::vector<SIZE> shape,
                                   SIZE max_larget_level) {
    std::stringstream ss;
    for (int i = 0; i < shape.size(); i++) {
      ss << shape[i] << " ";
    }
    log::info("Get hierarchy with shape ( " + ss.str() + ") and max level " +
              std::to_string(max_larget_level) + " from cache");
    return (*hierarchy_cache)[GenHierarchyKey(shape, max_larget_level)];
  }

  void Release() {
    log::info("Releasing refactor cache");
    delete hierarchy_cache;
    delete refactor;
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

  void Initialize(int num_buffers = 2) {
    log::info("Initializing refactor cache");
    hierarchy_cache = new std::unordered_map<std::string, HierarchyType>();
    refactor = new RefactorType();
    device_subdomain_buffer = new Array<D, T, DeviceType>[num_buffers];
    mdr_data = new MDRData<DeviceType>[num_buffers];
    initialized = true;
  }

  void SafeInitialize(int num_buffers = 2) {
    if (!initialized) {
      Initialize(num_buffers);
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
      if (refactor->initialized) {
        size += RefactorType::EstimateMemoryFootprint(
            refactor->hierarchy->level_shape(refactor->hierarchy->l_target()),
            refactor->config);
        size += refactor->hierarchy->total_num_elems() * sizeof(T) * 4;
      }
    }
    return size;
  }
};

template <DIM D, typename T, typename DeviceType, typename RefactorType>
class RefactorCache {
public:
  static inline thread_local RefactorBundle<D, T, DeviceType, RefactorType>
      cache;
};

} // namespace MDR
} // namespace mgard_x

#endif
