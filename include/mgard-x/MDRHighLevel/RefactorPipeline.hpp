/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_MDR_REFACTOR_PIPELINE_HPP
#define MGARD_X_MDR_REFACTOR_PIPELINE_HPP

namespace mgard_x {
namespace MDR {

template <DIM D, typename T, typename DeviceType, typename RefactorType>
void refactor_pipeline(
    DomainDecomposer<D, T, RefactorType, DeviceType> &domain_decomposer,
    Config &config, RefactoredMetadata &refactored_metadata,
    RefactoredData &refactored_data) {

  using Cache = RefactorCache<D, T, DeviceType, RefactorType>;
  using HierarchyType = typename RefactorType::HierarchyType;

  bool timing_pipeline = true;

  RefactorType &refactor = *Cache::cache.refactor;

  Array<D, T, DeviceType> *device_subdomain_buffer =
      Cache::cache.device_subdomain_buffer;
  MDRData<DeviceType> *mdr_data = Cache::cache.mdr_data;

  if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(0),
                                     domain_decomposer.uniform)) {
    Cache::cache.ClearHierarchyCache();
  }

  SIZE total_size = 0;

  for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
    if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(id),
                                       domain_decomposer.uniform)) {
      Cache::cache.InsertHierarchyCache(
          domain_decomposer.subdomain_hierarchy(id));
    }
  }

  // Size the reused buffers for the largest subdomain: they are not grown
  // when a later subdomain is bigger than the first.
  std::vector<SIZE> largest_shape =
      domain_decomposer.subdomain_shape(domain_decomposer.largest_subdomain());
  HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(largest_shape);

  log::info("Adjust device buffers");
  refactor.Adapt(hierarchy, config, 0);
  device_subdomain_buffer[0].resize(largest_shape, 0);
  device_subdomain_buffer[1].resize(largest_shape, 0);
  mdr_data[0].Resize(refactor, hierarchy, 0);
  mdr_data[1].Resize(refactor, hierarchy, 0);
  DeviceRuntime<DeviceType>::SyncDevice();

  Timer timer_series;
  if (timing_pipeline)
    timer_series.start();
  // Prefetch the first subdomain to one buffer
  int current_buffer = 0;
  int current_queue = 0;
  domain_decomposer.copy_subdomain(
      device_subdomain_buffer[current_buffer], 0,
      subdomain_copy_direction::OriginalToSubdomain, current_queue);

  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {
    SIZE next_subdomain_id;
    int next_buffer = (current_buffer + 1) % 2;
    int next_queue = (current_queue + 1) % 3;
    HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(
        domain_decomposer.subdomain_shape(curr_subdomain_id));
    log::info("Adapt Refactor to hierarchy");
    refactor.Adapt(hierarchy, config, current_queue);
    total_size += hierarchy.total_num_elems() * sizeof(T);
    // Prefetch the next subdomain
    if (curr_subdomain_id + 1 < domain_decomposer.num_subdomains()) {
      next_subdomain_id = curr_subdomain_id + 1;
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[next_buffer], next_subdomain_id,
          subdomain_copy_direction::OriginalToSubdomain, next_queue);
    }

    std::stringstream ss;
    for (DIM d = 0; d < D; d++) {
      ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
    }
    log::info("Refactoring subdomain " + std::to_string(curr_subdomain_id) +
              " with shape: " + ss.str());

    refactor.Refactor(device_subdomain_buffer[current_buffer],
                      refactored_metadata.metadata[curr_subdomain_id],
                      mdr_data[current_buffer], current_queue);
    refactor.Compress(refactored_metadata.metadata[curr_subdomain_id],
                      mdr_data[current_buffer], current_queue);
    refactor.StoreMetadata(refactored_metadata.metadata[curr_subdomain_id],
                           mdr_data[current_buffer], current_queue);
    mdr_data[current_buffer].CopyToRefactoredData(
        refactored_metadata.metadata[curr_subdomain_id],
        refactored_data.data[curr_subdomain_id],
        refactored_data.data_allocation_size[curr_subdomain_id], current_queue);

    current_buffer = next_buffer;
    current_queue = next_queue;
  }
  DeviceRuntime<DeviceType>::SyncDevice();
  if (timing_pipeline) {
    timer_series.end();
    // log::csv("time.csv", timer_series.get());
    timer_series.print("Refactor pipeline", total_size, true);
    timer_series.clear();
  }
}

} // namespace MDR
} // namespace mgard_x
#endif