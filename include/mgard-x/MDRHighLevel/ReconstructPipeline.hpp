/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_MDR_RECONSTRUCT_PIPELINE_HPP
#define MGARD_X_MDR_RECONSTRUCT_PIPELINE_HPP

namespace mgard_x {
namespace MDR {

template <DIM D, typename T, typename DeviceType, typename ReconstructorType>
void reconstruct_pipeline(
    DomainDecomposer<D, T, ReconstructorType, DeviceType> &domain_decomposer,
    Config &config, RefactoredMetadata &refactored_metadata,
    RefactoredData &refactored_data, ReconstructedData &reconstructed_data) {

  using Cache = ReconstructorCache<D, T, DeviceType, ReconstructorType>;
  using HierarchyType = typename ReconstructorType::HierarchyType;

  bool timing_pipeline = true;

  ReconstructorType &reconstructor = *Cache::cache.reconstructor;
  Array<D, T, DeviceType> *device_subdomain_buffer =
      Cache::cache.device_subdomain_buffer;
  MDRData<DeviceType> *mdr_data = Cache::cache.mdr_data;
  if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(0),
                                     domain_decomposer.uniform)) {
    Cache::cache.ClearHierarchyCache();
  }
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
  mdr_data[0].Resize(reconstructor, hierarchy, 0);
  mdr_data[1].Resize(reconstructor, hierarchy, 0);
  mdr_data[2].Resize(reconstructor, hierarchy, 0);
  device_subdomain_buffer[0].resize(largest_shape, 0);
  device_subdomain_buffer[1].resize(largest_shape, 0);
  device_subdomain_buffer[2].resize(largest_shape, 0);

  Timer timer_series;
  if (timing_pipeline)
    timer_series.start();
  // Prefetch the first subdomain
  int current_buffer = 0;
  int current_queue = 0;
  mdr_data[current_buffer].Resize(
      reconstructor, refactored_metadata.metadata[0], current_queue);
  device_subdomain_buffer[current_buffer].resize(
      domain_decomposer.subdomain_shape(0), current_queue);
  mdr_data[current_buffer].CopyFromRefactoredData(
      refactored_metadata.metadata[0], refactored_data.data[0], current_queue);
  mdr_data[current_buffer].CopyFromRefactoredSigns(
      refactored_metadata.metadata[0], refactored_data.level_signs[0],
      current_queue);
  // Load previously reconstructred data
  domain_decomposer.copy_subdomain(
      device_subdomain_buffer[current_buffer], 0,
      subdomain_copy_direction::OriginalToSubdomain, current_queue);

  SIZE total_size = 0;

  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {
    SIZE next_subdomain_id;
    int next_buffer = (current_buffer + 1) % 3;
    int next_queue = (current_queue + 1) % 3;
    HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(
        domain_decomposer.subdomain_shape(curr_subdomain_id));
    log::info("Adapt Refactor to hierarchy");
    reconstructor.Adapt(hierarchy, config, current_queue);
    total_size += hierarchy.total_num_elems() * sizeof(T);
    if (curr_subdomain_id + 1 < domain_decomposer.num_subdomains()) {
      // Prefetch the next subdomain
      next_subdomain_id = curr_subdomain_id + 1;
      mdr_data[next_buffer].Resize(
          reconstructor, refactored_metadata.metadata[next_subdomain_id],
          next_queue);
      device_subdomain_buffer[next_buffer].resize(
          domain_decomposer.subdomain_shape(next_subdomain_id), next_queue);

      mdr_data[next_buffer].CopyFromRefactoredData(
          refactored_metadata.metadata[next_subdomain_id],
          refactored_data.data[next_subdomain_id], next_queue);
      // Copy signs
      mdr_data[next_buffer].CopyFromRefactoredSigns(
          refactored_metadata.metadata[next_subdomain_id],
          refactored_data.level_signs[next_subdomain_id], next_queue);
      // Load previously reconstructred data
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[next_buffer], next_subdomain_id,
          subdomain_copy_direction::OriginalToSubdomain, next_queue);
    }

    std::stringstream ss;
    for (DIM d = 0; d < D; d++) {
      ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
    }
    log::info("Reconstruct subdomain " + std::to_string(curr_subdomain_id) +
              " with shape: " + ss.str());

    reconstructor.LoadMetadata(refactored_metadata.metadata[curr_subdomain_id],
                               mdr_data[current_buffer], current_queue);
    reconstructor.Decompress(refactored_metadata.metadata[curr_subdomain_id],
                             mdr_data[current_buffer], current_queue);

    if (curr_subdomain_id > 0) {
      // We delay D2H since since it can delay the D2H in lossless decompession
      // and dequantization
      int previous_buffer = std::abs((current_buffer + 3 - 1) % 3);
      int previous_queue = std::abs((current_queue + 3 - 1) % 3);
      SIZE prev_subdomain_id = curr_subdomain_id - 1;
      // Update level signs for future progressive reconstruction
      mdr_data[previous_buffer].CopyToRefactoredSigns(
          refactored_metadata.metadata[prev_subdomain_id],
          refactored_data.level_signs[prev_subdomain_id], previous_queue);

      // Update reconstructed data
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[previous_buffer], prev_subdomain_id,
          subdomain_copy_direction::SubdomainToOriginal, previous_queue);
    }

    // Reconstruct
    reconstructor.ProgressiveReconstruct(
        refactored_metadata.metadata[curr_subdomain_id],
        mdr_data[current_buffer], config.mdr_adaptive_resolution,
        device_subdomain_buffer[current_buffer], current_queue);

    // Need to ensure reconstruction is complete before next reconstruction
    DeviceRuntime<DeviceType>::SyncQueue(current_queue);

    // // Update level signs for future progressive reconstruction
    // mdr_data[current_buffer].CopyToRefactoredSigns(
    //     refactored_metadata.metadata[curr_subdomain_id],
    //     refactored_data.level_signs[curr_subdomain_id], current_queue);

    // // Update reconstructed data
    // domain_decomposer.copy_subdomain(
    //     device_subdomain_buffer[current_buffer], curr_subdomain_id,
    //     subdomain_copy_direction::SubdomainToOriginal, current_queue);

    if (config.mdr_adaptive_resolution) {
      reconstructed_data.shape[curr_subdomain_id] =
          device_subdomain_buffer[current_buffer].shape();
      reconstructed_data.offset[curr_subdomain_id] =
          domain_decomposer.subdomain_start(curr_subdomain_id);
    }
    current_buffer = next_buffer;
    current_queue = next_queue;
  }

  // Copy the last subdomain
  // (+ 3: in C++ (0 - 1) % 3 is -1, and abs(-1) is not the previous index.)
  int previous_buffer = (current_buffer + 3 - 1) % 3;
  int previous_queue = (current_queue + 3 - 1) % 3;
  SIZE prev_subdomain_id = domain_decomposer.num_subdomains() - 1;
  // Update level signs for future progressive reconstruction
  mdr_data[previous_buffer].CopyToRefactoredSigns(
      refactored_metadata.metadata[prev_subdomain_id],
      refactored_data.level_signs[prev_subdomain_id], previous_queue);

  // Update reconstructed data
  domain_decomposer.copy_subdomain(
      device_subdomain_buffer[previous_buffer], prev_subdomain_id,
      subdomain_copy_direction::SubdomainToOriginal, previous_queue);

  DeviceRuntime<DeviceType>::SyncDevice();
  if (timing_pipeline) {
    timer_series.end();
    // log::csv("time.csv", timer_series.get());
    timer_series.print("Reconstruct pipeline", total_size, true);
    timer_series.clear();
  }
}

} // namespace MDR
} // namespace mgard_x
#endif