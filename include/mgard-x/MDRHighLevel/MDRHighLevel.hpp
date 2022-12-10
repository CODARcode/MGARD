/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_MDR_HIGH_LEVEL_API_HPP
#define MGARD_X_MDR_HIGH_LEVEL_API_HPP

#include "../Config/Config.h"
#include "../Hierarchy/Hierarchy.h"
#include "../RuntimeX/RuntimeX.h"

#include "mdr_x_lowlevel.hpp"

// #include "../CompressionLowLevel/Compressor.hpp"
#include "../DomainDecomposer/DomainDecomposer.hpp"
#include "../Metadata/Metadata.hpp"
#include "../UncertaintyCollector/UncertaintyCollector.hpp"
#include "MDRHighLevel.h"

#define BINSIZE 10

namespace mgard_x {
namespace MDR {

template <DIM D, typename T, typename DeviceType>
void generate_request(DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>,
                                       DeviceType> &domain_decomposer,
                      Config config, RefactoredMetadata &refactored_metadata) {
  for (int subdomain_id = 0; subdomain_id < domain_decomposer.num_subdomains();
       subdomain_id++) {
    Hierarchy<D, T, DeviceType> hierarchy =
        domain_decomposer.subdomain_hierarchy(subdomain_id);
    ComposedReconstructor<D, T, DeviceType> reconstructor(hierarchy, config);
    reconstructor.GenerateRequest(refactored_metadata.metadata[subdomain_id]);
  }
}

template <DIM D, typename T, typename DeviceType>
void refactor_subdomain(
    DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>
        &domain_decomposer,
    SIZE subdomain_id, Config &config, RefactoredMetadata &refactored_metadata,
    RefactoredData &refactored_data, int dev_id) {

  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();
  Array<D, T, DeviceType> device_subdomain_buffer;
  // Trigger the copy constructor to copy hierarchy to the current device
  Hierarchy<D, T, DeviceType> hierarchy =
      domain_decomposer.subdomain_hierarchy(subdomain_id);
  MDRMetadata mdr_metadata;
  MDRData<DeviceType> mdr_data(hierarchy.l_target() + 1,
                               config.total_num_bitplanes);

  domain_decomposer.copy_subdomain(
      device_subdomain_buffer, subdomain_id,
      subdomain_copy_direction::OriginalToSubdomain, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  ComposedRefactor<D, T, DeviceType> refactor(hierarchy, config);
  std::stringstream ss;
  for (DIM d = 0; d < D; d++) {
    ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
  }
  log::info("Refactoring subdomain " + std::to_string(subdomain_id) +
            " with shape: " + ss.str());
  refactor.Refactor(device_subdomain_buffer, mdr_metadata, mdr_data, 0);
  mdr_data.CopyToRefactoredData(mdr_metadata,
                                refactored_data.data[subdomain_id], 0);
  refactored_metadata.metadata[subdomain_id] = mdr_metadata;
  DeviceRuntime<DeviceType>::SyncQueue(0);
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Refactor single subdomain");
    timer_series.clear();
  }
  DeviceRuntime<DeviceType>::SyncDevice();
}

template <DIM D, typename T, typename DeviceType>
void refactor_subdomain_series(
    DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>
        &domain_decomposer,
    std::vector<SIZE> &subdomain_ids, Config &config,
    RefactoredMetadata &refactored_metadata, RefactoredData &refactored_data,
    int dev_id) {
  assert(subdomain_ids.size() > 0);
  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();
  Array<D, T, DeviceType> device_subdomain_buffer;

  for (SIZE i = 0; i < subdomain_ids.size(); i++) {
    SIZE subdomain_id = subdomain_ids[i];
    // Trigger the copy constructor to copy hierarchy to the current device
    Hierarchy<D, T, DeviceType> hierarchy =
        domain_decomposer.subdomain_hierarchy(subdomain_id);
    domain_decomposer.copy_subdomain(
        device_subdomain_buffer, subdomain_id,
        subdomain_copy_direction::OriginalToSubdomain, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    MDRMetadata mdr_metadata;
    MDRData<DeviceType> mdr_data(hierarchy.l_target() + 1,
                                 config.total_num_bitplanes);
    ComposedRefactor<D, T, DeviceType> refactor(hierarchy, config);
    std::stringstream ss;
    for (DIM d = 0; d < D; d++) {
      ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
    }
    log::info("Refactoring subdomain " + std::to_string(subdomain_id) +
              " with shape: " + ss.str());
    refactor.Refactor(device_subdomain_buffer, mdr_metadata, mdr_data, 0);
    mdr_data.CopyToRefactoredData(mdr_metadata,
                                  refactored_data.data[subdomain_id], 0);
    refactored_metadata.metadata[subdomain_id] = mdr_metadata;
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Refactor subdomain series");
    timer_series.clear();
  }
  DeviceRuntime<DeviceType>::SyncDevice();
}

template <DIM D, typename T, typename DeviceType>
void refactor_subdomain_series_w_prefetch(
    DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>
        &domain_decomposer,
    std::vector<SIZE> &subdomain_ids, Config &config,
    RefactoredMetadata &refactored_metadata, RefactoredData &refactored_data,
    int dev_id) {
  assert(subdomain_ids.size() > 0);
  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();
  // Objects
  Hierarchy<D, T, DeviceType> hierarchy =
      domain_decomposer.subdomain_hierarchy(subdomain_ids[0]);
  ComposedRefactor<D, T, DeviceType> refactor(hierarchy, config);
  // Input
  Array<D, T, DeviceType> device_subdomain_buffer[2];
  // Pre-allocate to the size of the first subdomain
  // Following subdomains should be no bigger than the first one
  // We shouldn't need to reallocate in the future
  device_subdomain_buffer[0].resize(
      domain_decomposer.subdomain_shape(subdomain_ids[0]));
  device_subdomain_buffer[1].resize(
      domain_decomposer.subdomain_shape(subdomain_ids[0]));
  // Output
  MDRData<DeviceType> mdr_data[2];
  mdr_data[0].Resize(hierarchy.l_target() + 1, config.total_num_bitplanes);
  mdr_data[1].Resize(hierarchy.l_target() + 1, config.total_num_bitplanes);

  // Prefetch the first subdomain to one buffer
  int current_buffer = 0;
  int current_queue = current_buffer;
  domain_decomposer.copy_subdomain(
      device_subdomain_buffer[current_buffer], subdomain_ids[0],
      subdomain_copy_direction::OriginalToSubdomain, current_queue);

  for (SIZE i = 0; i < subdomain_ids.size(); i++) {
    SIZE curr_subdomain_id = subdomain_ids[i];
    SIZE next_subdomain_id;
    int next_buffer = (current_buffer + 1) % 2;
    int next_queue = next_buffer;
    // Prefetch the next subdomain
    if (i + 1 < subdomain_ids.size()) {
      next_subdomain_id = subdomain_ids[i + 1];
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[next_buffer], next_subdomain_id,
          subdomain_copy_direction::OriginalToSubdomain, next_queue);
    }
    // Check if we can reuse the existing objects
    if (!hierarchy.can_reuse(
            domain_decomposer.subdomain_shape(curr_subdomain_id))) {
      hierarchy = domain_decomposer.subdomain_hierarchy(curr_subdomain_id);
      refactor = ComposedRefactor<D, T, DeviceType>(hierarchy, config);
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
    mdr_data[current_buffer].CopyToRefactoredData(
        refactored_metadata.metadata[curr_subdomain_id],
        refactored_data.data[curr_subdomain_id], current_queue);

    current_buffer = next_buffer;
    current_queue = next_queue;
  }
  DeviceRuntime<DeviceType>::SyncDevice();
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Refactor subdomain series with prefetch");
    timer_series.clear();
  }
}

template <DIM D, typename T, typename DeviceType>
void reconstruct_subdomain(
    DomainDecomposer<D, T, ComposedReconstructor<D, T, DeviceType>, DeviceType>
        &domain_decomposer,
    DomainDecomposer<D, T, ComposedReconstructor<D, T, DeviceType>, DeviceType>
        &domain_decomposer_org,
    SIZE subdomain_id, Config &config, RefactoredMetadata &refactored_metadata,
    RefactoredData &refactored_data, ReconstructedData &reconstructed_data,
    int dev_id) {

  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();
  Array<D, T, DeviceType> device_subdomain_buffer;
  Array<D, T, DeviceType> device_subdomain_buffer_org;
  Array<D + 1, int, DeviceType> device_uncertainty_dist;
  device_subdomain_buffer.resize(
      domain_decomposer.subdomain_shape(subdomain_id));
  device_subdomain_buffer_org.resize(
      domain_decomposer.subdomain_shape(subdomain_id));
  device_subdomain_buffer.memset(0, 0);

  Hierarchy<D, T, DeviceType> hierarchy =
      domain_decomposer.subdomain_hierarchy(subdomain_id);

  MDRMetadata mdr_metadata = refactored_metadata.metadata[subdomain_id];
  MDRData<DeviceType> mdr_data;
  mdr_data.Resize(mdr_metadata);

  mdr_data.CopyFromRefactoredData(mdr_metadata,
                                  refactored_data.data[subdomain_id], 0);

  ComposedReconstructor<D, T, DeviceType> reconstructor(hierarchy, config);

  std::stringstream ss;
  for (DIM d = 0; d < D; d++) {
    ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
  }
  log::info("Reconstruct subdomain " + std::to_string(subdomain_id) +
            " with shape: " + ss.str());
  device_subdomain_buffer.resize(hierarchy.level_shape(hierarchy.l_target()));
  reconstructor.ProgressiveReconstruct(mdr_metadata, mdr_data,
                                       config.mdr_adaptive_resolution,
                                       device_subdomain_buffer, 0);
  // PrintSubarray("reconstructed_data", SubArray(device_subdomain_buffer));
  domain_decomposer.copy_subdomain(
      device_subdomain_buffer, subdomain_id,
      subdomain_copy_direction::SubdomainToOriginal, 0);
  if (config.mdr_adaptive_resolution) {
    reconstructed_data.shape[subdomain_id] = device_subdomain_buffer.shape();
    reconstructed_data.offset[subdomain_id] =
        domain_decomposer.dim_subdomain_offset(subdomain_id);
  }
  if (config.collect_uncertainty) {
    if (config.mdr_adaptive_resolution) {
      // Interpolate reconstructed data to full resolution
      reconstructor.InterpolateToLevel(
          device_subdomain_buffer,
          refactored_metadata.metadata[subdomain_id].CurrFinalLevel(),
          hierarchy.l_target(), 0);
    }
    // PrintSubarray("reconstructed_data", SubArray(device_subdomain_buffer));
    // Get original data
    domain_decomposer_org.copy_subdomain(
        device_subdomain_buffer_org, subdomain_id,
        subdomain_copy_direction::OriginalToSubdomain, 0);
    // Uncertainty distribution
    std::vector<SIZE> uncertainty_dist_shape = hierarchy.level_shape(
        refactored_metadata.metadata[subdomain_id].CurrFinalLevel());
    uncertainty_dist_shape.insert(uncertainty_dist_shape.begin(), BINSIZE);
    device_uncertainty_dist.resize(uncertainty_dist_shape);
    UncertaintyCollector<D, T, DeviceType> uncertainty_collector(
        hierarchy, mdr_metadata.prev_tol, BINSIZE);
    uncertainty_collector.Collect(
        device_subdomain_buffer_org, device_subdomain_buffer,
        refactored_metadata.metadata[subdomain_id].CurrFinalLevel(),
        device_uncertainty_dist, 0);
  }
  DeviceRuntime<DeviceType>::SyncQueue(0);
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Reconstruct single subdomain");
    timer_series.clear();
  }
  DeviceRuntime<DeviceType>::SyncDevice();
}

template <DIM D, typename T, typename DeviceType>
void reconstruct_subdomain_series(
    DomainDecomposer<D, T, ComposedReconstructor<D, T, DeviceType>, DeviceType>
        &domain_decomposer,
    DomainDecomposer<D, T, ComposedReconstructor<D, T, DeviceType>, DeviceType>
        &domain_decomposer_org,
    std::vector<SIZE> &subdomain_ids, Config &config,
    RefactoredMetadata &refactored_metadata, RefactoredData &refactored_data,
    ReconstructedData &reconstructed_data, int dev_id) {
  assert(subdomain_ids.size() > 0);
  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();
  Array<D, T, DeviceType> device_subdomain_buffer;
  Array<D, T, DeviceType> device_subdomain_buffer_org;
  Array<D + 1, int, DeviceType> device_uncertainty_dist;

  for (SIZE i = 0; i < subdomain_ids.size(); i++) {
    SIZE subdomain_id = subdomain_ids[i];
    Hierarchy<D, T, DeviceType> hierarchy =
        domain_decomposer.subdomain_hierarchy(subdomain_id);

    MDRMetadata mdr_metadata = refactored_metadata.metadata[subdomain_id];
    MDRData<DeviceType> mdr_data;
    mdr_data.Resize(mdr_metadata);

    mdr_data.CopyFromRefactoredData(mdr_metadata,
                                    refactored_data.data[subdomain_id], 0);

    ComposedReconstructor<D, T, DeviceType> reconstructor(hierarchy, config);

    std::stringstream ss;
    for (DIM d = 0; d < D; d++) {
      ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
    }
    log::info("Reconstruct subdomain " + std::to_string(subdomain_id) +
              " with shape: " + ss.str());
    device_subdomain_buffer.resize(hierarchy.level_shape(hierarchy.l_target()));
    device_subdomain_buffer.memset(0, 0);
    // Load previously reconstructred data
    domain_decomposer.copy_subdomain(
        device_subdomain_buffer, subdomain_id,
        subdomain_copy_direction::OriginalToSubdomain, 0);
    // Reconstruct
    reconstructor.ProgressiveReconstruct(mdr_metadata, mdr_data,
                                         config.mdr_adaptive_resolution,
                                         device_subdomain_buffer, 0);
    // Update reconstructed data
    domain_decomposer.copy_subdomain(
        device_subdomain_buffer, subdomain_id,
        subdomain_copy_direction::SubdomainToOriginal, 0);
    if (config.mdr_adaptive_resolution) {
      reconstructed_data.shape[subdomain_id] = device_subdomain_buffer.shape();
      reconstructed_data.offset[subdomain_id] =
          domain_decomposer.dim_subdomain_offset(subdomain_id);
    }
    if (config.collect_uncertainty) {
      if (config.mdr_adaptive_resolution) {
        // Interpolate reconstructed data to full resolution
        reconstructor.InterpolateToLevel(
            device_subdomain_buffer,
            refactored_metadata.metadata[subdomain_id].CurrFinalLevel(),
            hierarchy.l_target(), 0);
      }
      // Get original data
      domain_decomposer_org.copy_subdomain(
          device_subdomain_buffer_org, subdomain_id,
          subdomain_copy_direction::OriginalToSubdomain, 0);
      // Uncertainty distribution
      std::vector<SIZE> uncertainty_dist_shape = hierarchy.level_shape(
          refactored_metadata.metadata[subdomain_id].CurrFinalLevel());
      uncertainty_dist_shape.insert(uncertainty_dist_shape.begin(), BINSIZE);
      device_uncertainty_dist.resize(uncertainty_dist_shape);
      UncertaintyCollector<D, T, DeviceType> uncertainty_collector(
          hierarchy, mdr_metadata.prev_tol, BINSIZE);
      uncertainty_collector.Collect(
          device_subdomain_buffer_org, device_subdomain_buffer,
          refactored_metadata.metadata[subdomain_id].CurrFinalLevel(),
          device_uncertainty_dist, 0);
    }
  }
  DeviceRuntime<DeviceType>::SyncQueue(0);
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Reconstruct subdomain series");
    timer_series.clear();
  }
  DeviceRuntime<DeviceType>::SyncDevice();
}

template <DIM D, typename T, typename DeviceType>
void reconstruct_subdomain_series_w_prefetch(
    DomainDecomposer<D, T, ComposedReconstructor<D, T, DeviceType>, DeviceType>
        &domain_decomposer,
    DomainDecomposer<D, T, ComposedReconstructor<D, T, DeviceType>, DeviceType>
        &domain_decomposer_org,
    std::vector<SIZE> &subdomain_ids, Config &config,
    RefactoredMetadata &refactored_metadata, RefactoredData &refactored_data,
    ReconstructedData &reconstructed_data, int dev_id) {
  assert(subdomain_ids.size() > 0);
  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();

  // Objects
  Hierarchy<D, T, DeviceType> hierarchy =
      domain_decomposer.subdomain_hierarchy(subdomain_ids[0]);
  ComposedReconstructor<D, T, DeviceType> reconstructor(hierarchy, config);
  // Input
  MDRData<DeviceType> mdr_data[2];
  mdr_data[0].Resize(refactored_metadata.metadata[subdomain_ids[0]]);
  mdr_data[1].Resize(refactored_metadata.metadata[subdomain_ids[0]]);
  // Output
  Array<D, T, DeviceType> device_subdomain_buffer[2];
  device_subdomain_buffer[0].resize(
      domain_decomposer.subdomain_shape(subdomain_ids[0]));
  device_subdomain_buffer[1].resize(
      domain_decomposer.subdomain_shape(subdomain_ids[0]));

  Array<D, T, DeviceType> device_subdomain_buffer_org[2];
  device_subdomain_buffer_org[0].resize(
      domain_decomposer.subdomain_shape(subdomain_ids[0]));
  device_subdomain_buffer_org[1].resize(
      domain_decomposer.subdomain_shape(subdomain_ids[1]));

  Array<D + 1, int, DeviceType> device_uncertainty_dist[2];

  // Prefetch the first subdomain
  int current_buffer = 0;
  int current_queue = current_buffer;
  mdr_data[current_buffer].CopyFromRefactoredData(
      refactored_metadata.metadata[subdomain_ids[0]],
      refactored_data.data[subdomain_ids[0]], current_queue);

  for (SIZE i = 0; i < subdomain_ids.size(); i++) {
    SIZE curr_subdomain_id = subdomain_ids[i];
    SIZE next_subdomain_id;
    int next_buffer = (current_buffer + 1) % 2;
    int next_queue = next_buffer;
    if (i + 1 < subdomain_ids.size()) {
      // Prefetch the next subdomain
      next_subdomain_id = subdomain_ids[i + 1];
      mdr_data[next_buffer].CopyFromRefactoredData(
          refactored_metadata.metadata[next_subdomain_id],
          refactored_data.data[next_subdomain_id], current_queue);
    }

    // Check if we can reuse the existing objects
    if (!hierarchy.can_reuse(
            domain_decomposer.subdomain_shape(curr_subdomain_id))) {
      hierarchy = domain_decomposer.subdomain_hierarchy(curr_subdomain_id);
      reconstructor =
          ComposedReconstructor<D, T, DeviceType>(hierarchy, config);
    }

    std::stringstream ss;
    for (DIM d = 0; d < D; d++) {
      ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
    }
    log::info("Reconstruct subdomain " + std::to_string(curr_subdomain_id) +
              " with shape: " + ss.str());
    device_subdomain_buffer[current_buffer].resize(
        hierarchy.level_shape(hierarchy.l_target()));
    // Load previously reconstructred data
    domain_decomposer.copy_subdomain(
        device_subdomain_buffer[current_buffer], curr_subdomain_id,
        subdomain_copy_direction::OriginalToSubdomain, current_queue);
    // Reconstruct
    reconstructor.ProgressiveReconstruct(
        refactored_metadata.metadata[curr_subdomain_id],
        mdr_data[current_buffer], config.mdr_adaptive_resolution,
        device_subdomain_buffer[current_buffer], current_queue);
    // Update reconstructed data
    domain_decomposer.copy_subdomain(
        device_subdomain_buffer[current_buffer], curr_subdomain_id,
        subdomain_copy_direction::SubdomainToOriginal, current_queue);
    if (config.mdr_adaptive_resolution) {
      reconstructed_data.shape[curr_subdomain_id] =
          device_subdomain_buffer[current_buffer].shape();
      reconstructed_data.offset[curr_subdomain_id] =
          domain_decomposer.dim_subdomain_offset(curr_subdomain_id);
    }
    if (config.collect_uncertainty) {
      if (config.mdr_adaptive_resolution) {
        // Interpolate reconstructed data to full resolution
        reconstructor.InterpolateToLevel(
            device_subdomain_buffer[current_buffer],
            refactored_metadata.metadata[curr_subdomain_id].CurrFinalLevel(),
            hierarchy.l_target(), current_queue);
      }
      // Get original data
      domain_decomposer_org.copy_subdomain(
          device_subdomain_buffer_org[current_buffer], curr_subdomain_id,
          subdomain_copy_direction::OriginalToSubdomain, current_queue);
      // Uncertainty distribution
      std::vector<SIZE> uncertainty_dist_shape = hierarchy.level_shape(
          refactored_metadata.metadata[curr_subdomain_id].CurrFinalLevel());
      uncertainty_dist_shape.insert(uncertainty_dist_shape.begin(), BINSIZE);
      device_uncertainty_dist[current_buffer].resize(uncertainty_dist_shape);
      UncertaintyCollector<D, T, DeviceType> uncertainty_collector(
          hierarchy, refactored_metadata.metadata[curr_subdomain_id].prev_tol,
          BINSIZE);
      uncertainty_collector.Collect(
          device_subdomain_buffer_org[current_buffer],
          device_subdomain_buffer[current_buffer],
          refactored_metadata.metadata[curr_subdomain_id].CurrFinalLevel(),
          device_uncertainty_dist[current_buffer], 0);
    }
    current_buffer = next_buffer;
    current_queue = next_queue;
  }
  DeviceRuntime<DeviceType>::SyncDevice();
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Reconstruct subdomain series with prefetch");
    timer_series.clear();
  }
}

template <typename DeviceType>
void load(Config &config, Metadata<DeviceType> &metadata) {
  config.domain_decomposition = metadata.ddtype;
  config.decomposition = metadata.decomposition;
  config.lossless = metadata.ltype;
  config.huff_dict_size = metadata.huff_dict_size;
  config.huff_block_size = metadata.huff_block_size;
  config.reorder = metadata.reorder;
  config.total_num_bitplanes = metadata.number_bitplanes;
}

template <DIM D, typename T, typename DeviceType>
void MDRefactor(std::vector<SIZE> shape, const void *original_data,
                bool uniform, std::vector<T *> coords,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated) {
  size_t total_num_elem = 1;
  for (int i = 0; i < D; i++)
    total_num_elem *= shape[i];
  config.apply();

  if (config.num_dev <= 0) {
    log::err("Number of device needs to be greater than 0.");
    exit(-1);
  }

  int adjusted_num_dev =
      std::min(DeviceRuntime<DeviceType>::GetDeviceCount(), config.num_dev);
  if (adjusted_num_dev != config.num_dev) {
    log::info("Using " + std::to_string(adjusted_num_dev) +
              " devices (adjusted from " + std::to_string(config.num_dev) +
              " devices)");
  } else {
    log::info("Using " + std::to_string(adjusted_num_dev) + " devices.");
  }

  Timer timer_total, timer_each;
  if (log::level & log::TIME)
    timer_total.start();

  DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>
      domain_decomposer;
  if (uniform) {
    domain_decomposer =
        DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>(
            shape, adjusted_num_dev, config);
  } else {
    domain_decomposer =
        DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>(
            shape, adjusted_num_dev, config, coords);
  }
  domain_decomposer.set_original_data((T *)original_data);

  if (domain_decomposer.domain_decomposed()) {
    MemoryManager<DeviceType>::ReduceMemoryFootprint = true;
  }

  if (log::level & log::TIME)
    timer_each.start();
  bool input_previously_pinned =
      !MemoryManager<DeviceType>::IsDevicePointer((void *)original_data) &&
      MemoryManager<DeviceType>::CheckHostRegister((void *)original_data);
  if (!input_previously_pinned) {
    MemoryManager<DeviceType>::HostRegister((void *)original_data,
                                            total_num_elem * sizeof(T));
  }

  refactored_metadata.Initialize(domain_decomposer.num_subdomains());
  refactored_data.Initialize(domain_decomposer.num_subdomains());

  log::info("Output preallocated: " + std::to_string(output_pre_allocated));
  log::info("Input previously pinned: " +
            std::to_string(input_previously_pinned));

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Prepare input and output buffer");
    timer_each.clear();
  }

  if (log::level & log::TIME)
    timer_each.start();
    // Set the number of threads equal to the number of devices
    // So that each thread is responsible for one device
#if MGARD_ENABLE_MULTI_DEVICE
  omp_set_num_threads(adjusted_num_dev);
#pragma omp parallel for firstprivate(config)
#endif
  for (SIZE dev_id = 0; dev_id < adjusted_num_dev; dev_id++) {
#if MGARD_ENABLE_MULTI_DEVICE
    config.dev_id = dev_id;
#endif
    DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
    // Create a series of subdomain ids that are assigned to the current device
    std::vector<SIZE> subdomain_ids =
        domain_decomposer.subdomain_ids_for_device(dev_id);
    if (subdomain_ids.size() == 1) {
      refactor_subdomain(domain_decomposer, subdomain_ids[0], config,
                         refactored_metadata, refactored_data, dev_id);
    } else {
      // Compress a series of subdomains according to the subdomain id list
      if (!config.prefetch) {
        refactor_subdomain_series(domain_decomposer, subdomain_ids, config,
                                  refactored_metadata, refactored_data, dev_id);
      } else {
        refactor_subdomain_series_w_prefetch(domain_decomposer, subdomain_ids,
                                             config, refactored_metadata,
                                             refactored_data, dev_id);
      }
    }
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Aggregated low-level refactoring");
    log::time("Aggregated low-level refactoring throughput: " +
              std::to_string((double)(total_num_elem * sizeof(T)) /
                             timer_each.get() / 1e9) +
              " GB/s");
    timer_each.clear();
  }

  if (log::level & log::TIME)
    timer_each.start();
  Metadata<DeviceType> m;
  if (uniform) {
    m.FillForMDR(
        (T)0.0, config.decomposition, config.lossless, config.huff_dict_size,
        config.huff_block_size, shape, domain_decomposer.domain_decomposed(),
        config.domain_decomposition, domain_decomposer.domain_decomposed_dim(),
        domain_decomposer.domain_decomposed_size(), config.total_num_bitplanes);
  } else {
    m.FillForMDR(
        (T)0.0, config.decomposition, config.lossless, config.huff_dict_size,
        config.huff_block_size, shape, domain_decomposer.domain_decomposed(),
        config.domain_decomposition, domain_decomposer.domain_decomposed_dim(),
        domain_decomposer.domain_decomposed_size(), config.total_num_bitplanes,
        coords);
  }

  uint32_t metadata_size;
  Byte *metadata = m.Serialize(metadata_size);
  refactored_metadata.header.resize(metadata_size);
  MemoryManager<DeviceType>::Copy1D(refactored_metadata.header.data(), metadata,
                                    metadata_size);
  MemoryManager<DeviceType>::Free(metadata);

  if (!input_previously_pinned) {
    MemoryManager<DeviceType>::HostUnregister((void *)original_data);
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Serialization");
    timer_each.clear();
    timer_total.end();
    timer_total.print("High-level refactoring");
    log::time("High-level refactoring throughput: " +
              std::to_string((double)(total_num_elem * sizeof(T)) /
                             timer_total.get() / 1e9) +
              " GB/s");
    timer_total.clear();
  }
}

template <DIM D, typename T, typename DeviceType>
void MDRefactor(std::vector<SIZE> shape, const void *original_data,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated) {

  MDRefactor<D, T, DeviceType>(shape, original_data, true, std::vector<T *>(0),
                               refactored_metadata, refactored_data, config,
                               output_pre_allocated);
}

template <DIM D, typename T, typename DeviceType>
void MDRefactor(std::vector<SIZE> shape, const void *original_data,
                std::vector<T *> coords,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated) {

  MDRefactor<D, T, DeviceType>(shape, original_data, false, coords,
                               refactored_metadata, refactored_data, config,
                               output_pre_allocated);
}

template <DIM D, typename T, typename DeviceType>
void MDRequest(std::vector<SIZE> shape,
               RefactoredMetadata &refactored_metadata) {
  Config config;
  Metadata<DeviceType> m;
  m.Deserialize((SERIALIZED_TYPE *)refactored_metadata.header.data());
  load(config, m);
  DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>
      domain_decomposer;
  int adjusted_num_dev = 1;
  domain_decomposer =
      DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>(
          shape, adjusted_num_dev, m.domain_decomposed, m.domain_decomposed_dim,
          m.domain_decomposed_size, config);
  generate_request(domain_decomposer, config, refactored_metadata);
}

template <DIM D, typename T, typename DeviceType>
void MDReconstruct(std::vector<SIZE> shape,
                   RefactoredMetadata &refactored_metadata,
                   RefactoredData &refactored_data,
                   ReconstructedData &reconstructed_data, Config config,
                   bool output_pre_allocated, const void *original_data) {
  config.apply();

  size_t total_num_elem = 1;
  for (int i = 0; i < D; i++)
    total_num_elem *= shape[i];

  if (config.num_dev <= 0) {
    log::err("Number of device needs to be greater than 0.");
    exit(-1);
  }

  int adjusted_num_dev =
      std::min(DeviceRuntime<DeviceType>::GetDeviceCount(), config.num_dev);
  if (adjusted_num_dev != config.num_dev) {
    log::info("Using " + std::to_string(adjusted_num_dev) +
              " devices (adjusted from " + std::to_string(config.num_dev) +
              " devices)");
  } else {
    log::info("Using " + std::to_string(adjusted_num_dev) + " devices.");
  }

  Timer timer_total, timer_each;
  if (log::level & log::TIME)
    timer_total.start();

  if (log::level & log::TIME)
    timer_each.start();

  Metadata<DeviceType> m;
  m.Deserialize((SERIALIZED_TYPE *)refactored_metadata.header.data());
  load(config, m);

  std::vector<T *> coords(D);
  if (m.dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
    for (DIM d = 0; d < D; d++) {
      coords[d] = new T[shape[d]];
      for (SIZE i = 0; i < shape[d]; i++) {
        coords[d][i] = (float)m.coords[d][i];
      }
    }
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Deserialization");
    timer_each.clear();
  }

  if (log::level & log::TIME)
    timer_each.start();

  if (m.domain_decomposed) {
    // Fast copy for domain decomposition need we disable pitched memory
    // allocation
    MemoryManager<DeviceType>::ReduceMemoryFootprint = true;
  }

  // Initialize DomainDecomposer
  DomainDecomposer<D, T, ComposedReconstructor<D, T, DeviceType>, DeviceType>
      domain_decomposer;
  if (m.dstype == data_structure_type::Cartesian_Grid_Uniform) {
    domain_decomposer =
        DomainDecomposer<D, T, ComposedReconstructor<D, T, DeviceType>,
                         DeviceType>(
            shape, adjusted_num_dev, m.domain_decomposed,
            m.domain_decomposed_dim, m.domain_decomposed_size, config);
  } else {
    domain_decomposer =
        DomainDecomposer<D, T, ComposedReconstructor<D, T, DeviceType>,
                         DeviceType>(
            shape, adjusted_num_dev, m.domain_decomposed,
            m.domain_decomposed_dim, m.domain_decomposed_size, config, coords);
  }
  if (!config.mdr_adaptive_resolution) {
    if (!reconstructed_data.IsInitialized()) {
      // First time reconstruction
      reconstructed_data.Initialize(1);
      reconstructed_data.data[0] = (Byte *)malloc(total_num_elem * sizeof(T));
      reconstructed_data.offset[0] = std::vector<SIZE>(D, 0);
      reconstructed_data.shape[0] = shape;
    }
    domain_decomposer.set_original_data((T *)reconstructed_data.data[0]);
  } else {
    if (!reconstructed_data.IsInitialized()) {
      // First time reconstruction
      reconstructed_data.Initialize(domain_decomposer.num_subdomains());
      for (int subdomain_id = 0;
           subdomain_id < domain_decomposer.num_subdomains(); subdomain_id++) {
        SIZE n = 1;
        for (int i = 0;
             i < domain_decomposer.subdomain_shape(subdomain_id).size(); i++) {
          n *= domain_decomposer.subdomain_shape(subdomain_id)[i];
        }
        reconstructed_data.data[subdomain_id] = (Byte *)malloc(n * sizeof(T));
      }
    }
    std::vector<T *> decomposed_original_data(
        domain_decomposer.num_subdomains());
    for (int subdomain_id = 0;
         subdomain_id < domain_decomposer.num_subdomains(); subdomain_id++) {
      decomposed_original_data[subdomain_id] =
          (T *)reconstructed_data.data[subdomain_id];
    }
    domain_decomposer.set_decomposed_original_data(decomposed_original_data);
  }

  // This is for accessing original data for error collection
  DomainDecomposer<D, T, ComposedReconstructor<D, T, DeviceType>, DeviceType>
      domain_decomposer_org;
  if (config.collect_uncertainty) {
    if (m.dstype == data_structure_type::Cartesian_Grid_Uniform) {
      domain_decomposer_org =
          DomainDecomposer<D, T, ComposedReconstructor<D, T, DeviceType>,
                           DeviceType>(
              shape, adjusted_num_dev, m.domain_decomposed,
              m.domain_decomposed_dim, m.domain_decomposed_size, config);
    } else {
      domain_decomposer_org =
          DomainDecomposer<D, T, ComposedReconstructor<D, T, DeviceType>,
                           DeviceType>(
              shape, adjusted_num_dev, m.domain_decomposed,
              m.domain_decomposed_dim, m.domain_decomposed_size, config,
              coords);
    }
    domain_decomposer_org.set_original_data((T *)original_data);
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Prepare input and output buffer");
    timer_each.clear();
  }

  if (log::level & log::TIME)
    timer_each.start();
    // decompress
    // Set the number of threads equal to the number of devices
    // So that each thread is responsible for one device
#if MGARD_ENABLE_MULTI_DEVICE
  omp_set_num_threads(adjusted_num_dev);
#pragma omp parallel for firstprivate(config)
#endif
  for (SIZE dev_id = 0; dev_id < adjusted_num_dev; dev_id++) {
#if MGARD_ENABLE_MULTI_DEVICE
    config.dev_id = dev_id;
#endif
    DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
    // Create a series of subdomain ids that are assigned to the current device
    std::vector<SIZE> subdomain_ids =
        domain_decomposer.subdomain_ids_for_device(dev_id);
    if (subdomain_ids.size() == 1) {
      reconstruct_subdomain(domain_decomposer, domain_decomposer_org,
                            subdomain_ids[0], config, refactored_metadata,
                            refactored_data, reconstructed_data, dev_id);
    } else {
      // Decompress a series of subdomains according to the subdomain id list
      if (!config.prefetch) {
        reconstruct_subdomain_series(
            domain_decomposer, domain_decomposer_org, subdomain_ids, config,
            refactored_metadata, refactored_data, reconstructed_data, dev_id);
      } else {
        reconstruct_subdomain_series_w_prefetch(
            domain_decomposer, domain_decomposer_org, subdomain_ids, config,
            refactored_metadata, refactored_data, reconstructed_data, dev_id);
      }
    }
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Aggregated low-level reconstruction");
    log::time("Aggregated low-level reconstruction throughput: " +
              std::to_string((double)(total_num_elem * sizeof(T)) /
                             timer_each.get() / 1e9) +
              " GB/s");
    timer_each.clear();
  }

  if (m.dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
    for (DIM d = 0; d < D; d++)
      delete[] coords[d];
  }

  if (log::level & log::TIME) {
    timer_total.end();
    timer_total.print("High-level reconstruction");
    log::time("High-level reconstruction throughput: " +
              std::to_string((double)(total_num_elem * sizeof(T)) /
                             timer_total.get() / 1e9) +
              " GB/s");
    timer_total.clear();
  }
}

template <typename DeviceType>
void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape,
                const void *original_data,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated) {
  if (dtype == data_type::Float) {
    if (D == 1) {
      MDRefactor<1, float, DeviceType>(shape, original_data,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 2) {
      MDRefactor<2, float, DeviceType>(shape, original_data,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 3) {
      MDRefactor<3, float, DeviceType>(shape, original_data,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 4) {
      MDRefactor<4, float, DeviceType>(shape, original_data,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 5) {
      MDRefactor<5, float, DeviceType>(shape, original_data,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (D == 1) {
      MDRefactor<1, double, DeviceType>(shape, original_data,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 2) {
      MDRefactor<2, double, DeviceType>(shape, original_data,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 3) {
      MDRefactor<3, double, DeviceType>(shape, original_data,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 4) {
      MDRefactor<4, double, DeviceType>(shape, original_data,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 5) {
      MDRefactor<5, double, DeviceType>(shape, original_data,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else {
    log::err("do not support types other than double and float!");
    exit(-1);
  }
}

template <typename DeviceType>
void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape,
                const void *original_data, std::vector<const Byte *> coords,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated) {
  if (dtype == data_type::Float) {
    std::vector<float *> float_coords;
    for (auto &coord : coords)
      float_coords.push_back((float *)coord);
    if (D == 1) {
      MDRefactor<1, float, DeviceType>(shape, original_data, float_coords,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 2) {
      MDRefactor<2, float, DeviceType>(shape, original_data, float_coords,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 3) {
      MDRefactor<3, float, DeviceType>(shape, original_data, float_coords,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 4) {
      MDRefactor<4, float, DeviceType>(shape, original_data, float_coords,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 5) {
      MDRefactor<5, float, DeviceType>(shape, original_data, float_coords,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    std::vector<double *> double_coords;
    for (auto &coord : coords)
      double_coords.push_back((double *)coord);
    if (D == 1) {
      MDRefactor<1, double, DeviceType>(shape, original_data, double_coords,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 2) {
      MDRefactor<2, double, DeviceType>(shape, original_data, double_coords,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 3) {
      MDRefactor<3, double, DeviceType>(shape, original_data, double_coords,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 4) {
      MDRefactor<4, double, DeviceType>(shape, original_data, double_coords,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 5) {
      MDRefactor<5, double, DeviceType>(shape, original_data, double_coords,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else {
    log::err("do not support types other than double and float!");
    exit(-1);
  }
}

template <typename DeviceType>
void MDRequest(RefactoredMetadata &refactored_metadata) {
  Metadata<DeviceType> meta;
  meta.Deserialize((SERIALIZED_TYPE *)refactored_metadata.header.data());

  std::vector<SIZE> shape = std::vector<SIZE>(meta.total_dims);
  for (DIM d = 0; d < shape.size(); d++)
    shape[d] = (SIZE)meta.shape[d];
  data_type dtype = meta.dtype;

  if (dtype == data_type::Float) {
    if (shape.size() == 1) {
      MDRequest<1, float, DeviceType>(shape, refactored_metadata);
    } else if (shape.size() == 2) {
      MDRequest<2, float, DeviceType>(shape, refactored_metadata);
    } else if (shape.size() == 3) {
      MDRequest<3, float, DeviceType>(shape, refactored_metadata);
    } else if (shape.size() == 4) {
      MDRequest<4, float, DeviceType>(shape, refactored_metadata);
    } else if (shape.size() == 5) {
      MDRequest<5, float, DeviceType>(shape, refactored_metadata);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (shape.size() == 1) {
      MDRequest<1, double, DeviceType>(shape, refactored_metadata);
    } else if (shape.size() == 2) {
      MDRequest<2, double, DeviceType>(shape, refactored_metadata);
    } else if (shape.size() == 3) {
      MDRequest<3, double, DeviceType>(shape, refactored_metadata);
    } else if (shape.size() == 4) {
      MDRequest<4, double, DeviceType>(shape, refactored_metadata);
    } else if (shape.size() == 5) {
      MDRequest<5, double, DeviceType>(shape, refactored_metadata);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else {
    log::err("do not support types other than double and float!");
    exit(-1);
  }
}

template <typename DeviceType>
void MDReconstruct(RefactoredMetadata &refactored_metadata,
                   RefactoredData &refactored_data,
                   ReconstructedData &reconstructed_data, Config config,
                   bool output_pre_allocated, const void *original_data) {

  Metadata<DeviceType> meta;
  meta.Deserialize((SERIALIZED_TYPE *)refactored_metadata.header.data());

  std::vector<SIZE> shape = std::vector<SIZE>(meta.total_dims);
  for (DIM d = 0; d < shape.size(); d++)
    shape[d] = (SIZE)meta.shape[d];
  data_type dtype = meta.dtype;

  if (dtype == data_type::Float) {
    if (shape.size() == 1) {
      MDReconstruct<1, float, DeviceType>(
          shape, refactored_metadata, refactored_data, reconstructed_data,
          config, output_pre_allocated, original_data);
    } else if (shape.size() == 2) {
      MDReconstruct<2, float, DeviceType>(
          shape, refactored_metadata, refactored_data, reconstructed_data,
          config, output_pre_allocated, original_data);
    } else if (shape.size() == 3) {
      MDReconstruct<3, float, DeviceType>(
          shape, refactored_metadata, refactored_data, reconstructed_data,
          config, output_pre_allocated, original_data);
    } else if (shape.size() == 4) {
      MDReconstruct<4, float, DeviceType>(
          shape, refactored_metadata, refactored_data, reconstructed_data,
          config, output_pre_allocated, original_data);
    } else if (shape.size() == 5) {
      MDReconstruct<5, float, DeviceType>(
          shape, refactored_metadata, refactored_data, reconstructed_data,
          config, output_pre_allocated, original_data);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (shape.size() == 1) {
      MDReconstruct<1, double, DeviceType>(
          shape, refactored_metadata, refactored_data, reconstructed_data,
          config, output_pre_allocated, original_data);
    } else if (shape.size() == 2) {
      MDReconstruct<2, double, DeviceType>(
          shape, refactored_metadata, refactored_data, reconstructed_data,
          config, output_pre_allocated, original_data);
    } else if (shape.size() == 3) {
      MDReconstruct<3, double, DeviceType>(
          shape, refactored_metadata, refactored_data, reconstructed_data,
          config, output_pre_allocated, original_data);
    } else if (shape.size() == 4) {
      MDReconstruct<4, double, DeviceType>(
          shape, refactored_metadata, refactored_data, reconstructed_data,
          config, output_pre_allocated, original_data);
    } else if (shape.size() == 5) {
      MDReconstruct<5, double, DeviceType>(
          shape, refactored_metadata, refactored_data, reconstructed_data,
          config, output_pre_allocated, original_data);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else {
    log::err("do not support types other than double and float!");
    exit(-1);
  }
}

} // namespace MDR
} // namespace mgard_x

#endif