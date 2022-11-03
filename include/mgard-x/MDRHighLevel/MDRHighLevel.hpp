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

#include "mdr_x.hpp"
#include "MDRHighLevel.h"
#include "../CompressionLowLevel/Compressor.hpp"
#include "../CompressionHighLevel/Metadata.hpp"
#include "../CompressionHighLevel/DomainDecomposer.hpp"

namespace mgard_x {
namespace MDR {

template <DIM D, typename T, typename DeviceType>
void generate_request(DomainDecomposer<D, T, DeviceType> &domain_decomposer, Config config,
                      AggregatedMDRMetaData &aggregated_mdr_metadata,
                      double tol, double s) {
  for (int subdomain_id = 0; subdomain_id < domain_decomposer.num_subdomains(); subdomain_id++) {
    Hierarchy<D, T, DeviceType> hierarchy =
      domain_decomposer.subdomain_hierarchy(subdomain_id);
    ComposedReconstructor<D, T, DeviceType> reconstructor(hierarchy, config);
    reconstructor.GenerateRequest(aggregated_mdr_metadata.metadata[subdomain_id], tol, s);
  }
} 

template <DIM D, typename T, typename DeviceType>
void refactor_subdomain(DomainDecomposer<D, T, DeviceType> &domain_decomposer,
                        SIZE subdomain_id, Config &config,
                        AggregatedMDRMetaData &aggregated_mdr_metadata,
                        AggregatedMDRData &aggregated_mdr_data,
                        int dev_id) {

  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();
  Array<D, T, DeviceType> device_subdomain_buffer;
  MDRMetaData mdr_metadata;
  MDRData<DeviceType> mdr_data;
  domain_decomposer.copy_subdomain(device_subdomain_buffer, subdomain_id,
                                   ORIGINAL_TO_SUBDOMAIN, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  // Trigger the copy constructor to copy hierarchy to the current device
  Hierarchy<D, T, DeviceType> hierarchy =
      domain_decomposer.subdomain_hierarchy(subdomain_id);

  ComposedRefactor<D, T, DeviceType> refactor(hierarchy, config);
  std::stringstream ss;
  for (DIM d = 0; d < D; d++) {
    ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
  }
  log::info("Refactoring subdomain " + std::to_string(subdomain_id) +
            " with shape: " + ss.str());
  refactor.Refactor(device_subdomain_buffer, mdr_metadata, mdr_data, 0);
  mdr_data.CopyToAggregatedMDRData(mdr_metadata, aggregated_mdr_data.data[subdomain_id], 0);
  aggregated_mdr_metadata.metadata[subdomain_id] = mdr_metadata;
  DeviceRuntime<DeviceType>::SyncQueue(0);
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Refactor single subdomain");
    timer_series.clear();
  }
  DeviceRuntime<DeviceType>::SyncDevice();
}

template <DIM D, typename T, typename DeviceType>
void reconstruct_subdomain(DomainDecomposer<D, T, DeviceType> &domain_decomposer,
                          SIZE subdomain_id, Config &config,
                          AggregatedMDRMetaData &aggregated_mdr_metadata,
                          AggregatedMDRData &aggregated_mdr_data,
                          int dev_id) {

  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();
  Array<D, T, DeviceType> device_subdomain_buffer;
  MDRMetaData mdr_metadata;
  MDRData<DeviceType> mdr_data;
  mdr_data.Resize(mdr_metadata);
  mdr_data.CopyFromAggregatedMDRData(mdr_metadata, aggregated_mdr_data.data[subdomain_id], 0);

  Hierarchy<D, T, DeviceType> hierarchy =
      domain_decomposer.subdomain_hierarchy(subdomain_id);
  ComposedReconstructor<D, T, DeviceType> reconstructor(hierarchy, config);

  std::stringstream ss;
  for (DIM d = 0; d < D; d++) {
    ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
  }
  log::info("Reconstruct subdomain " + std::to_string(subdomain_id) +
            " with shape: " + ss.str());
  reconstructor.ProgressiveReconstruct(mdr_metadata, mdr_data, device_subdomain_buffer, 0);

  domain_decomposer.copy_subdomain(device_subdomain_buffer, subdomain_id,
                                   SUBDOMAIN_TO_ORIGINAL, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Reconstruct single subdomain");
    timer_series.clear();
  }
  DeviceRuntime<DeviceType>::SyncDevice();
}

template <typename DeviceType>
void load(Config &config, Metadata<DeviceType> &metadata) {
  config.decomposition = metadata.decomposition;
  config.lossless = metadata.ltype;
  config.huff_dict_size = metadata.huff_dict_size;
  config.huff_block_size = metadata.huff_block_size;
  config.reorder = metadata.reorder;
}

template <DIM D, typename T, typename DeviceType>
void MDRefactor(std::vector<SIZE> shape, const void *original_data,
                AggregatedMDRMetaData &refactored_metadata,
                AggregatedMDRData &refactored_data,
                Config config, bool uniform, std::vector<T *> coords,
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

  DomainDecomposer<D, T, DeviceType> domain_decomposer;
  if (uniform) {
    domain_decomposer = DomainDecomposer<D, T, DeviceType>(
        (T *)original_data, shape, adjusted_num_dev, config);
  } else {
    domain_decomposer = DomainDecomposer<D, T, DeviceType>(
        (T *)original_data, shape, adjusted_num_dev, config, coords);
  }

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
        // TODO
      } else {
        // TODO
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
    m.Fill(error_bound_type::REL, (T)0.0, (T)0.0, (T)0.0, config.decomposition, config.reorder,
           config.lossless, config.huff_dict_size, config.huff_block_size,
           shape, domain_decomposer.domain_decomposed(),
           domain_decomposer.domain_decomposed_dim(),
           domain_decomposer.domain_decomposed_size());
  } else {
    m.Fill(error_bound_type::REL, (T)0.0, (T)0.0, (T)0.0, config.decomposition, config.reorder,
           config.lossless, config.huff_dict_size, config.huff_block_size,
           shape, domain_decomposer.domain_decomposed(),
           domain_decomposer.domain_decomposed_dim(),
           domain_decomposer.domain_decomposed_size(), coords);
  }

  uint32_t metadata_size;
  refactored_metadata.header = m.Serialize(metadata_size);

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
                AggregatedMDRMetaData &refactored_metadata,
                AggregatedMDRData &refactored_data,
                Config config, bool output_pre_allocated) {

  MDRefactor<D, T, DeviceType>(shape, original_data, refactored_metadata,
             refactored_data, config, true, std::vector<T *>(0),
             output_pre_allocated);
}

template <DIM D, typename T, typename DeviceType>
void MDRefactor(std::vector<SIZE> shape, const void *original_data,
                AggregatedMDRMetaData &refactored_metadata,
                AggregatedMDRData &refactored_data,
                Config config, std::vector<T *> coords, bool output_pre_allocated) {

  MDRefactor<D, T, DeviceType>(shape, original_data, refactored_metadata,
             refactored_data, config, false, coords,
             output_pre_allocated);
}

template <DIM D, typename T, typename DeviceType>
void MDRequest(std::vector<SIZE> shape, 
               AggregatedMDRMetaData &refactored_metadata, double tol, double s,
               enum error_bound_type ebtype) {
  Config config;
  Metadata<DeviceType> m;
  m.Deserialize((SERIALIZED_TYPE *)refactored_metadata.header);

  DomainDecomposer<D, T, DeviceType> domain_decomposer;
  int adjusted_num_dev = 1;
  domain_decomposer = DomainDecomposer<D, T, DeviceType>(
      (T *)NULL, shape, adjusted_num_dev, m.domain_decomposed,
      m.domain_decomposed_dim, m.domain_decomposed_size, config);  
  generate_request(domain_decomposer, config, refactored_metadata, tol, s);
}

template <DIM D, typename T, typename DeviceType>
void MDRconstruct(std::vector<SIZE> shape,
                  AggregatedMDRMetaData &refactored_metadata,
                  AggregatedMDRData &refactored_data,
                  ReconstructuredData &reconstructed_data, Config config,
                  bool output_pre_allocated) {
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

  config.apply();

  Timer timer_total, timer_each;
  if (log::level & log::TIME)
    timer_total.start();
  if (log::level & log::TIME)
    timer_each.start();


  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Prepare input and output buffer");
    timer_each.clear();
  }

  if (log::level & log::TIME)
    timer_each.start();

  Metadata<DeviceType> m;
  m.Deserialize((SERIALIZED_TYPE *)refactored_metadata.header);
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

  SIZE num_subdomains;
  if (!m.domain_decomposed) {
    num_subdomains = 1;
  } else {
    num_subdomains =
        (shape[m.domain_decomposed_dim] - 1) / m.domain_decomposed_size + 1;
  }

  // Preparing decompression parameters
  T local_tol;
  enum error_bound_type local_ebtype;

  if (log::level & log::TIME)
    timer_each.start();

  if (m.domain_decomposed) {
    // Fast copy for domain decomposition need we disable pitched memory
    // allocation
    MemoryManager<DeviceType>::ReduceMemoryFootprint = true;
  }

  // Initialize DomainDecomposer
  DomainDecomposer<D, T, DeviceType> domain_decomposer;

  if (m.dstype == data_structure_type::Cartesian_Grid_Uniform) {
    domain_decomposer = DomainDecomposer<D, T, DeviceType>(
        (T *)NULL, shape, adjusted_num_dev, m.domain_decomposed,
        m.domain_decomposed_dim, m.domain_decomposed_size, config);
  } else {
    domain_decomposer = domain_decomposer = DomainDecomposer<D, T, DeviceType>(
        (T *)NULL, shape, adjusted_num_dev, m.domain_decomposed,
        m.domain_decomposed_dim, m.domain_decomposed_size, config, coords);
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Deserialization");
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
      reconstruct_subdomain(domain_decomposer, subdomain_ids[0], config,
                           refactored_metadata, refactored_data, dev_id);
    } else {
      // Decompress a series of subdomains according to the subdomain id list
      if (!config.prefetch) {
        //TODO
      } else {
        //TODO
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
void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape, const void *original_data,
          AggregatedMDRMetaData &refactored_metadata,
          AggregatedMDRData &refactored_data,
          Config config, bool output_pre_allocated) {
  if (dtype == data_type::Float) {
    if (D == 1) {
      MDRefactor<1, float, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, output_pre_allocated);
    } else if (D == 2) {
      MDRefactor<2, float, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, output_pre_allocated);
    } else if (D == 3) {
      MDRefactor<3, float, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, output_pre_allocated);
    } else if (D == 4) {
      MDRefactor<4, float, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, output_pre_allocated);
    } else if (D == 5) {
      MDRefactor<5, float, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (D == 1) {
      MDRefactor<1, double, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, output_pre_allocated);
    } else if (D == 2) {
      MDRefactor<2, double, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, output_pre_allocated);
    } else if (D == 3) {
      MDRefactor<3, double, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, output_pre_allocated);
    } else if (D == 4) {
      MDRefactor<4, double, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, output_pre_allocated);
    } else if (D == 5) {
      MDRefactor<5, double, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, output_pre_allocated);
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
void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape, const void *original_data,
          AggregatedMDRMetaData &refactored_metadata,
          AggregatedMDRData &refactored_data,
          Config config, std::vector<const Byte *> coords, bool output_pre_allocated) {
  if (dtype == data_type::Float) {
    std::vector<float *> float_coords;
    for (auto &coord : coords)
      float_coords.push_back((float *)coord);
    if (D == 1) {
      MDRefactor<1, float, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, float_coords, output_pre_allocated);
    } else if (D == 2) {
      MDRefactor<2, float, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, float_coords, output_pre_allocated);
    } else if (D == 3) {
      MDRefactor<3, float, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, float_coords, output_pre_allocated);
    } else if (D == 4) {
      MDRefactor<4, float, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, float_coords, output_pre_allocated);
    } else if (D == 5) {
      MDRefactor<5, float, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, float_coords, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    std::vector<double *> double_coords;
    for (auto &coord : coords)
      double_coords.push_back((double *)coord);
    if (D == 1) {
      MDRefactor<1, double, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, double_coords, output_pre_allocated);
    } else if (D == 2) {
      MDRefactor<2, double, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, double_coords, output_pre_allocated);
    } else if (D == 3) {
      MDRefactor<3, double, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, double_coords, output_pre_allocated);
    } else if (D == 4) {
      MDRefactor<4, double, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, double_coords, output_pre_allocated);
    } else if (D == 5) {
      MDRefactor<5, double, DeviceType>(shape, original_data, refactored_metadata, 
                                      refactored_data, config, double_coords, output_pre_allocated);
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
void MDRequest(AggregatedMDRMetaData &refactored_metadata, double tol, double s,
               enum error_bound_type ebtype) {

  Metadata<DeviceType> meta;
  meta.Deserialize((SERIALIZED_TYPE *)refactored_metadata.header);

  std::vector<SIZE> shape = std::vector<SIZE>(meta.total_dims);
  for (DIM d = 0; d < shape.size(); d++)
    shape[d] = (SIZE)meta.shape[d];
  data_type dtype = meta.dtype;

  if (dtype == data_type::Float) {
    if (shape.size() == 1) {
      MDRequest<1, float, DeviceType>(shape, refactored_metadata, tol, s, ebtype);
    } else if (shape.size() == 2) {
      MDRequest<2, float, DeviceType>(shape, refactored_metadata, tol, s, ebtype);
    } else if (shape.size() == 3) {
      MDRequest<3, float, DeviceType>(shape, refactored_metadata, tol, s, ebtype);
    } else if (shape.size() == 4) {
      MDRequest<4, float, DeviceType>(shape, refactored_metadata, tol, s, ebtype);
    } else if (shape.size() == 5) {
      MDRequest<5, float, DeviceType>(shape, refactored_metadata, tol, s, ebtype);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (shape.size() == 1) {
      MDRequest<1, double, DeviceType>(shape, refactored_metadata, tol, s, ebtype);
    } else if (shape.size() == 2) {
      MDRequest<2, double, DeviceType>(shape, refactored_metadata, tol, s, ebtype);
    } else if (shape.size() == 3) {
      MDRequest<3, double, DeviceType>(shape, refactored_metadata, tol, s, ebtype);
    } else if (shape.size() == 4) {
      MDRequest<4, double, DeviceType>(shape, refactored_metadata, tol, s, ebtype);
    } else if (shape.size() == 5) {
      MDRequest<5, double, DeviceType>(shape, refactored_metadata, tol, s, ebtype);
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
void MDRconstruct(AggregatedMDRMetaData &refactored_metadata,
                  AggregatedMDRData &refactored_data,
                  ReconstructuredData &reconstructed_data, Config config,
                  bool output_pre_allocated) {

  Metadata<DeviceType> meta;
  meta.Deserialize((SERIALIZED_TYPE *)refactored_metadata.header);

  std::vector<SIZE> shape = std::vector<SIZE>(meta.total_dims);
  for (DIM d = 0; d < shape.size(); d++)
    shape[d] = (SIZE)meta.shape[d];
  data_type dtype = meta.dtype;

  if (dtype == data_type::Float) {
    if (shape.size() == 1) {
      MDRconstruct<1, float, DeviceType>(shape, refactored_metadata, refactored_data, reconstructed_data,
                                         config, output_pre_allocated);
    } else if (shape.size() == 2) {
      MDRconstruct<2, float, DeviceType>(shape, refactored_metadata, refactored_data, reconstructed_data,
                                         config, output_pre_allocated);
    } else if (shape.size() == 3) {
      MDRconstruct<3, float, DeviceType>(shape, refactored_metadata, refactored_data, reconstructed_data,
                                         config, output_pre_allocated);
    } else if (shape.size() == 4) {
      MDRconstruct<4, float, DeviceType>(shape, refactored_metadata, refactored_data, reconstructed_data,
                                         config, output_pre_allocated);
    } else if (shape.size() == 5) {
      MDRconstruct<5, float, DeviceType>(shape, refactored_metadata, refactored_data, reconstructed_data,
                                         config, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (shape.size() == 1) {
      MDRconstruct<1, double, DeviceType>(shape, refactored_metadata, refactored_data, reconstructed_data,
                                         config, output_pre_allocated);
    } else if (shape.size() == 2) {
      MDRconstruct<2, double, DeviceType>(shape, refactored_metadata, refactored_data, reconstructed_data,
                                         config, output_pre_allocated);
    } else if (shape.size() == 3) {
      MDRconstruct<3, double, DeviceType>(shape, refactored_metadata, refactored_data, reconstructed_data,
                                         config, output_pre_allocated);
    } else if (shape.size() == 4) {
      MDRconstruct<4, double, DeviceType>(shape, refactored_metadata, refactored_data, reconstructed_data,
                                         config, output_pre_allocated);
    } else if (shape.size() == 5) {
      MDRconstruct<5, double, DeviceType>(shape, refactored_metadata, refactored_data, reconstructed_data,
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

}
}

#endif