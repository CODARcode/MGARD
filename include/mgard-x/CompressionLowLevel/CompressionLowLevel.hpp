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
#include <vector>

#include "../Utilities/Types.h"

#include "../Config/Config.h"
#include "../Hierarchy/Hierarchy.hpp"
#include "../RuntimeX/RuntimeX.h"
#include "CompressionLowLevel.h"

#ifndef MGARD_X_COMPRESSION_LOW_LEVEL_HPP
#define MGARD_X_COMPRESSION_LOW_LEVEL_HPP

namespace mgard_x {

static bool debug_print_compression = true;

template <DIM D, typename T, typename DeviceType>
void compress(Hierarchy<D, T, DeviceType> &hierarchy,
         Array<D, T, DeviceType> &original_array, enum error_bound_type type,
         T tol, T s, T &norm, Config config,
         CompressionLowLevelWorkspace<D, T, DeviceType> &workspace,
         Array<1, Byte, DeviceType> &compressed_array) {

  config.apply();

  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
  Timer timer_total;
  for (int d = D - 1; d >= 0; d--) {
    if (hierarchy.level_shape(hierarchy.l_target(), d) !=
        original_array.shape(d)) {
      log::err("The shape of input array does not match the shape initilized "
               "in hierarchy!");
      return;
    }
  }

  if (log::level & log::TIME)
    timer_total.start();

  // Norm
  if (type == error_bound_type::REL) {
    norm = norm_calculator(original_array, workspace.norm_tmp_subarray,
                           workspace.norm_subarray, s,
                           config.normalize_coordinates);
  }

  Decompose(hierarchy, original_array, config, workspace.data_refactoring_workspace, 0);

  LinearQuanziation(hierarchy, original_array,
                    config, type, tol, s, norm, workspace, 0);

  LosslessCompress(hierarchy, compressed_array, config, workspace);

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_total.end();
    timer_total.print("Low-level compression");
    log::time("Low-level compression throughput: " +
              std::to_string((double)(hierarchy.total_num_elems() * sizeof(T)) /
                             timer_total.get() / 1e9) +
              " GB/s");
    timer_total.clear();
  }
}

template <DIM D, typename T, typename DeviceType>
void decompress(Hierarchy<D, T, DeviceType> &hierarchy,
           Array<1, unsigned char, DeviceType> &compressed_array,
           enum error_bound_type type, T tol, T s, T norm, Config config,
           CompressionLowLevelWorkspace<D, T, DeviceType> &workspace,
           Array<D, T, DeviceType>& decompressed_array) {

  config.apply();

  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
  Timer timer_total, timer_each;

  if (log::level & log::TIME)
    timer_total.start();

  LosslessDecompress(hierarchy, compressed_array, config, workspace);

  LinearDequanziation(hierarchy, decompressed_array, config, type, tol, s, norm,
                      workspace, 0);

  Recompose(hierarchy, decompressed_array, config, workspace.data_refactoring_workspace, 0);

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_total.end();
    timer_total.print("Low-level decompression");
    log::time("Low-level decompression throughput: " +
              std::to_string((double)(hierarchy.total_num_elems() * sizeof(T)) /
                             timer_total.get() / 1e9) +
              " GB/s");
    timer_total.clear();
  }
}
} // namespace mgard_x

#endif