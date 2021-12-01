/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "compress_cuda.hpp"
#include "mgard-x/Handle.h" 
#include "mgard-x/Metadata.hpp"
#include "mgard-x/RuntimeX/RuntimeXPublic.h"

#ifndef MGARD_X_HIGH_LEVEL_API_H
#define MGARD_X_HIGH_LEVEL_API_H

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void compress(std::vector<SIZE> shape, T tol, T s, enum error_bound_type mode,
              const void *original_data, void *&compressed_data,
              size_t &compressed_size, Config config, bool output_pre_allocated);

template <DIM D, typename T, typename DeviceType>
void compress(std::vector<SIZE> shape, T tol, T s, enum error_bound_type mode,
              const void *original_data, void *&compressed_data,
              size_t &compressed_size, Config config, std::vector<T *> coords,
              bool output_pre_allocated);

template <DIM D, typename T, typename DeviceType>
void decompress(std::vector<SIZE> shape, const void *compressed_data,
                size_t compressed_size, void *&decompressed_data,
                std::vector<T *> coords, Config config,
                bool output_pre_allocated);

template <DIM D, typename T, typename DeviceType>
void decompress(std::vector<SIZE> shape, const void *compressed_data,
                size_t compressed_size, void *&decompressed_data,
                Config config,
                bool output_pre_allocated);

}

#endif