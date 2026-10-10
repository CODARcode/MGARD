/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: October 9, 2026
 */

#ifndef MGARD_X_DOMAIN_DECOMPOSER_HPP
#define MGARD_X_DOMAIN_DECOMPOSER_HPP

#include <algorithm>
#include <sstream>

#include "../Config/Config.h"
#include "../Hierarchy/Hierarchy.hpp"

namespace mgard_x {

enum class subdomain_copy_direction : uint8_t {
  OriginalToSubdomain,
  SubdomainToOriginal
};

// Splits an N-D array into a grid of blocks (subdomains) that are processed
// one at a time.
//
// All blocks share one shape, block_shape. Along each dimension block b starts
// at b * block_shape[d]; the last block takes whatever remains, and a
// remainder narrower than MIN_BLOCK_EXTENT is merged into the block before it,
// so every block is a valid Hierarchy. Subdomain ids enumerate the block grid
// row-major, like the data (last dimension fastest). A single block covering
// the whole array means no decomposition.
//
// Compression picks the block shape from the Config:
// - Auto: one block if the array fits the memory limit; otherwise the largest
//   block extent is halved until the largest block fits.
// - Manual: Config::domain_decomposition_block_size, rejected if a block would
//   be narrower than MIN_BLOCK_EXTENT or would not fit the memory limit.
// The block shape is recorded in the header, and decompression rebuilds the
// same grid from it.
template <DIM D, typename T, typename OperatorType, typename DeviceType>
class DomainDecomposer {
public:
  // Every block becomes its own Hierarchy, which needs at least this many
  // nodes along each dimension.
  static constexpr SIZE MIN_BLOCK_EXTENT = 3;

  DomainDecomposer() {}

  // Compression: choose the block shape according to the Config. A non-empty
  // `coords` means a non-uniform grid.
  DomainDecomposer(std::vector<SIZE> shape, Config config,
                   std::vector<T *> coords = std::vector<T *>())
      : shape(shape), config(config), coords(coords), uniform(coords.empty()) {
    if (config.domain_decomposition_mode ==
        domain_decomposition_mode_type::Manual) {
      set_block_shape(manual_block_shape());
    } else {
      set_block_shape(auto_block_shape());
    }
  }

  // Decompression: rebuild the block grid recorded in the header.
  DomainDecomposer(std::vector<SIZE> shape, std::vector<SIZE> block_shape,
                   Config config, std::vector<T *> coords = std::vector<T *>())
      : shape(shape), config(config), coords(coords), uniform(coords.empty()) {
    if (block_shape.size() != D) {
      throw InvalidDataException("domain decomposition block shape has " +
                                 std::to_string(block_shape.size()) +
                                 " dimensions, the data has " +
                                 std::to_string(D) + ".");
    }
    set_block_shape(block_shape);
  }

  // ---------------------------------------------------------------- the grid

  // Number of blocks of extent `block` along an extent of `n` nodes, with a
  // too-narrow remainder merged into the last full block.
  static SIZE num_blocks(SIZE n, SIZE block) {
    SIZE count = (n - 1) / block + 1;
    SIZE remainder = n % block;
    if (count > 1 && remainder != 0 && remainder < MIN_BLOCK_EXTENT) {
      count--;
    }
    return count;
  }

  // Extent of block i along an extent of `n` nodes; the last block takes the
  // rest, merged remainder included.
  static SIZE block_extent(SIZE n, SIZE block, SIZE i) {
    SIZE count = num_blocks(n, block);
    return i + 1 < count ? block : n - (count - 1) * block;
  }

  bool domain_decomposed() { return _num_subdomains > 1; }

  SIZE num_subdomains() { return _num_subdomains; }

  // Common block shape (clamped to the data); what the header records.
  std::vector<SIZE> block_shape() { return _block_shape; }

  // Position of a subdomain in the block grid, per dimension.
  std::vector<SIZE> subdomain_grid_index(SIZE subdomain_id) {
    check_subdomain_id(subdomain_id);
    std::vector<SIZE> index(D);
    for (int d = D - 1; d >= 0; d--) {
      index[d] = subdomain_id % _blocks_per_dim[d];
      subdomain_id /= _blocks_per_dim[d];
    }
    return index;
  }

  // First node of a subdomain in the full array, per dimension.
  std::vector<SIZE> subdomain_start(SIZE subdomain_id) {
    std::vector<SIZE> index = subdomain_grid_index(subdomain_id);
    std::vector<SIZE> start(D);
    for (DIM d = 0; d < D; d++) {
      start[d] = index[d] * _block_shape[d];
    }
    return start;
  }

  std::vector<SIZE> subdomain_shape(SIZE subdomain_id) {
    std::vector<SIZE> index = subdomain_grid_index(subdomain_id);
    std::vector<SIZE> extent(D);
    for (DIM d = 0; d < D; d++) {
      extent[d] = block_extent(shape[d], _block_shape[d], index[d]);
    }
    return extent;
  }

  // Id of a subdomain with the most nodes (sizes the shared device buffers).
  SIZE largest_subdomain() {
    SIZE max_count = 0, max_id = 0;
    for (SIZE id = 0; id < _num_subdomains; id++) {
      SIZE count = num_nodes(subdomain_shape(id));
      if (count > max_count) {
        max_count = count;
        max_id = id;
      }
    }
    return max_id;
  }

  SIZE subdomain_compressed_buffer_size(SIZE subdomain_id) {
    SIZE size = 0;
    size += num_nodes(subdomain_shape(subdomain_id)) * sizeof(HUFFMAN_CODE);
    size += config.estimate_outlier_ratio * sizeof(QUANTIZED_INT);
    return size;
  }

  typename OperatorType::HierarchyType subdomain_hierarchy(SIZE subdomain_id) {
    using HierarchyType = typename OperatorType::HierarchyType;
    std::vector<SIZE> extent = subdomain_shape(subdomain_id);
    if (uniform) {
      return HierarchyType(extent, config);
    }
    // Non-uniform: the block's slice of each coordinate array.
    std::vector<SIZE> start = subdomain_start(subdomain_id);
    std::vector<std::vector<T>> block_coords(D);
    std::vector<T *> block_coord_ptrs(D);
    for (DIM d = 0; d < D; d++) {
      block_coords[d].resize(extent[d]);
      MemoryManager<DeviceType>::Copy1D(block_coords[d].data(),
                                        coords[d] + start[d], extent[d], 0);
      block_coord_ptrs[d] = block_coords[d].data();
    }
    DeviceRuntime<DeviceType>::SyncQueue(0);
    return HierarchyType(extent, block_coord_ptrs, config);
  }

  // ------------------------------------------------------ moving block data

  void set_original_data(T *original_data) {
    this->original_data = original_data;
    keep_original_data_decomposed = false;
  }

  // The original data is supplied (or wanted) as one buffer per subdomain,
  // each holding its block contiguously.
  void set_decomposed_original_data(std::vector<T *> decomposed_original_data) {
    this->decomposed_original_data = decomposed_original_data;
    original_data = decomposed_original_data[0];
    keep_original_data_decomposed = true;
  }

  // Copies one subdomain between the original data and a device buffer (which
  // is resized to the subdomain when copying into it).
  void copy_subdomain(Array<D, T, DeviceType> &subdomain_data,
                      SIZE subdomain_id,
                      enum subdomain_copy_direction direction, int queue_idx) {
    std::vector<SIZE> extent = subdomain_shape(subdomain_id);
    if (direction == subdomain_copy_direction::OriginalToSubdomain) {
      subdomain_data.resize(extent, queue_idx);
    }

    T *original;
    std::vector<SIZE> original_strides;
    if (keep_original_data_decomposed) {
      original = decomposed_original_data[subdomain_id];
      original_strides = contiguous_strides(extent);
    } else {
      original_strides = contiguous_strides(shape);
      std::vector<SIZE> start = subdomain_start(subdomain_id);
      SIZE offset = 0;
      for (DIM d = 0; d < D; d++) {
        offset += start[d] * original_strides[d];
      }
      original = original_data + offset;
    }

    // The device buffer is pitched: rows of ld(D - 1) elements.
    std::vector<SIZE> buffer_strides(D);
    buffer_strides[D - 1] = 1;
    if (D >= 2) {
      buffer_strides[D - 2] = subdomain_data.ld(D - 1);
    }
    for (int d = (int)D - 3; d >= 0; d--) {
      buffer_strides[d] = buffer_strides[d + 1] * subdomain_data.shape(d + 1);
    }

    if (direction == subdomain_copy_direction::OriginalToSubdomain) {
      copy_box(subdomain_data.data(), buffer_strides, original,
               original_strides, extent, queue_idx);
    } else {
      copy_box(original, original_strides, subdomain_data.data(),
               buffer_strides, extent, queue_idx);
    }
  }

  // Copies an N-D box of `extent` nodes; strides are in elements, per
  // dimension, for each side. Dimensions that are contiguous on both sides are
  // merged first, so a box spanning whole rows or planes becomes a single 1D
  // or 2D copy. What remains maps onto Copy3D, looping only over dimensions
  // beyond the third.
  static void copy_box(T *dst, std::vector<SIZE> dst_strides, const T *src,
                       std::vector<SIZE> src_strides, std::vector<SIZE> extent,
                       int queue_idx) {
    struct Run {
      SIZE n, dst_stride, src_stride;
    };
    // Innermost first.
    std::vector<Run> runs;
    for (int d = D - 1; d >= 0; d--) {
      if (extent[d] == 1) {
        continue;
      }
      if (!runs.empty() &&
          dst_strides[d] == runs.back().dst_stride * runs.back().n &&
          src_strides[d] == runs.back().src_stride * runs.back().n) {
        runs.back().n *= extent[d];
      } else {
        runs.push_back({extent[d], dst_strides[d], src_strides[d]});
      }
    }
    // The copy primitives move unit-stride rows.
    if (runs.empty() || runs[0].dst_stride != 1 || runs[0].src_stride != 1) {
      runs.insert(runs.begin(), {1, 1, 1});
    }

    using Mem = MemoryManager<DeviceType>;
    if (runs.size() == 1) {
      Mem::Copy1D(dst, src, runs[0].n, queue_idx);
    } else if (runs.size() == 2) {
      Mem::Copy2D(dst, runs[1].dst_stride, src, runs[1].src_stride, runs[0].n,
                  runs[1].n, queue_idx);
    } else {
      // One Copy3D per position in the remaining outer runs.
      std::vector<SIZE> outer(runs.size() - 3, 0);
      while (true) {
        SIZE dst_offset = 0, src_offset = 0;
        for (size_t i = 0; i < outer.size(); i++) {
          dst_offset += outer[i] * runs[3 + i].dst_stride;
          src_offset += outer[i] * runs[3 + i].src_stride;
        }
        Mem::Copy3D(dst + dst_offset, runs[1].dst_stride, runs[2].dst_stride,
                    src + src_offset, runs[1].src_stride, runs[2].src_stride,
                    runs[0].n, runs[1].n, runs[2].n, queue_idx);
        size_t i = 0;
        while (i < outer.size() && ++outer[i] == runs[3 + i].n) {
          outer[i++] = 0;
        }
        if (i == outer.size()) {
          break;
        }
      }
    }
  }

  // ------------------------------------------------------- memory estimates

  size_t EstimateMemoryFootprint(std::vector<SIZE> shape,
                                 bool enable_prefetch) {
    size_t estimate_memory_usgae = 0;

    Array<1, T, DeviceType> array_with_pitch({1});
    size_t pitch_size = array_with_pitch.ld(0) * sizeof(T);

    size_t input_space = roundup((size_t)shape[D - 1] * sizeof(T), pitch_size);
    for (DIM d = 0; d < D - 1; d++) {
      input_space *= shape[d];
    }

    size_t output_space = 0;
    output_space += num_nodes(shape) * sizeof(HUFFMAN_CODE);
    output_space += config.estimate_outlier_ratio * sizeof(QUANTIZED_INT);

    estimate_memory_usgae = input_space + output_space;

    log::dbg("Input output space: " +
             std::to_string((double)(input_space + output_space) / 1e9) +
             " GB");

    using HierarchyType = typename OperatorType::HierarchyType;
    size_t hierarchy_space =
        HierarchyType::EstimateMemoryFootprint(shape, config);
    estimate_memory_usgae += hierarchy_space;
    log::dbg("Hierarchy space: " +
             std::to_string((double)hierarchy_space / 1e9) + " GB");

    // For prefetching
    if (enable_prefetch) {
      estimate_memory_usgae *= 2;
    }
    estimate_memory_usgae +=
        OperatorType::EstimateMemoryFootprint(shape, config);
    log::dbg("Operation space: " +
             std::to_string(
                 (double)OperatorType::EstimateMemoryFootprint(shape, config) /
                 1e9) +
             " GB");

    return estimate_memory_usgae;
  }

  // Memory one block may use: the device's free memory, capped by
  // Config::max_memory_footprint.
  size_t available_memory() {
    return std::min((SIZE)DeviceRuntime<DeviceType>::GetAvailableMemory(),
                    config.max_memory_footprint);
  }

  bool need_domain_decomposition(std::vector<SIZE> shape,
                                 bool enable_prefetch) {
    size_t estm = EstimateMemoryFootprint(shape, enable_prefetch);
    size_t aval = available_memory();
    log::dbg("Estimated memory usage: " + std::to_string((double)estm / 1e9) +
             "GB, Available: " + std::to_string((double)aval / 1e9) + "GB");
    return estm >= aval;
  }

  std::vector<SIZE> shape;
  Config config;
  std::vector<T *> coords;
  bool uniform = true;

private:
  static SIZE num_nodes(std::vector<SIZE> shape) {
    SIZE count = 1;
    for (SIZE n : shape) {
      count *= n;
    }
    return count;
  }

  static std::vector<SIZE> contiguous_strides(std::vector<SIZE> shape) {
    std::vector<SIZE> strides(D);
    strides[D - 1] = 1;
    for (int d = (int)D - 2; d >= 0; d--) {
      strides[d] = strides[d + 1] * shape[d + 1];
    }
    return strides;
  }

  static std::string shape_string(std::vector<SIZE> shape) {
    std::stringstream ss;
    for (size_t d = 0; d < shape.size(); d++) {
      ss << (d ? "x" : "") << shape[d];
    }
    return ss.str();
  }

  void check_subdomain_id(SIZE subdomain_id) {
    if (subdomain_id >= _num_subdomains) {
      throw std::runtime_error("DomainDecomposer: subdomain id " +
                               std::to_string(subdomain_id) + " out of range.");
    }
  }

  // Number of blocks of shape `block`, and the shape of the largest one (a
  // merged remainder makes the last block along a dimension the widest),
  // which bounds the memory any block needs.
  SIZE count_blocks(std::vector<SIZE> block) {
    SIZE count = 1;
    for (DIM d = 0; d < D; d++) {
      count *= num_blocks(shape[d], block[d]);
    }
    return count;
  }

  std::vector<SIZE> largest_block_shape(std::vector<SIZE> block) {
    std::vector<SIZE> largest(D);
    for (DIM d = 0; d < D; d++) {
      largest[d] = block_extent(shape[d], block[d],
                                num_blocks(shape[d], block[d]) - 1);
    }
    return largest;
  }

  bool fits_memory(std::vector<SIZE> block) {
    return !need_domain_decomposition(largest_block_shape(block),
                                      count_blocks(block) > 1);
  }

  std::vector<SIZE> auto_block_shape() {
    std::vector<SIZE> block = shape;
    while (!fits_memory(block)) {
      // Halve the largest extent that can still shrink. Ties go to the slowest
      // dimension, whose blocks are the most contiguous in memory.
      int split = -1;
      for (DIM d = 0; d < D; d++) {
        if (block[d] > MIN_BLOCK_EXTENT &&
            (split < 0 || block[d] > block[split])) {
          split = d;
        }
      }
      if (split < 0) {
        throw Exception("Automatic domain decomposition: even the smallest "
                        "blocks (" +
                        shape_string(block) +
                        ") do not fit the memory limit of " +
                        std::to_string(available_memory()) + " bytes.");
      }
      block[split] = std::max((block[split] - 1) / 2 + 1, MIN_BLOCK_EXTENT);
    }
    return block;
  }

  std::vector<SIZE> manual_block_shape() {
    const std::vector<SIZE> &requested = config.domain_decomposition_block_size;
    if (requested.size() != D) {
      throw Exception("Manual domain decomposition: needs a block size for "
                      "each of the " +
                      std::to_string(D) + " dimensions, got " +
                      std::to_string(requested.size()) + ".");
    }
    std::vector<SIZE> block(D);
    for (DIM d = 0; d < D; d++) {
      if (requested[d] < MIN_BLOCK_EXTENT) {
        throw Exception("Manual domain decomposition: block size " +
                        std::to_string(requested[d]) + " along dimension " +
                        std::to_string(d) + " is smaller than " +
                        std::to_string(MIN_BLOCK_EXTENT) + ".");
      }
      block[d] = std::min(requested[d], shape[d]);
    }
    if (!fits_memory(block)) {
      std::vector<SIZE> largest = largest_block_shape(block);
      throw Exception(
          "Manual domain decomposition: blocks of " + shape_string(block) +
          " need about " +
          std::to_string(
              EstimateMemoryFootprint(largest, count_blocks(block) > 1)) +
          " bytes (largest block " + shape_string(largest) + "), more than " +
          "the " + std::to_string(available_memory()) + " bytes available.");
    }
    return block;
  }

  void set_block_shape(std::vector<SIZE> block) {
    _block_shape.resize(D);
    _blocks_per_dim.resize(D);
    _num_subdomains = 1;
    for (DIM d = 0; d < D; d++) {
      _block_shape[d] = std::min(block[d], shape[d]);
      // A dimension narrower than MIN_BLOCK_EXTENT can only be kept whole.
      if (_block_shape[d] < std::min(MIN_BLOCK_EXTENT, shape[d])) {
        throw InvalidDataException(
            "domain decomposition block extent " +
            std::to_string(_block_shape[d]) + " along dimension " +
            std::to_string(d) + " is smaller than " +
            std::to_string(MIN_BLOCK_EXTENT) + ".");
      }
      _blocks_per_dim[d] = num_blocks(shape[d], _block_shape[d]);
      _num_subdomains *= _blocks_per_dim[d];
    }
    if (domain_decomposed()) {
      log::info("DomainDecomposer: decomposed into " +
                std::to_string(_num_subdomains) + " subdomains of " +
                shape_string(_block_shape));
    } else {
      log::info("DomainDecomposer: no decomposition used");
    }
  }

  std::vector<SIZE> _block_shape;
  std::vector<SIZE> _blocks_per_dim;
  SIZE _num_subdomains = 1;
  T *original_data = nullptr;
  std::vector<T *> decomposed_original_data;
  bool keep_original_data_decomposed = false;
};

} // namespace mgard_x

#endif
