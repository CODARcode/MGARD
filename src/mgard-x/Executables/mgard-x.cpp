/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "compress_x.hpp"
#include "mgard-x/Utilities/ErrorCalculator.h"

#include "ArgumentParser.h"

#define OUTPUT_SAFTY_OVERHEAD 1e6

using namespace std::chrono;

void print_usage_message(std::string error) {
  if (error.compare("") != 0) {
    std::cout << mgard_x::log::log_err << error << std::endl;
  }
  printf("Options\n\
\t -z / --compress: compress mode\n\
\t\t -i / --input <path to original data>\n\
\t\t -o / --output <path to compressed data>\n\
\t\t -dt / --data-type <s/single|d/double>: data type (s: single; d:double)\n\
\t\t -dim / --dimension <int>: total number of dimensions\n\
\t\t\t [int]: slowest dimention\n\
\t\t\t [int]: 2nd slowest dimention\n\
\t\t\t  ...\n\
\t\t\t [int]: fastest dimention\n\
\t\t -em / --error-bound-mode <abs|rel>: error bound mode (abs: abolute; rel: relative)\n\
\t\t -e / --error-bound <float>: error bound\n\
\t\t -r / --roi-tolerance-map <path>: path to ROI tolerance map file\n\
\t\t -roi / --enable-roi: enable ROI mode (use per-block tolerances from -r)\n\
\t\t -s / --smoothness <float>: smoothness parameter\n\
\t\t -l / --lossless <huffman|huffman-lz4|lz4|huffman-zstd|blockdelta[-fixed|-delta|-outlier]|zerorle-rans|symbol-rans>: lossless compression\n\
\t\t -d / --device <auto|serial|cuda|hip>: device type\n\
\t\t (optional) -hh / --hybrid: use hybrid (block-local + global) hierarchy\n\
\t\t (optional) -ll / --local-levels <int>: number of hybrid block-local refactoring levels (default: 1)\n\
\t\t (optional) -gl / --global-levels <int>: number of global refactoring levels,\n\
\t\t\t multi-dim or the hybrid global stage (default: 3)\n\
\t\t (optional) -pm / --projection-mode <auto|orthogonal|hierarchical>:\n\
\t\t\t transform basis, plain or hybrid (default: auto -- hierarchical\n\
\t\t\t under an L-inf bound, orthogonal otherwise; hierarchical is L-inf only)\n\
\t\t (optional) -nkf / --no-kernel-fusion: run the hybrid local stage as\n\
\t\t\t separate decompose and quantize passes instead of fused kernels\n\
\t\t\t (same reconstruction either way, but slower -- use it to time the\n\
\t\t\t two stages apart). Fusion is on by default.\n\
\t\t (optional) -v / --verbose <0|1|2|3> 0: error; 1: error+info; 2: error+timing; 3: all\n\
\t\t (optional) -w / --warm-up: run a throwaway compress+decompress pass on a small\n\
\t\t\t array first to pay HIP's one-time per-kernel load cost before timing\n\
\n\
\t -x / --decompress: decompress mode\n\
\t\t -i / --input <path to compressed data>\n\
\t\t -o / --output <path to decompressed data>\n\
\t\t -d / --device <auto|serial|cuda|hip>: device type\n\
\t\t (optional) -nkf / --no-kernel-fusion: as above, for the dequantize+\n\
\t\t\t recompose stage\n\
\t\t (optional) -v / --verbose <0|1|2|3> 0: error; 1: error+info; 2: error+timing; 3: all\n");
  exit(0);
}

template <typename T> void min_max(size_t n, T *in_buff) {
  T min = std::numeric_limits<T>::infinity();
  T max = 0;
  for (size_t i = 0; i < n; i++) {
    if (min > in_buff[i]) {
      min = in_buff[i];
    }
    if (max < in_buff[i]) {
      max = in_buff[i];
    }
  }
  printf("Min: %f, Max: %f\n", min, max);
}

template <typename T> size_t readfile(const char *input_file, T *&in_buff) {
  std::cout << mgard_x::log::log_info << "Loading file: " << input_file << "\n";

  FILE *pFile;
  pFile = fopen(input_file, "rb");
  if (pFile == NULL) {
    std::cout << mgard_x::log::log_err << "file open error!\n";
    exit(1);
  }
  fseek(pFile, 0, SEEK_END);
  size_t lSize = ftell(pFile);
  rewind(pFile);
  in_buff = (T *)malloc(lSize);
  lSize = fread(in_buff, 1, lSize, pFile);
  fclose(pFile);
  // min_max(lSize/sizeof(T), in_buff);
  return lSize;
}

template <typename T>
void writefile(const char *output_file, size_t num_bytes, T *out_buff) {
  FILE *file = fopen(output_file, "w");
  fwrite(out_buff, 1, num_bytes, file);
  fclose(file);
}

// ============================
// ROI Block-wise Verification
// ============================

std::vector<mgard_x::SIZE>
LinearToCoord(mgard_x::SIZE linear_idx,
              const std::vector<mgard_x::SIZE> &dims) {
  std::vector<mgard_x::SIZE> coord(dims.size());
  for (int d = dims.size() - 1; d >= 0; --d) {
    coord[d] = linear_idx % dims[d];
    linear_idx /= dims[d];
  }
  return coord;
}

mgard_x::SIZE CoordToLinear(const std::vector<mgard_x::SIZE> &coord,
                            const std::vector<mgard_x::SIZE> &dims) {
  mgard_x::SIZE linear = 0;
  mgard_x::SIZE stride = 1;
  for (int d = dims.size() - 1; d >= 0; --d) {
    linear += coord[d] * stride;
    stride *= dims[d];
  }
  return linear;
}

struct BlockViolation {
  std::vector<mgard_x::SIZE> block_coord; // Block coordinate
  double tolerance;                       // Expected tolerance
  double actual_error;                    // Actual L_inf error in this block
  double violation_ratio;                 // actual_error / tolerance
};

// Block-wise ROI verification
template <typename T>
std::vector<BlockViolation>
verify_roi_blocks(const std::vector<mgard_x::SIZE> &shape, T *original_data,
                  T *decompressed_data, const std::vector<double> &tol_map,
                  enum mgard_x::error_bound_type mode, T global_norm) {

  const mgard_x::SIZE BLOCK_SIZE = 8;
  size_t D = shape.size();

  // Compute number of blocks in each dimension
  std::vector<mgard_x::SIZE> num_blocks(D);
  mgard_x::SIZE total_blocks = 1;
  for (size_t d = 0; d < D; d++) {
    num_blocks[d] = (shape[d] + BLOCK_SIZE - 1) / BLOCK_SIZE;
    total_blocks *= num_blocks[d];
  }

  std::vector<BlockViolation> violations;

  // Iterate over all blocks
  for (mgard_x::SIZE block_idx = 0; block_idx < total_blocks; block_idx++) {
    // Get block coordinate
    std::vector<mgard_x::SIZE> block_coord =
        LinearToCoord(block_idx, num_blocks);

    // Get tolerance for this block
    double block_tol = tol_map[block_idx];

    // Compute block boundaries in data space
    std::vector<mgard_x::SIZE> block_start(D), block_end(D);
    for (size_t d = 0; d < D; d++) {
      block_start[d] = block_coord[d] * BLOCK_SIZE;
      block_end[d] = std::min(block_start[d] + BLOCK_SIZE, shape[d]);
    }

    // Compute L_inf error within this block
    T block_max_error = 0;

    // Iterate over all elements in this block using nested approach
    std::vector<mgard_x::SIZE> elem_coord(D);
    std::function<void(size_t)> iterate_block = [&](size_t dim) {
      if (dim == D) {
        // Compute linear index in data
        mgard_x::SIZE data_idx = CoordToLinear(elem_coord, shape);

        // Compute error for this element
        T error =
            std::abs(original_data[data_idx] - decompressed_data[data_idx]);
        if (mode == mgard_x::error_bound_type::REL) {
          error = error / global_norm;
        }
        block_max_error = std::max(block_max_error, error);
        return;
      }

      for (mgard_x::SIZE i = block_start[dim]; i < block_end[dim]; i++) {
        elem_coord[dim] = i;
        iterate_block(dim + 1);
      }
    };

    iterate_block(0);

    // Check if this block violates its tolerance
    if (block_max_error > block_tol) {
      BlockViolation v;
      v.block_coord = block_coord;
      v.tolerance = block_tol;
      v.actual_error = block_max_error;
      v.violation_ratio = block_max_error / block_tol;
      violations.push_back(v);
    }
  }

  return violations;
}

// Print ROI block-wise statistics (NEW)
template <typename T>
void print_statistics_roi(double s, enum mgard_x::error_bound_type mode,
                          std::vector<mgard_x::SIZE> shape, T *original_data,
                          T *decompressed_data,
                          const std::vector<double> &tol_map,
                          bool normalize_coordinates) {
  const mgard_x::SIZE BLOCK_SIZE = 8;
  size_t D = shape.size();

  // Compute number of blocks
  std::vector<mgard_x::SIZE> num_blocks(D);
  mgard_x::SIZE total_blocks = 1;
  for (size_t d = 0; d < D; d++) {
    num_blocks[d] = (shape[d] + BLOCK_SIZE - 1) / BLOCK_SIZE;
    total_blocks *= num_blocks[d];
  }

  // Verify tol_map size
  if (tol_map.size() != static_cast<size_t>(total_blocks)) {
    std::cout << mgard_x::log::log_err
              << "ROI tolerance map size mismatch: expected " << total_blocks
              << ", got " << tol_map.size() << "\n";
    return;
  }

  std::cout << mgard_x::log::log_info
            << "=== ROI Block-wise Verification ===\n";
  std::cout << mgard_x::log::log_info << "Block size: " << BLOCK_SIZE;
  for (size_t d = 1; d < D; d++)
    std::cout << "x" << BLOCK_SIZE;
  std::cout << "\n";

  std::cout << mgard_x::log::log_info << "Number of blocks: ";
  for (size_t d = 0; d < D; d++) {
    std::cout << num_blocks[d];
    if (d < D - 1)
      std::cout << " x ";
  }
  std::cout << " = " << total_blocks << " total\n";

  // Compute global norm for relative error mode
  T global_norm = 1;
  if (mode == mgard_x::error_bound_type::REL) {
    mgard_x::SIZE n = 1;
    for (size_t d = 0; d < D; d++)
      n *= shape[d];
    global_norm = 0;
    for (mgard_x::SIZE i = 0; i < n; i++) {
      global_norm = std::max(global_norm, std::abs(original_data[i]));
    }
    std::cout << mgard_x::log::log_info
              << "Global L_inf norm: " << std::scientific << global_norm
              << std::defaultfloat << "\n";
  }

  // Perform block-wise verification
  std::vector<BlockViolation> violations = verify_roi_blocks(
      shape, original_data, decompressed_data, tol_map, mode, global_norm);

  mgard_x::SIZE num_violations = violations.size();
  mgard_x::SIZE num_satisfied = total_blocks - num_violations;
  double satisfaction_rate = 100.0 * num_satisfied / total_blocks;

  std::cout << mgard_x::log::log_info << "Blocks satisfied: " << num_satisfied
            << "/" << total_blocks << " (" << std::fixed << std::setprecision(2)
            << satisfaction_rate << "%)\n";
  std::cout << mgard_x::log::log_info << "Blocks violated: " << num_violations
            << "/" << total_blocks << " (" << std::fixed << std::setprecision(2)
            << (100.0 - satisfaction_rate) << "%)\n";
  std::cout << std::defaultfloat;

  if (num_violations == 0) {
    std::cout << mgard_x::log::log_info
              << "\e[32mAll blocks satisfied their tolerances!\e[0m\n";
  } else {
    std::cout << mgard_x::log::log_info << "\e[31mViolated blocks:\e[0m\n";

    // Sort violations by violation ratio (worst first)
    std::sort(violations.begin(), violations.end(),
              [](const BlockViolation &a, const BlockViolation &b) {
                return a.violation_ratio > b.violation_ratio;
              });

    // Print header
    std::cout << mgard_x::log::log_info << std::setw(20) << "Block Coord"
              << std::setw(15) << "Tolerance" << std::setw(15) << "Actual Error"
              << std::setw(10) << "Ratio"
              << "\n";
    std::cout << mgard_x::log::log_info << std::string(60, '-') << "\n";

    // Print all violations
    std::cout << std::scientific << std::setprecision(4);
    for (const auto &v : violations) {
      // Format block coordinate
      std::ostringstream coord_str;
      coord_str << "(";
      for (size_t d = 0; d < v.block_coord.size(); d++) {
        coord_str << v.block_coord[d];
        if (d < v.block_coord.size() - 1)
          coord_str << ",";
      }
      coord_str << ")";

      std::cout << mgard_x::log::log_info << std::setw(20) << coord_str.str()
                << std::setw(15) << v.tolerance << std::setw(15)
                << v.actual_error << std::setw(10) << std::fixed
                << std::setprecision(2) << v.violation_ratio << "x\n";
      std::cout << std::scientific << std::setprecision(4);
    }
    std::cout << std::defaultfloat;

    // Print worst violation summary
    const auto &worst = violations[0];
    std::ostringstream worst_coord;
    worst_coord << "(";
    for (size_t d = 0; d < worst.block_coord.size(); d++) {
      worst_coord << worst.block_coord[d];
      if (d < worst.block_coord.size() - 1)
        worst_coord << ",";
    }
    worst_coord << ")";
    std::cout << mgard_x::log::log_info << "Worst violation: block "
              << worst_coord.str() << " with " << std::scientific
              << worst.actual_error << " > " << worst.tolerance << " ("
              << std::fixed << std::setprecision(2) << worst.violation_ratio
              << "x)\n";
    std::cout << std::defaultfloat;
  }

  // Also print global statistics
  std::cout << mgard_x::log::log_info << "=== Global Statistics ===\n";
  mgard_x::SIZE n = 1;
  for (size_t d = 0; d < shape.size(); d++)
    n *= shape[d];

  std::cout << mgard_x::log::log_info
            << "MSE: " << mgard_x::MSE(n, original_data, decompressed_data)
            << "\n";
  std::cout << mgard_x::log::log_info
            << "PSNR: " << mgard_x::PSNR(n, original_data, decompressed_data)
            << "\n";
}

// ROI vs background error breakdown for standalone decompression
template <typename T>
void print_decompress_roi_statistics(std::vector<mgard_x::SIZE> shape,
                                     T *original_data, T *decompressed_data,
                                     const std::vector<double> &tol_map,
                                     enum mgard_x::error_bound_type mode) {

  const mgard_x::SIZE BLOCK_SIZE = 8;
  size_t D = shape.size();

  std::vector<mgard_x::SIZE> num_blocks(D);
  mgard_x::SIZE total_blocks = 1;
  for (size_t d = 0; d < D; d++) {
    num_blocks[d] = (shape[d] + BLOCK_SIZE - 1) / BLOCK_SIZE;
    total_blocks *= num_blocks[d];
  }

  // Compute global norm for REL mode
  T global_norm = 1;
  if (mode == mgard_x::error_bound_type::REL) {
    mgard_x::SIZE n = 1;
    for (size_t d = 0; d < D; d++)
      n *= shape[d];
    for (mgard_x::SIZE i = 0; i < n; i++)
      global_norm = std::max(global_norm, std::abs(original_data[i]));
    std::cout << mgard_x::log::log_info
              << "Global L_inf norm: " << std::scientific << global_norm
              << std::defaultfloat << "\n";
  }

  // Identify ROI tolerance (minimum) vs background (maximum)
  double min_tol = *std::min_element(tol_map.begin(), tol_map.end());
  double max_tol = *std::max_element(tol_map.begin(), tol_map.end());
  double split = (min_tol + max_tol) * 0.5;

  struct GroupStats {
    mgard_x::SIZE total = 0, satisfied = 0;
    double sum_error = 0, max_error = 0, max_ratio = 0;
  };
  GroupStats roi_stats, bg_stats;

  for (mgard_x::SIZE block_idx = 0; block_idx < total_blocks; block_idx++) {
    std::vector<mgard_x::SIZE> block_coord =
        LinearToCoord(block_idx, num_blocks);
    double block_tol = tol_map[block_idx];
    bool is_roi = (block_tol <= split);

    std::vector<mgard_x::SIZE> block_start(D), block_end(D);
    for (size_t d = 0; d < D; d++) {
      block_start[d] = block_coord[d] * BLOCK_SIZE;
      block_end[d] = std::min(block_start[d] + BLOCK_SIZE, shape[d]);
    }

    T block_max_error = 0;
    std::vector<mgard_x::SIZE> elem_coord(D);
    std::function<void(size_t)> iterate = [&](size_t dim) {
      if (dim == D) {
        mgard_x::SIZE idx = CoordToLinear(elem_coord, shape);
        T err = std::abs(original_data[idx] - decompressed_data[idx]);
        if (mode == mgard_x::error_bound_type::REL)
          err /= global_norm;
        block_max_error = std::max(block_max_error, err);
        return;
      }
      for (mgard_x::SIZE i = block_start[dim]; i < block_end[dim]; i++) {
        elem_coord[dim] = i;
        iterate(dim + 1);
      }
    };
    iterate(0);

    GroupStats &g = is_roi ? roi_stats : bg_stats;
    g.total++;
    g.sum_error += block_max_error;
    g.max_error = std::max(g.max_error, (double)block_max_error);
    if (block_max_error <= block_tol) {
      g.satisfied++;
    } else {
      g.max_ratio = std::max(g.max_ratio, (double)block_max_error / block_tol);
    }
  }

  auto print_group = [&](const char *label, const GroupStats &g, double tol) {
    if (g.total == 0)
      return;
    double avg_err = g.sum_error / g.total;
    double sat_pct = 100.0 * g.satisfied / g.total;
    std::cout << mgard_x::log::log_info << "--- " << label
              << " (tolerance=" << std::scientific << tol
              << ", blocks=" << g.total << ") ---\n"
              << std::defaultfloat;
    std::cout << mgard_x::log::log_info << "  Satisfied: " << g.satisfied << "/"
              << g.total << " (" << std::fixed << std::setprecision(2)
              << sat_pct << "%)\n";
    std::cout << mgard_x::log::log_info
              << "  Avg block L_inf error: " << std::scientific << avg_err
              << "\n";
    std::cout << mgard_x::log::log_info
              << "  Max block L_inf error: " << std::scientific << g.max_error
              << "\n";
    if (g.satisfied < g.total)
      std::cout << mgard_x::log::log_info
                << "  Worst violation ratio: " << std::fixed
                << std::setprecision(2) << g.max_ratio << "x\n";
    std::cout << std::defaultfloat;
  };

  std::cout << mgard_x::log::log_info
            << "=== Decompression ROI Error Verification ===\n";
  print_group("ROI blocks", roi_stats, min_tol);
  print_group("Background blocks", bg_stats, max_tol);
  std::cout << mgard_x::log::log_info << "Total blocks: " << total_blocks
            << " (ROI=" << roi_stats.total << ", BG=" << bg_stats.total
            << ")\n";
}

template <typename T>
void print_statistics(double s, enum mgard_x::error_bound_type mode,
                      std::vector<mgard_x::SIZE> shape, T *original_data,
                      T *decompressed_data, T tol, bool normalize_coordinates) {
  mgard_x::SIZE n = 1;
  for (mgard_x::DIM d = 0; d < shape.size(); d++)
    n *= shape[d];
  T actual_error = 0.0;
  std::cout << std::scientific;
  if (s == std::numeric_limits<T>::infinity()) {
    actual_error =
        mgard_x::L_inf_error(n, original_data, decompressed_data, mode);
    if (mode == mgard_x::error_bound_type::ABS) {
      std::cout << mgard_x::log::log_info
                << "Absoluate L_inf error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    } else if (mode == mgard_x::error_bound_type::REL) {
      std::cout << mgard_x::log::log_info
                << "Relative L_inf error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    }
  } else {
    actual_error = mgard_x::L_2_error(shape, original_data, decompressed_data,
                                      mode, normalize_coordinates);
    if (mode == mgard_x::error_bound_type::ABS) {
      std::cout << mgard_x::log::log_info
                << "Absoluate L_2 error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    } else if (mode == mgard_x::error_bound_type::REL) {
      std::cout << mgard_x::log::log_info
                << "Relative L_2 error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    }
  }

  std::cout << mgard_x::log::log_info
            << "MSE: " << mgard_x::MSE(n, original_data, decompressed_data)
            << "\n";
  std::cout << std::defaultfloat;
  std::cout << mgard_x::log::log_info
            << "PSNR: " << mgard_x::PSNR(n, original_data, decompressed_data)
            << "\n";
}

int verbose_to_log_level(int verbose) {
  if (verbose == 0) {
    return mgard_x::log::ERR;
  } else if (verbose == 1) {
    return mgard_x::log::ERR | mgard_x::log::INFO;
  } else if (verbose == 2) {
    return mgard_x::log::ERR | mgard_x::log::TIME;
  } else if (verbose == 3) {
    return mgard_x::log::ERR | mgard_x::log::INFO | mgard_x::log::TIME;
  } else if (verbose == 4) {
    return mgard_x::log::ERR | mgard_x::log::INFO | mgard_x::log::TIME |
           mgard_x::log::DBG;
  }
}

template <typename T>
int launch_compress(mgard_x::DIM D, enum mgard_x::data_type dtype,
                    const char *input_file, const char *output_file,
                    std::vector<mgard_x::SIZE> shape, double tol,
                    std::vector<double> tol_map, bool enable_roi, double s,
                    enum mgard_x::error_bound_type mode, std::string lossless,
                    std::string domain_decomposition, mgard_x::SIZE block_size,
                    enum mgard_x::device_type dev_type, int verbose,
                    mgard_x::SIZE max_memory_footprint,
                    mgard_x::SIZE num_local_levels,
                    mgard_x::SIZE num_global_levels, bool use_hybrid,
                    bool warm_up,
                    bool kernel_fusion,
                    mgard_x::compression_projection_mode_type projection_mode) {
  mgard_x::Config config;
  config.log_level = verbose_to_log_level(verbose);
  config.fuse_decompose_quantize = kernel_fusion;
  config.fuse_dequantize_recompose = kernel_fusion;
  // Hybrid (block-local + global) hierarchy decomposition is opt-in via
  // -hh/--hybrid; the default remains the standard multi-dim decomposition.
  if (use_hybrid) {
    config.decomposition = mgard_x::decomposition_type::Hybrid;
  } else {
    config.decomposition = mgard_x::decomposition_type::MultiDim;
  }
  config.num_local_refactoring_level = num_local_levels;
  config.num_global_refactoring_level = num_global_levels;
  config.projection_mode = projection_mode;

  // Switch for ROI
  config.enable_roi = enable_roi;
  if (enable_roi) {
    config.roi_tolerance_map = tol_map;
  }

  if (!enable_roi && tol <= 0) {
    std::cout << mgard_x::log::log_err
              << "Error tolerance (-e) is required when not using ROI mode\n";
    exit(-1);
  }

  // config.compressor = mgard_x::compressor_type::ZFP;

  if (domain_decomposition == "block") {
    config.domain_decomposition = mgard_x::domain_decomposition_type::Block;
    config.block_size = block_size;
  } else {
    config.domain_decomposition = mgard_x::domain_decomposition_type::MaxDim;
  }

  config.cpu_mode = mgard_x::cpu_parallelization_mode::INTRA_BLOCK;

  // config.domain_decomposition = mgard_x::domain_decomposition_type::Variable;
  config.domain_decomposition_dim = 0;
  // NYX
  // config.domain_d  ecomposition_sizes = {512, 512};

  // config.domain_decomposition_sizes = {512, 512, 512, 512};
  // config.domain_decomposition_sizes = {2048};
  // config.domain_decomposition_sizes = {128, 248, 315, 348, 384, 424, 201};
  // config.domain_decomposition_sizes = std::vector<mgard_x::SIZE>(128, 16);
  // config.domain_decomposition_sizes = std::vector<mgard_x::SIZE>(128, 16);

  // XGC
  // config.domain_decomposition_sizes = {312, 312, 312, 312};
  // config.domain_decomposition_sizes = {1248};
  // config.domain_decomposition_sizes = {156, 283, 514, 295};
  // config.domain_decomposition_sizes = std::vector<mgard_x::SIZE>(96, 13);

  // E3SM
  // config.domain_decomposition_sizes = {720, 720, 720, 720};
  // config.domain_decomposition_sizes = {180, 368, 463, 529, 605, 692, 43};
  // config.domain_decomposition_sizes = std::vector<mgard_x::SIZE>(192, 15);

  config.estimate_outlier_ratio = 1.0;

  config.dev_type = dev_type;
  config.auto_pin_host_buffers = true;
  config.max_memory_footprint = max_memory_footprint;
  // config.huff_dict_size = 32768;
  // config.huff_dict_size = 16384;
  config.huff_dict_size = 8192;
  // config.huff_dict_size = 4096;
  // config.huff_dict_size = 2048;
  config.adjust_shape = false;
  config.auto_cache_release = false;

  if (lossless == "huffman") {
    config.lossless = mgard_x::lossless_type::Huffman;
  } else if (lossless == "huffman-lz4") {
    config.lossless = mgard_x::lossless_type::Huffman_LZ4;
  } else if (lossless == "lz4") {
    config.lossless = mgard_x::lossless_type::LZ4;
  } else if (lossless == "huffman-zstd") {
    config.lossless = mgard_x::lossless_type::Huffman_Zstd;
  } else if (lossless == "blockdelta" || lossless == "blockdelta-delta") {
    config.lossless = mgard_x::lossless_type::BlockDelta;
    config.block_delta_mode = mgard_x::block_delta_mode_type::Delta;
  } else if (lossless == "blockdelta-fixed") {
    config.lossless = mgard_x::lossless_type::BlockDelta;
    config.block_delta_mode = mgard_x::block_delta_mode_type::Fixed;
  } else if (lossless == "blockdelta-outlier") {
    config.lossless = mgard_x::lossless_type::BlockDelta;
    config.block_delta_mode = mgard_x::block_delta_mode_type::Outlier;
  } else if (lossless == "zerorle-rans") {
    config.lossless = mgard_x::lossless_type::ZeroRLE_Rans;
  } else if (lossless == "symbol-rans") {
    config.lossless = mgard_x::lossless_type::SymbolRans;
  }

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  T *original_data = (T *)malloc(original_size * sizeof(T));
  size_t in_size = 0;
  if (std::string(input_file).compare("random") == 0) {
    in_size = original_size * sizeof(T);
    srand(7117);
    T c = 0;
    for (size_t i = 0; i < original_size; i++) {
      original_data[i] = rand() % 10 + 1;
    }
  } else {
    T *file_data;
    in_size = readfile(input_file, file_data);

    size_t loaded_size = 0;
    while (loaded_size < original_size) {
      // std::cout << "copy input\n";
      std::memcpy(original_data + loaded_size, file_data,
                  std::min(in_size / sizeof(T), original_size - loaded_size) *
                      sizeof(T));
      loaded_size += std::min(in_size / sizeof(T), original_size - loaded_size);
    }
    in_size = loaded_size * sizeof(T);
  }
  if (in_size != original_size * sizeof(T)) {
    std::cout << mgard_x::log::log_warn << "input file size mismatch "
              << in_size << " vs. " << original_size * sizeof(T) << "!\n";
  }

  // HIP pays a one-time cold-start cost (~10-50ms) the first time each
  // distinct kernel template is launched in a process (lazy code-object
  // loading), which otherwise gets fully attributed to whichever pipeline
  // stage happens to launch that kernel first. Stages launched many times
  // per compress() call (decomposition) amortize it away; single-shot
  // stages (quantization, most Huffman kernels) pay it in full. Run a
  // throwaway pass on a small array of the same dtype/config first so the
  // real, timed run below only measures steady-state performance.
  if (warm_up && !enable_roi) {
    std::vector<mgard_x::SIZE> warmup_shape(D);
    for (mgard_x::DIM i = 0; i < D; i++) {
      warmup_shape[i] = std::min(shape[i], (mgard_x::SIZE)33);
    }
    size_t warmup_size = 1;
    for (mgard_x::DIM i = 0; i < D; i++)
      warmup_size *= warmup_shape[i];
    T *warmup_data = (T *)malloc(warmup_size * sizeof(T));
    for (size_t i = 0; i < warmup_size; i++)
      warmup_data[i] = (T)(i % 10 + 1);
    size_t warmup_compressed_size = warmup_size * sizeof(T) * 2;
    void *warmup_compressed_data = (void *)malloc(warmup_compressed_size);
    void *warmup_decompressed_data = malloc(warmup_size * sizeof(T));

    mgard_x::Config warmup_config = config;
    warmup_config.log_level = mgard_x::log::ERR;
    mgard_x::compress(D, dtype, warmup_shape, tol, s, mode, warmup_data,
                      warmup_compressed_data, warmup_compressed_size,
                      warmup_config, false);
    mgard_x::decompress(warmup_compressed_data, warmup_compressed_size,
                        warmup_decompressed_data, warmup_config, false);

    free(warmup_data);
    free(warmup_compressed_data);
    free(warmup_decompressed_data);
  }

  size_t compressed_size = original_size * sizeof(T) * 2;
  void *compressed_data = (void *)malloc(compressed_size);
  mgard_x::pin_memory(original_data, original_size * sizeof(T), config);
  mgard_x::pin_memory(compressed_data, compressed_size, config);
  mgard_x::compress_status_type ret;
  ret = mgard_x::compress(D, dtype, shape, tol, s, mode, original_data,
                          compressed_data, compressed_size, config, true);
  if (ret != mgard_x::compress_status_type::Success) {
    throw std::runtime_error("Compression failed");
  }
  writefile(output_file, compressed_size, compressed_data);
  std::cout << mgard_x::log::log_info << "Compression ratio: "
            << (double)original_size * sizeof(T) / compressed_size << "\n";

  void *decompressed_data = malloc(original_size * sizeof(T));
  mgard_x::pin_memory(decompressed_data, original_size * sizeof(T), config);
  mgard_x::decompress(compressed_data, compressed_size, decompressed_data,
                      config, true);

  if (config.enable_roi) {
    print_statistics_roi<T>(s, mode, shape, original_data,
                            (T *)decompressed_data, tol_map,
                            config.normalize_coordinates);
  } else {
    print_statistics<T>(s, mode, shape, original_data, (T *)decompressed_data,
                        tol, config.normalize_coordinates);
  }

  mgard_x::unpin_memory(decompressed_data, config);
  free(decompressed_data);

  mgard_x::unpin_memory(original_data, config);
  mgard_x::unpin_memory(compressed_data, config);
  free(original_data);
  free(compressed_data);
  return 0;
}

// Decompression parameters the user may override on the command line. The
// hybrid (BlockMGARD) parameters now travel in the file header, so every field
// here is unset by default and the metadata drives decompression; a field is
// only applied when the corresponding flag was actually passed. Overriding is
// kept for debugging a file whose header disagrees with the data.
struct DecompressOverrides {
  bool has_local_levels = false;
  mgard_x::SIZE num_local_levels = 0;
  bool has_global_levels = false;
  mgard_x::SIZE num_global_levels = 0;
  bool has_roi = false;
  bool enable_roi = false;
  // Also used, independently of any override, as the reference map for the
  // optional -orig block-error report.
  std::vector<double> tol_map;
};

int launch_decompress(
    const char *input_file, const char *output_file,
    enum mgard_x::device_type dev_type, int verbose, bool kernel_fusion,
    const DecompressOverrides &overrides, const char *original_file = nullptr,
    enum mgard_x::error_bound_type ebtype = mgard_x::error_bound_type::ABS) {
  mgard_x::Config config;
  config.log_level = verbose_to_log_level(verbose);
  config.fuse_decompose_quantize = kernel_fusion;
  config.fuse_dequantize_recompose = kernel_fusion;
  config.dev_type = dev_type;
  config.auto_pin_host_buffers = true;
  config.auto_cache_release = true;
  // Leave the hybrid fields at their defaults unless explicitly overridden:
  // decompress() restores them from the header, and overwriting them here with
  // guesses is exactly the bug this replaces.
  if (overrides.has_local_levels) {
    config.num_local_refactoring_level = overrides.num_local_levels;
  }
  if (overrides.has_global_levels) {
    config.num_global_refactoring_level = overrides.num_global_levels;
  }
  if (overrides.has_roi) {
    config.enable_roi = overrides.enable_roi;
    if (overrides.enable_roi) {
      config.roi_tolerance_map = overrides.tol_map;
    }
  }
  const std::vector<double> &tol_map = overrides.tol_map;

  mgard_x::SERIALIZED_TYPE *compressed_data;
  size_t compressed_size = readfile(input_file, compressed_data);
  std::vector<mgard_x::SIZE> shape;
  mgard_x::data_type dtype;
  void *decompressed_data;

  mgard_x::decompress(compressed_data, compressed_size, decompressed_data,
                      shape, dtype, config, false);

  int elem_size = 0;
  if (dtype == mgard_x::data_type::Double) {
    elem_size = 8;
  } else if (dtype == mgard_x::data_type::Float) {
    elem_size = 4;
  }

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < shape.size(); i++) {
    original_size *= shape[i];
  }

  writefile(output_file, original_size * elem_size, decompressed_data);

  // Block-wise error verification. Purely diagnostic and independent of how
  // the file was decompressed, so it runs whenever the user supplied both the
  // original data and a reference tolerance map (-orig and -r).
  if (original_file != nullptr && !tol_map.empty()) {
    void *orig_raw;
    size_t orig_bytes = readfile(original_file, orig_raw);
    if (orig_bytes == original_size * elem_size) {
      if (dtype == mgard_x::data_type::Float) {
        print_decompress_roi_statistics<float>(shape, (float *)orig_raw,
                                               (float *)decompressed_data,
                                               tol_map, ebtype);
      } else if (dtype == mgard_x::data_type::Double) {
        print_decompress_roi_statistics<double>(shape, (double *)orig_raw,
                                                (double *)decompressed_data,
                                                tol_map, ebtype);
      }
      free(orig_raw);
    } else {
      std::cout << mgard_x::log::log_warn
                << "Original file size mismatch, skipping verification\n";
    }
  }

  delete[] compressed_data;
  return 0;
}

bool try_compression(int argc, char *argv[]) {
  if (!has_arg(argc, argv, "-z", "--compress"))
    return false;
  mgard_x::log::info("mode: compress", true);
  std::string input_file =
      get_arg<std::string>(argc, argv, "Original data", "-i", "--input");
  std::string output_file =
      get_arg<std::string>(argc, argv, "Compressed data", "-o", "--output");
  enum mgard_x::data_type dtype = get_data_type(argc, argv);
  std::vector<mgard_x::SIZE> shape =
      get_args<mgard_x::SIZE>(argc, argv, "Dimensions", "-dim", "--dimension");
  enum mgard_x::error_bound_type mode =
      get_error_bound_mode(argc, argv); // REL or ABS
  double tol = -1.0;
  if (has_arg(argc, argv, "-e", "--error-bound")) {
    tol = get_arg<double>(argc, argv, "Error bound", "-e", "--error-bound");
  }
  bool enable_roi = has_arg(argc, argv, "-roi", "--enable-roi");
  std::vector<double> tol_map;
  if (has_arg(argc, argv, "-r", "--roi-tolerance-map")) {
    std::string roi_file = get_arg<std::string>(argc, argv, "ROI tolerance map",
                                                "-r", "--roi-tolerance-map");

    double *roi_map_buffer;
    size_t roi_map_bytes = readfile(roi_file.c_str(), roi_map_buffer);
    size_t roi_map_size = roi_map_bytes / sizeof(double);
    tol_map.resize(roi_map_size);
    for (size_t i = 0; i < roi_map_size; i++) {
      tol_map[i] = static_cast<double>(roi_map_buffer[i]);
    }
    free(roi_map_buffer);

    size_t expected_roi_map_size = 1;
    for (mgard_x::DIM i = 0; i < shape.size(); i++) {
      expected_roi_map_size *= (shape[i] + 8 - 1) / 8;
    }
    if (tol_map.size() != expected_roi_map_size) {
      std::cout << mgard_x::log::log_warn << "ROI map size mismatch: expected "
                << expected_roi_map_size << ", got " << tol_map.size() << "\n";
    }
  }

  if (enable_roi && tol_map.empty()) {
    std::cout << mgard_x::log::log_err
              << "--enable-roi requires -r/--roi-tolerance-map\n";
    exit(-1);
  }
  if (!enable_roi && tol <= 0) {
    std::cout << mgard_x::log::log_err
              << "-e/--tolerance is required when not using ROI mode\n";
    exit(-1);
  }
  double s = get_arg<double>(argc, argv, "Smoothness", "-s", "--smoothness");
  std::string lossless =
      get_arg<std::string>(argc, argv, "Lossless", "-l", "--lossless");
  enum mgard_x::device_type dev_type = get_device_type(argc, argv);
  int verbose = 0;
  if (has_arg(argc, argv, "-v", "--verbose")) {
    verbose = get_arg<int>(argc, argv, "Verbose", "-v", "--verbose");
  }
  bool warm_up = has_arg(argc, argv, "-w", "--warm-up");
  mgard_x::SIZE max_memory_footprint =
      std::numeric_limits<mgard_x::SIZE>::max();
  if (has_arg(argc, argv, "-m", "--max-memory")) {
    max_memory_footprint = (mgard_x::SIZE)get_arg<double>(
        argc, argv, "Max memory", "-m", "--max-memory");
  }

  mgard_x::SIZE num_local_levels =
      mgard_x::Config().num_local_refactoring_level;
  if (has_arg(argc, argv, "-ll", "--local-levels")) {
    num_local_levels = get_arg<mgard_x::SIZE>(argc, argv, "Local levels", "-ll",
                                              "--local-levels");
  }

  bool use_hybrid = has_arg(argc, argv, "-hh", "--hybrid");

  mgard_x::compression_projection_mode_type projection_mode =
      mgard_x::compression_projection_mode_type::Auto;
  if (has_arg(argc, argv, "-pm", "--projection-mode")) {
    std::string value = get_arg<std::string>(argc, argv, "Projection mode",
                                             "-pm", "--projection-mode");
    if (value == "auto") {
      projection_mode = mgard_x::compression_projection_mode_type::Auto;
    } else if (value == "orthogonal") {
      projection_mode = mgard_x::compression_projection_mode_type::Orthogonal;
    } else if (value == "hierarchical") {
      projection_mode = mgard_x::compression_projection_mode_type::Hierarchical;
    } else {
      std::cout << mgard_x::log::log_err
                << "--projection-mode must be auto, orthogonal, or "
                   "hierarchical\n";
      exit(-1);
    }
  }

  // Fusion is on by default; the flag selects the separate-pass path.
  bool kernel_fusion = !has_arg(argc, argv, "-nkf", "--no-kernel-fusion");

  mgard_x::SIZE num_global_levels =
      mgard_x::Config().num_global_refactoring_level;
  if (has_arg(argc, argv, "-gl", "--global-levels")) {
    num_global_levels = get_arg<mgard_x::SIZE>(argc, argv, "Global levels",
                                               "-gl", "--global-levels");
  }

  std::string domain_decomposition = "max-dim";
  mgard_x::SIZE block_size = 0;
  if (has_arg(argc, argv, "-dd", "--domain-decomposition")) {
    domain_decomposition = get_arg<std::string>(
        argc, argv, "Domain decomposition", "-dd", "--domain-decomposition");
    if (domain_decomposition == "block") {
      block_size = get_arg<mgard_x::SIZE>(argc, argv, "Block size", "-dd-size",
                                          "--domain-decomposition-size");
    }
  }

  if (dtype == mgard_x::data_type::Double) {
    launch_compress<double>(
        shape.size(), dtype, input_file.c_str(), output_file.c_str(), shape,
        tol, tol_map, enable_roi, s, mode, lossless, domain_decomposition,
        block_size, dev_type, verbose, max_memory_footprint, num_local_levels,
        num_global_levels, use_hybrid, warm_up, kernel_fusion, projection_mode);
  } else if (dtype == mgard_x::data_type::Float) {
    launch_compress<float>(
        shape.size(), dtype, input_file.c_str(), output_file.c_str(), shape,
        tol, tol_map, enable_roi, s, mode, lossless, domain_decomposition,
        block_size, dev_type, verbose, max_memory_footprint, num_local_levels,
        num_global_levels, use_hybrid, warm_up, kernel_fusion, projection_mode);
  }
  mgard_x::release_cache(mgard_x::Config());
  return true;
}

bool try_decompression(int argc, char *argv[]) {
  if (!has_arg(argc, argv, "-x", "--decompress"))
    return false;
  mgard_x::log::info("mode: decompress", true);
  std::string input_file =
      get_arg<std::string>(argc, argv, "Compressed data", "-i", "--input");
  std::string output_file =
      get_arg<std::string>(argc, argv, "Decompressed data", "-o", "--output");
  enum mgard_x::device_type dev_type = get_device_type(argc, argv);
  int verbose = 0;
  if (has_arg(argc, argv, "-v", "--verbose")) {
    verbose = get_arg<int>(argc, argv, "Verbose", "-v", "--verbose");
  }
  // All of these are optional overrides: the hybrid parameters are restored
  // from the file header, so a plain "mgard-x -x -i f.mgard -o f.raw" now
  // decompresses a BlockMGARD file correctly with no extra flags.
  DecompressOverrides overrides;
  if (has_arg(argc, argv, "-roi", "--enable-roi")) {
    overrides.has_roi = true;
    overrides.enable_roi = true;
  }
  if (has_arg(argc, argv, "-r", "--roi-tolerance-map")) {
    std::string roi_file = get_arg<std::string>(argc, argv, "ROI tolerance map",
                                                "-r", "--roi-tolerance-map");
    double *roi_map_buffer;
    size_t roi_map_bytes = readfile(roi_file.c_str(), roi_map_buffer);
    size_t roi_map_size = roi_map_bytes / sizeof(double);
    overrides.tol_map.resize(roi_map_size);
    for (size_t i = 0; i < roi_map_size; i++) {
      overrides.tol_map[i] = static_cast<double>(roi_map_buffer[i]);
    }
    free(roi_map_buffer);
  }
  if (has_arg(argc, argv, "-ll", "--local-levels")) {
    overrides.has_local_levels = true;
    overrides.num_local_levels = get_arg<mgard_x::SIZE>(
        argc, argv, "Local levels", "-ll", "--local-levels");
  }
  if (has_arg(argc, argv, "-gl", "--global-levels")) {
    overrides.has_global_levels = true;
    overrides.num_global_levels = get_arg<mgard_x::SIZE>(
        argc, argv, "Global levels", "-gl", "--global-levels");
  }
  // Optional: original data file for error verification
  std::string original_file;
  if (has_arg(argc, argv, "-orig", "--original-data")) {
    original_file = get_arg<std::string>(argc, argv, "Original data", "-orig",
                                         "--original-data");
  }
  enum mgard_x::error_bound_type ebtype = mgard_x::error_bound_type::REL;
  if (has_arg(argc, argv, "-em", "--error-bound-mode")) {
    ebtype = get_error_bound_mode(argc, argv);
  }
  bool kernel_fusion = !has_arg(argc, argv, "-nkf", "--no-kernel-fusion");
  launch_decompress(input_file.c_str(), output_file.c_str(), dev_type, verbose,
                    kernel_fusion, overrides,
                    original_file.empty() ? nullptr : original_file.c_str(),
                    ebtype);
  mgard_x::release_cache(mgard_x::Config());
  return true;
}

int main(int argc, char *argv[]) {
  if (!try_compression(argc, argv) && !try_decompression(argc, argv)) {
    print_usage_message("");
  }
  return 0;
}
