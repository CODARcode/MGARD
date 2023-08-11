/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_HYBRID_HIERARCHY_LINEAR_QUANTIZATION_TEMPLATE
#define MGARD_X_HYBRID_HIERARCHY_LINEAR_QUANTIZATION_TEMPLATE

#include "../RuntimeX/RuntimeX.h"
#include "LinearQuantization.hpp"
#include "QuantizationInterface.hpp"

namespace mgard_x {

template <DIM D, typename T, typename Q, OPTION OP, typename DeviceType>
class QuantizeLevelFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT QuantizeLevelFunctor() {}
  MGARDX_CONT
  QuantizeLevelFunctor(T quantizer, SubArray<1, T, DeviceType> v,
                       SubArray<1, Q, DeviceType> quantized_v,
                       bool prep_huffman, SIZE dict_size,
                       ATOMIC_IDX outlier_idx_hierarchy_offset,
                       SubArray<1, ATOMIC_IDX, DeviceType> outlier_count,
                       SubArray<1, ATOMIC_IDX, DeviceType> outlier_indexes,
                       SubArray<1, QUANTIZED_INT, DeviceType> outliers)
      : quantizer(quantizer), v(v), quantized_v(quantized_v),
        prep_huffman(prep_huffman), dict_size(dict_size),
        outlier_count(outlier_count), outlier_indexes(outlier_indexes),
        outliers(outliers) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    idx = FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX() +
          FunctorBase<DeviceType>::GetThreadIdX();
    if (idx < v.shape(0)) {
      T t = *v(idx);
      Q quantized_data;
      T volume = 1;
      if constexpr (OP == MGARDX_QUANTIZE) {
        if (sizeof(T) == sizeof(double))
          quantized_data = copysign((T)0.5 + fabs(t * quantizer * volume), t);
        else if (sizeof(T) == sizeof(float))
          quantized_data = copysign((T)0.5 + fabsf(t * quantizer * volume), t);
        if (prep_huffman) {
          quantized_data += dict_size / 2;
          if (quantized_data >= 0 && quantized_data < dict_size) {
            // do nothing
          } else {
            ATOMIC_IDX outlier_write_offset =
                Atomic<ATOMIC_IDX, AtomicGlobalMemory, AtomicDeviceScope,
                       DeviceType>::Add(outlier_count((IDX)0), (ATOMIC_IDX)1);

            ATOMIC_IDX outlier_idx = idx + outlier_idx_hierarchy_offset;
            // Avoid out of range error
            // If we have too much outlier than our allocation
            // we return the true outlier_count and do quanziation again
            if (outlier_write_offset < outlier_indexes.shape(0)) {
              *outlier_indexes(outlier_write_offset) = outlier_idx;
              *outliers(outlier_write_offset) = quantized_data;
            }
            quantized_data = 0;
          }
        }
        // store quantized value
        *quantized_v(idx) = quantized_data;

      } else if constexpr (OP == MGARDX_DEQUANTIZE) {
        // read quantized value
        quantized_data = *quantized_v(idx);
        if (prep_huffman) {
          quantized_data -= dict_size / 2;
        }
        *v(idx) = (quantizer * volume) * (T)quantized_data;
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  SIZE idx;
  T quantizer;
  SubArray<1, T, DeviceType> v;
  SubArray<1, Q, DeviceType> quantized_v;
  bool prep_huffman;
  SIZE dict_size;
  ATOMIC_IDX outlier_idx_hierarchy_offset;
  SubArray<1, ATOMIC_IDX, DeviceType> outlier_count;
  SubArray<1, ATOMIC_IDX, DeviceType> outlier_indexes;
  SubArray<1, QUANTIZED_INT, DeviceType> outliers;
};

template <DIM D, typename T, typename Q, OPTION OP, typename DeviceType>
class QuantizeLevelKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lwpk";
  MGARDX_CONT
  QuantizeLevelKernel(T quantizer, SubArray<1, T, DeviceType> v,
                      SubArray<1, Q, DeviceType> quantized_v, bool prep_huffman,
                      SIZE dict_size, ATOMIC_IDX outlier_idx_hierarchy_offset,
                      SubArray<1, ATOMIC_IDX, DeviceType> outlier_count,
                      SubArray<1, ATOMIC_IDX, DeviceType> outlier_indexes,
                      SubArray<1, QUANTIZED_INT, DeviceType> outliers)
      : quantizer(quantizer), v(v), quantized_v(quantized_v),
        prep_huffman(prep_huffman), dict_size(dict_size),
        outlier_count(outlier_count), outlier_indexes(outlier_indexes),
        outliers(outliers) {}

  MGARDX_CONT Task<QuantizeLevelFunctor<D, T, Q, OP, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = QuantizeLevelFunctor<D, T, Q, OP, DeviceType>;
    FunctorType functor(quantizer, v, quantized_v, prep_huffman, dict_size,
                        outlier_idx_hierarchy_offset, outlier_count,
                        outlier_indexes, outliers);

    SIZE total_thread_z = 1;
    SIZE total_thread_y = 1;
    SIZE total_thread_x = v.shape(0);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1],
    // shape.dataHost()[0]); PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  T quantizer;
  SubArray<1, T, DeviceType> v;
  SubArray<1, Q, DeviceType> quantized_v;
  bool prep_huffman;
  SIZE dict_size;
  ATOMIC_IDX outlier_idx_hierarchy_offset;
  SubArray<1, ATOMIC_IDX, DeviceType> outlier_count;
  SubArray<1, ATOMIC_IDX, DeviceType> outlier_indexes;
  SubArray<1, QUANTIZED_INT, DeviceType> outliers;
};

template <DIM D, typename T, typename Q, typename DeviceType>
class HybridHierarchyLinearQuantizer
    : public QuantizationInterface<D, T, Q, DeviceType> {
public:
  HybridHierarchyLinearQuantizer() : initialized(false) {}

  HybridHierarchyLinearQuantizer(Hierarchy<D, T, DeviceType> &hierarchy,
                                 Config config)
      : initialized(true), hierarchy(&hierarchy), config(config),
        global_quantizer(hierarchy, config) {

    coarse_shape = hierarchy.level_shape(hierarchy.l_target());
    // If we do at least one level of local refactoring
    if (config.num_local_refactoring_level > 0) {
      for (int l = 0; l < config.num_local_refactoring_level; l++) {
        SIZE last_level_size = 1, curr_level_size = 1;
        for (DIM d = 0; d < D; d++) {
          last_level_size *= coarse_shape[d];
          coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 5;
          curr_level_size *= coarse_shape[d];
        }
        coarse_shapes.push_back(coarse_shape);
        coarse_num_elems.push_back(last_level_size);
        local_coeff_size.push_back(last_level_size - curr_level_size);
      }
    }

    global_hierarchy = Hierarchy<D, T, DeviceType>(coarse_shape, config);
    global_quantizer =
        LinearQuantizer<D, T, Q, DeviceType>(global_hierarchy, config);
  }

  void Adapt(Hierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;
    coarse_shape = hierarchy.level_shape(hierarchy.l_target());
    coarse_shapes.clear();
    coarse_num_elems.clear();
    local_coeff_size.clear();
    // If we do at least one level of local refactoring
    if (config.num_local_refactoring_level > 0) {
      for (int l = 0; l < config.num_local_refactoring_level; l++) {
        SIZE last_level_size = 1, curr_level_size = 1;
        for (DIM d = 0; d < D; d++) {
          last_level_size *= coarse_shape[d];
          coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 5;
          curr_level_size *= coarse_shape[d];
        }
        coarse_shapes.push_back(coarse_shape);
        coarse_num_elems.push_back(last_level_size);
        local_coeff_size.push_back(last_level_size - curr_level_size);
      }
    }

    global_hierarchy = Hierarchy<D, T, DeviceType>(coarse_shape, config);
    global_quantizer.Adapt(global_hierarchy, config, queue_idx);
  }

  void CalcQuantizers(size_t dof, enum error_bound_type type, T tol, T s,
                      T norm, SIZE global_l_target,
                      SIZE num_local_refactoring_level,
                      enum decomposition_type decomposition, bool reciprocal,
                      T &quantizer, T &coarse_abs_tol) {

    double abs_tol = tol;
    if (type == error_bound_type::REL) {
      abs_tol *= norm;
    }
    abs_tol *= 2;
    SIZE total_num_levels = global_l_target + num_local_refactoring_level + 1;
    // std::cout << "total_num_levels: " << total_num_levels << "\n";
    if (s == std::numeric_limits<T>::infinity()) {
      quantizer = (abs_tol) / (total_num_levels * (1 + std::pow(3, D)));
      // std::cout << "quantizer: " << quantizer << "\n";
      coarse_abs_tol =
          (quantizer * (global_l_target + 1) * (1 + std::pow(3, D))) / 2;
      // std::cout << "coarse_abs_tol: " << coarse_abs_tol << "\n";
      if (reciprocal)
        quantizer = 1.0f / quantizer;
    } else { // s != inf

      log::err("s != inf not supported yet.");
      exit(-1);
      // xin - uniform
      // T C2 = 1 + 3 * std::sqrt(3) / 4;
      // T c = std::sqrt(std::pow(2, D - 2 * s));
      // T cc = (1 - c) / (1 - std::pow(c, l_target + 1));
      // T level_eb = cc * tol / C2;
      // for (int l = 0; l < l_target + 1; l++) {
      //   quantizers[l] = level_eb;
      //   // T c = std::sqrt(std::pow(2, 2*s*l + D * (l_target - l)));
      //   level_eb *= c;
      //   if (reciprocal)
      //     quantizers[l] = 1.0f / quantizers[l];
      // }

      // ben - uniform
      for (int l = 0; l < total_num_levels; l++) {
        quantizer = (abs_tol) / (std::exp2(s * l) * std::sqrt(dof));
        if (reciprocal)
          quantizer = 1.0f / quantizer;
      }
    }
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    Hierarchy<D, T, DeviceType> hierarchy;
    hierarchy.EstimateMemoryFootprint(shape);
    size_t size = 0;
    size += sizeof(T) * (hierarchy->l_target() + 1);
    return size;
  }

  void Quantize(SubArray<D, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<D, Q, DeviceType> quantized_data, int queue_idx) {}

  void Dequantize(SubArray<D, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<D, Q, DeviceType> quantized_data, int queue_idx) {}

  template <typename LosslessCompressorType>
  void Quantize(SubArray<1, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<1, Q, DeviceType> quantized_data,
                LosslessCompressorType &lossless, int queue_idx) {
    bool prep_huffman =
        config.lossless != lossless_type::CPU_Lossless; // always do Huffman

    Array<D, T, DeviceType> coarse_data(coarse_shape, original_data.data());
    Array<D, Q, DeviceType> coarse_quantized_data(coarse_shape,
                                                  quantized_data.data());

    T quantizer, coarse_abs_tol;
    CalcQuantizers(hierarchy->total_num_elems(), ebtype, tol, s, norm,
                   global_hierarchy.l_target(),
                   config.num_local_refactoring_level, config.decomposition,
                   true, quantizer, coarse_abs_tol);

    // Array<D, T, DeviceType> original_data_array(coarse_shape,
    // original_data.data()); Array<1, T, DeviceType>
    // norm_tmp_array({coarse_num_elems[coarse_num_elems.size()-1]}); Array<1,
    // T, DeviceType> norm_array({1});

    // verify_matrix_cuda(coarse_shape[0], coarse_shape[1], coarse_shape[2],
    //               original_data.data(), coarse_shape[0], coarse_shape[1],
    //               coarse_shape[0], "coarse_data", true, false);

    // T coarse_norm = norm_calculator(original_data_array,
    // SubArray(norm_tmp_array),
    //                     SubArray(norm_array), s,
    //                     config.normalize_coordinates);
    // std::cout << "coarse_norm: " << coarse_norm << "\n";
    log::info("coarse_abs_tol: " + std::to_string(coarse_abs_tol));
    log::info("local quantizer: " + std::to_string(quantizer));
    global_quantizer.Quantize(coarse_data, error_bound_type::ABS,
                              coarse_abs_tol, s, norm, coarse_quantized_data,
                              lossless, queue_idx);

    SIZE accumulated_local_coeff_size = 0;
    for (int l = 0; l < config.num_local_refactoring_level; l++) {
      accumulated_local_coeff_size += local_coeff_size[l];
      SubArray<1, T, DeviceType> local_data(
          {local_coeff_size[l]},
          original_data(original_data.shape(0) - accumulated_local_coeff_size));
      SubArray<1, Q, DeviceType> local_quantized_data(
          {local_coeff_size[l]}, quantized_data(quantized_data.shape(0) -
                                                accumulated_local_coeff_size));

      Timer timer;
      if (log::level & log::TIME)
        timer.start();
      bool done_quantization = false;
      DeviceLauncher<DeviceType>::Execute(
          QuantizeLevelKernel<D, T, Q, MGARDX_QUANTIZE, DeviceType>(
              quantizer, local_data, local_quantized_data, prep_huffman,
              config.huff_dict_size, global_hierarchy.total_num_elems(),
              lossless.huffman.workspace.outlier_count_subarray,
              lossless.huffman.workspace.outlier_idx_subarray,
              lossless.huffman.workspace.outlier_subarray),
          queue_idx);

      MemoryManager<DeviceType>::Copy1D(
          &lossless.huffman.workspace.outlier_count,
          lossless.huffman.workspace.outlier_count_subarray.data(), 1,
          queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      if (lossless.huffman.workspace.outlier_count <=
          lossless.huffman.workspace.outlier_subarray.shape(0)) {
        // outlier buffer has sufficient size
        done_quantization = true;
        if (log::level & log::TIME) {
          timer.end();
          timer.print("Quantization");
          log::time("Quantization throughput: " +
                    std::to_string((double)(coarse_num_elems[l] * sizeof(T)) /
                                   timer.get() / 1e9) +
                    " GB/s");
          timer.clear();
        }
        log::info("Outlier ratio: " +
                  std::to_string(lossless.huffman.workspace.outlier_count) +
                  "/" + std::to_string(hierarchy->total_num_elems()) + " (" +
                  std::to_string((double)100 *
                                 lossless.huffman.workspace.outlier_count /
                                 hierarchy->total_num_elems()) +
                  "%)");
      } else {
        log::err("Not enough workspace for outliers.");
        exit(-1);
      }
    }
  }

  template <typename LosslessCompressorType>
  void Dequantize(SubArray<1, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<1, Q, DeviceType> quantized_data,
                  LosslessCompressorType &lossless_compressor, int queue_idx) {}

  bool initialized;
  Hierarchy<D, T, DeviceType> *hierarchy;
  Hierarchy<D, T, DeviceType> global_hierarchy;
  Config config;
  std::vector<SIZE> coarse_shape;
  std::vector<SIZE> coarse_num_elems;
  LinearQuantizer<D, T, Q, DeviceType> global_quantizer;
  std::vector<std::vector<SIZE>> coarse_shapes;
  std::vector<SIZE> local_coeff_size;
};

} // namespace mgard_x

#endif