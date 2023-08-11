/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_DOMAIN_DECOMPOSER_HPP
#define MGARD_X_DOMAIN_DECOMPOSER_HPP

#include "../Config/Config.h"
#include "../Hierarchy/Hierarchy.hpp"

namespace mgard_x {

enum class subdomain_copy_direction : uint8_t {
  OriginalToSubdomain,
  SubdomainToOriginal
};

template <DIM D, typename T, typename OperatorType, typename DeviceType>
class DomainDecomposer {
public:
  size_t EstimateMemoryFootprint(std::vector<SIZE> shape,
                                 bool enable_prefetch) {
    size_t estimate_memory_usgae = 0;

    Array<1, T, DeviceType> array_with_pitch({1});
    size_t pitch_size = array_with_pitch.ld(0) * sizeof(T);

    size_t input_space = roundup((size_t)shape[D - 1] * sizeof(T), pitch_size);
    for (DIM d = 0; d < D - 1; d++) {
      input_space *= shape[d];
    }

    SIZE num_elements = 1;
    for (int i = 0; i < shape.size(); i++)
      num_elements *= shape[i];
    size_t output_space = 0;
    output_space += num_elements * sizeof(HUFFMAN_CODE);
    output_space += config.estimate_outlier_ratio * sizeof(QUANTIZED_INT);

    estimate_memory_usgae = input_space + output_space;

    log::info("Input output space: " +
              std::to_string((double)(input_space + output_space) / 1e9) +
              " GB");

    using HierarchyType = typename OperatorType::HierarchyType;
    HierarchyType hierarchy;
    estimate_memory_usgae += hierarchy.EstimateMemoryFootprint(shape);
    log::info(
        "Hierarchy space: " +
        std::to_string((double)hierarchy.EstimateMemoryFootprint(shape) / 1e9) +
        " GB");

    // For prefetching
    if (enable_prefetch) {
      estimate_memory_usgae *= 2;
    }
    estimate_memory_usgae +=
        OperatorType::EstimateMemoryFootprint(shape, config);
    log::info("Compressor space: " +
              std::to_string(
                  (double)OperatorType::EstimateMemoryFootprint(shape, config) /
                  1e9) +
              " GB");

    return estimate_memory_usgae;
  }

  bool need_domain_decomposition(std::vector<SIZE> shape,
                                 bool enable_prefetch) {
    // using Cache = CompressorCache<D, T, DeviceType, CompressorType>;
    size_t estm = EstimateMemoryFootprint(shape, enable_prefetch);
    size_t aval =
        std::min((SIZE)DeviceRuntime<DeviceType>::GetAvailableMemory(),
                 config.max_memory_footprint);
    log::info("Estimated memory usage: " + std::to_string((double)estm / 1e9) +
              "GB, Available: " + std::to_string((double)aval / 1e9) + "GB");
    bool need = estm >= aval;
    if (need) {
      // Fast copy for domain decomposition need we disable pitched memory
      // allocation
      log::info("ReduceMemoryFootprint set to 1");
      MemoryManager<DeviceType>::ReduceMemoryFootprint = true;
    }
    return need;
  }

  std::vector<SIZE> dim_num_subdomain() {
    std::vector<SIZE> _dim_num_subdomain(D, 1);
    if (config.domain_decomposition == domain_decomposition_type::MaxDim ||
        config.domain_decomposition == domain_decomposition_type::TemporalDim) {
      _dim_num_subdomain[_domain_decomposed_dim] =
          (shape[_domain_decomposed_dim] - 1) / _domain_decomposed_size + 1;
    } else if (config.domain_decomposition ==
               domain_decomposition_type::Block) {
      for (int d = D - 1; d >= 0; d--) {
        _dim_num_subdomain[d] = (shape[d] - 1) / _domain_decomposed_size + 1;
      }
    }
    return _dim_num_subdomain;
  }

  std::vector<SIZE> dim_subdomain_id(int subdomain_id) {
    std::vector<SIZE> _dim_num_subdomain = dim_num_subdomain();
    std::vector<SIZE> _dim_subdomain_id(D);
    for (int d = D - 1; d >= 0; d--) {
      _dim_subdomain_id[d] = subdomain_id % _dim_num_subdomain[d];
      subdomain_id /= _dim_num_subdomain[d];
    }
    return _dim_subdomain_id;
  }

  std::vector<SIZE> dim_subdomain_offset(int subdomain_id) {
    std::vector<SIZE> _dim_subdomain_id = dim_subdomain_id(subdomain_id);
    std::vector<SIZE> _dim_subdomain_offset(D);
    for (int d = D - 1; d >= 0; d--) {
      _dim_subdomain_offset[d] = _dim_subdomain_id[d] * _domain_decomposed_size;
    }
    return _dim_subdomain_offset;
  }

  std::vector<SIZE> subdomain_shape(int subdomain_id) {
    if (subdomain_id >= _num_subdomains) {
      log::err("DomainDecomposer::subdomain_shape wrong subdomain_id.");
      exit(-1);
    }
    if (!_domain_decomposed) {
      return shape;
    } else {
      if (config.domain_decomposition == domain_decomposition_type::MaxDim ||
          config.domain_decomposition ==
              domain_decomposition_type::TemporalDim) {
        if (subdomain_id <
            shape[_domain_decomposed_dim] / _domain_decomposed_size) {
          std::vector<SIZE> chunck_shape = shape;
          chunck_shape[_domain_decomposed_dim] = _domain_decomposed_size;
          return chunck_shape;
        } else {
          SIZE leftover_dim_size =
              shape[_domain_decomposed_dim] % _domain_decomposed_size;
          std::vector<SIZE> leftover_shape = shape;
          leftover_shape[_domain_decomposed_dim] = leftover_dim_size;
          return leftover_shape;
        }
      } else if (config.domain_decomposition ==
                 domain_decomposition_type::Block) {
        std::vector<SIZE> chunck_shape(D);
        std::vector<SIZE> _dim_subdomain_id = dim_subdomain_id(subdomain_id);
        for (int d = D - 1; d >= 0; d--) {
          if (_dim_subdomain_id[d] < shape[d] / _domain_decomposed_size) {
            chunck_shape[d] = _domain_decomposed_size;
          } else {
            chunck_shape[d] = shape[d] % _domain_decomposed_size;
          }
        }
        return chunck_shape;
      } else {
        log::err("Wrong domain decomposition type.");
        exit(-1);
        return shape;
      }
    }
  }

  SIZE subdomain_compressed_buffer_size(int subdomain_id) {
    std::vector<SIZE> shape = subdomain_shape(subdomain_id);
    SIZE num_elements = 1;
    for (int i = 0; i < shape.size(); i++)
      num_elements *= shape[i];
    SIZE size = 0;
    size += num_elements * sizeof(HUFFMAN_CODE);
    size += config.estimate_outlier_ratio * sizeof(QUANTIZED_INT);
    return size;
  }

  bool generate_max_dim_domain_decomposition_strategy(
      std::vector<SIZE> shape, DIM &_domain_decomposed_dim,
      SIZE &_domain_decomposed_size) {
    // determine max dimension
    SIZE max_dim = 0;
    for (DIM d = 0; d < D; d++) {
      if (shape[d] > max_dim) {
        max_dim = shape[d];
        _domain_decomposed_dim = d;
      }
    }

    // domain decomposition strategy
    std::vector<SIZE> chunck_shape = shape;

    SIZE curr_num_subdomains = 1;

    bool need_prefetch = false;
    // Then check if each chunk can fit into device memory
    while (need_domain_decomposition(chunck_shape, need_prefetch)) {
      // Divide by 2 and round up
      chunck_shape[_domain_decomposed_dim] =
          (chunck_shape[_domain_decomposed_dim] - 1) / 2 + 1;

      curr_num_subdomains = (shape[_domain_decomposed_dim] - 1) /
                                chunck_shape[_domain_decomposed_dim] +
                            1;
      need_prefetch = curr_num_subdomains > 1;
    }
    _domain_decomposed_size = chunck_shape[_domain_decomposed_dim];
    return true;
  }

  bool generate_temporal_dim_domain_decomposition_strategy(
      std::vector<SIZE> shape, DIM &_domain_decomposed_dim,
      SIZE &_domain_decomposed_size) {

    std::vector<SIZE> chunck_shape = shape;

    // First divide by temporal dimension
    chunck_shape[_domain_decomposed_dim] = _domain_decomposed_size;

    SIZE curr_num_subdomains = (shape[_domain_decomposed_dim] - 1) /
                                   chunck_shape[_domain_decomposed_dim] +
                               1;

    // Need prefetch if there are more subdomains than devices
    bool need_prefetch = curr_num_subdomains > 1;
    // Then check if each chunk can fit into device memory
    while (need_domain_decomposition(chunck_shape, need_prefetch)) {
      // Divide by 2 and round up
      chunck_shape[_domain_decomposed_dim] =
          (chunck_shape[_domain_decomposed_dim] - 1) / 2 + 1;

      curr_num_subdomains = (shape[_domain_decomposed_dim] - 1) /
                                chunck_shape[_domain_decomposed_dim] +
                            1;
      need_prefetch = curr_num_subdomains > 1;
    }
    _domain_decomposed_size = chunck_shape[_domain_decomposed_dim];
    return true;
  }

  bool
  generate_block_domain_decomposition_strategy(std::vector<SIZE> shape,
                                               SIZE &_domain_decomposed_size) {
    std::vector<SIZE> chunck_shape(D, _domain_decomposed_size);

    int curr_num_subdomains = 1;
    for (DIM d = 0; d < D; d++) {
      curr_num_subdomains *= (shape[d] - 1) / _domain_decomposed_size + 1;
    }

    // Need prefetch if there are more subdomains than devices
    bool need_prefetch = curr_num_subdomains > 1;
    // Then check if each chunk can fit into device memory
    while (need_domain_decomposition(chunck_shape, need_prefetch)) {
      // Divide by 2 and round up
      _domain_decomposed_size = (_domain_decomposed_size - 1) / 2 + 1;
      chunck_shape = std::vector<SIZE>(D, _domain_decomposed_size);

      curr_num_subdomains = 1;
      for (DIM d = 0; d < D; d++) {
        curr_num_subdomains *= (shape[d] - 1) / _domain_decomposed_size + 1;
      }

      need_prefetch = curr_num_subdomains > 1;
    }
    return true;
  }

  Hierarchy<D, T, DeviceType> subdomain_hierarchy(int subdomain_id) {
    if (uniform) {
      return Hierarchy<D, T, DeviceType>(subdomain_shape(subdomain_id), config);
    } else {
      if (config.domain_decomposition == domain_decomposition_type::MaxDim ||
          config.domain_decomposition ==
              domain_decomposition_type::TemporalDim) {
        std::vector<T *> chunck_coords = coords;
        std::vector<SIZE> chunck_shape = subdomain_shape(subdomain_id);
        T *decompose_dim_coord = new T[chunck_shape[_domain_decomposed_dim]];
        MemoryManager<DeviceType>::Copy1D(
            decompose_dim_coord,
            coords[_domain_decomposed_dim] +
                subdomain_id * _domain_decomposed_size,
            chunck_shape[_domain_decomposed_dim], 0);
        DeviceRuntime<DeviceType>::SyncQueue(0);
        chunck_coords[_domain_decomposed_dim] = decompose_dim_coord;
        Hierarchy<D, T, DeviceType> hierarchy(chunck_shape, chunck_coords,
                                              config);
        delete[] decompose_dim_coord;
        return hierarchy;
      } else if (config.domain_decomposition ==
                 domain_decomposition_type::Block) {
        std::vector<T *> chunck_coords(D);
        std::vector<SIZE> chunck_shape = subdomain_shape(subdomain_id);
        std::vector<SIZE> _dim_subdomain_id = dim_subdomain_id(subdomain_id);
        for (int d = D - 1; d >= 0; d--) {
          chunck_coords[d] = new T[chunck_shape[d]];
          MemoryManager<DeviceType>::Copy1D(
              chunck_coords[d],
              coords[d] + _dim_subdomain_id[d] * _domain_decomposed_size,
              chunck_shape[d], 0);
          DeviceRuntime<DeviceType>::SyncQueue(0);
        }
        Hierarchy<D, T, DeviceType> hierarchy(chunck_shape, chunck_coords,
                                              config);
        for (int d = D - 1; d >= 0; d--)
          delete[] chunck_coords[d];
        return hierarchy;
      } else {
        log::err("Wrong domain decomposition type.");
        exit(-1);
      }
    }
  }

  DomainDecomposer() : _domain_decomposed(false) {}

  // Find domain decomposion method
  DomainDecomposer(std::vector<SIZE> shape, Config config)
      : original_data(nullptr), shape(shape), config(config),
        keep_original_data_decomposed(false) {
    if (!need_domain_decomposition(shape, false) &&
        config.domain_decomposition != domain_decomposition_type::Block &&
        config.domain_decomposition != domain_decomposition_type::TemporalDim) {
      this->_domain_decomposed = false;
      this->_domain_decomposed_dim = 0;
      this->_domain_decomposed_size = this->shape[0];
      this->_num_subdomains = 1;
      log::info("DomainDecomposer: no decomposition used");
    } else {
      this->_domain_decomposed = true;
      if (config.domain_decomposition == domain_decomposition_type::MaxDim) {
        generate_max_dim_domain_decomposition_strategy(
            shape, this->_domain_decomposed_dim, this->_domain_decomposed_size);
        this->_num_subdomains = (shape[this->_domain_decomposed_dim] - 1) /
                                    this->_domain_decomposed_size +
                                1;
        log::info(
            "Domain decomposed dim: " + std::to_string(_domain_decomposed_dim) +
            ", Domain decomposed size: " +
            std::to_string(_domain_decomposed_size));
        log::info("DomainDecomposer: decomposed into " +
                  std::to_string(this->_num_subdomains) +
                  " subdomains using MaxDim method");
      } else if (config.domain_decomposition ==
                 domain_decomposition_type::TemporalDim) {
        this->_domain_decomposed_dim = config.temporal_dim;
        this->_domain_decomposed_size =
            std::min(config.temporal_dim_size, shape[config.temporal_dim]);
        generate_temporal_dim_domain_decomposition_strategy(
            shape, this->_domain_decomposed_dim, this->_domain_decomposed_size);
        this->_num_subdomains = (shape[this->_domain_decomposed_dim] - 1) /
                                    this->_domain_decomposed_size +
                                1;
        log::info(
            "Domain decomposed dim: " + std::to_string(_domain_decomposed_dim) +
            ", Domain decomposed size: " +
            std::to_string(_domain_decomposed_size));
        log::info("DomainDecomposer: decomposed into " +
                  std::to_string(this->_num_subdomains) +
                  " subdomains using TemporalDim method");
      } else if (config.domain_decomposition ==
                 domain_decomposition_type::Block) {
        this->_domain_decomposed_size = config.block_size;
        generate_block_domain_decomposition_strategy(
            shape, this->_domain_decomposed_size);
        this->_num_subdomains = 1;
        for (DIM d = 0; d < D; d++) {
          this->_num_subdomains *=
              (shape[d] - 1) / this->_domain_decomposed_size + 1;
        }
        log::info("Domain decomposed size: " +
                  std::to_string(_domain_decomposed_size));
        log::info("DomainDecomposer: decomposed into " +
                  std::to_string(this->_num_subdomains) +
                  " subdomains using Block method");
      } else {
        log::err("Wrong domain decomposition type.");
        exit(-1);
      }
    }

    uniform = true;
  }

  // Find domain decomposion method
  DomainDecomposer(std::vector<SIZE> shape, Config config,
                   std::vector<T *> coords)
      : original_data(nullptr), shape(shape), config(config), coords(coords),
        keep_original_data_decomposed(false) {
    if (!need_domain_decomposition(shape, false) &&
        config.domain_decomposition != domain_decomposition_type::Block) {
      this->_domain_decomposed = false;
      this->_domain_decomposed_dim = 0;
      this->_domain_decomposed_size = this->shape[0];
      this->_num_subdomains = 1;
      log::info("DomainDecomposer: no decomposition used");
    } else {
      this->_domain_decomposed = true;
      if (config.domain_decomposition == domain_decomposition_type::MaxDim) {
        generate_max_dim_domain_decomposition_strategy(
            shape, this->_domain_decomposed_dim, this->_domain_decomposed_size);
        this->_num_subdomains = (shape[this->_domain_decomposed_dim] - 1) /
                                    this->_domain_decomposed_size +
                                1;
        log::info(
            "Domain decomposed dim: " + std::to_string(_domain_decomposed_dim) +
            ", Domain decomposed size: " +
            std::to_string(_domain_decomposed_size));
        log::info("DomainDecomposer: decomposed into " +
                  std::to_string(this->_num_subdomains) +
                  " subdomains using MaxDim method");
      } else if (config.domain_decomposition ==
                 domain_decomposition_type::TemporalDim) {
        this->_domain_decomposed_dim = config.temporal_dim;
        this->_domain_decomposed_size =
            std::min(config.temporal_dim_size, shape[config.temporal_dim]);
        generate_temporal_dim_domain_decomposition_strategy(
            shape, this->_domain_decomposed_dim, this->_domain_decomposed_size);
        this->_num_subdomains = (shape[this->_domain_decomposed_dim] - 1) /
                                    this->_domain_decomposed_size +
                                1;
        log::info(
            "Domain decomposed dim: " + std::to_string(_domain_decomposed_dim) +
            ", Domain decomposed size: " +
            std::to_string(_domain_decomposed_size));
        log::info("DomainDecomposer: decomposed into " +
                  std::to_string(this->_num_subdomains) +
                  " subdomains using TemporalDim method");
      } else if (config.domain_decomposition ==
                 domain_decomposition_type::Block) {
        this->_domain_decomposed_size = config.block_size;
        generate_block_domain_decomposition_strategy(
            shape, this->_domain_decomposed_size);
        this->_num_subdomains = 1;
        for (DIM d = 0; d < D; d++) {
          this->_num_subdomains *=
              (shape[d] - 1) / this->_domain_decomposed_size + 1;
        }
        log::info("Domain decomposed size: " +
                  std::to_string(_domain_decomposed_size));
        log::info("DomainDecomposer: decomposed into " +
                  std::to_string(this->_num_subdomains) +
                  " subdomains using Block method");
      } else {
        log::err("Wrong domain decomposition type.");
        exit(-1);
      }
    }

    uniform = false;
  }

  // Force to use this domain decomposion method
  DomainDecomposer(std::vector<SIZE> shape, bool _domain_decomposed,
                   DIM _domain_decomposed_dim, SIZE _domain_decomposed_size,
                   Config config)
      : original_data(nullptr), shape(shape),
        _domain_decomposed_dim(_domain_decomposed_dim),
        _domain_decomposed_size(_domain_decomposed_size),
        _domain_decomposed(_domain_decomposed), config(config),
        keep_original_data_decomposed(false) {
    if (!this->_domain_decomposed) {
      this->_domain_decomposed_dim = 0;
      this->_domain_decomposed_size = this->shape[0];
      this->_num_subdomains = 1;
      log::info("DomainDecomposer: no decomposition used");
    } else {
      // Fast copy for domain decomposition need we disable pitched memory
      // allocation
      log::info("ReduceMemoryFootprint set to 1");
      MemoryManager<DeviceType>::ReduceMemoryFootprint = true;
      if (config.domain_decomposition == domain_decomposition_type::MaxDim ||
          config.domain_decomposition ==
              domain_decomposition_type::TemporalDim) {
        this->_num_subdomains = (shape[this->_domain_decomposed_dim] - 1) /
                                    this->_domain_decomposed_size +
                                1;
        log::info(
            "Domain decomposed dim: " + std::to_string(_domain_decomposed_dim) +
            ", Domain decomposed size: " +
            std::to_string(_domain_decomposed_size));
        if (config.domain_decomposition == domain_decomposition_type::MaxDim) {
          log::info("DomainDecomposer: decomposed into " +
                    std::to_string(this->_num_subdomains) +
                    " subdomains using MaxDim method");
        } else if (config.domain_decomposition ==
                   domain_decomposition_type::TemporalDim) {
          log::info("DomainDecomposer: decomposed into " +
                    std::to_string(this->_num_subdomains) +
                    " subdomains using TemporalDim method");
        }
      } else if (config.domain_decomposition ==
                 domain_decomposition_type::Block) {
        this->_num_subdomains = 1;
        for (DIM d = 0; d < D; d++) {
          this->_num_subdomains *=
              (shape[d] - 1) / this->_domain_decomposed_size + 1;
        }
        log::info("Domain decomposed size: " +
                  std::to_string(_domain_decomposed_size));
        log::info("DomainDecomposer: decomposed into " +
                  std::to_string(this->_num_subdomains) +
                  " subdomains using Block method");
      } else {
        log::err("Wrong domain decomposition type.");
        exit(-1);
      }
    }

    uniform = true;
  }

  // Force to use this domain decomposion method
  DomainDecomposer(std::vector<SIZE> shape, bool _domain_decomposed,
                   DIM _domain_decomposed_dim, SIZE _domain_decomposed_size,
                   Config config, std::vector<T *> coords)
      : original_data(nullptr), shape(shape),
        _domain_decomposed_dim(_domain_decomposed_dim),
        _domain_decomposed_size(_domain_decomposed_size),
        _domain_decomposed(_domain_decomposed), config(config), coords(coords),
        keep_original_data_decomposed(false) {
    if (!this->_domain_decomposed) {
      this->_domain_decomposed_dim = 0;
      this->_domain_decomposed_size = this->shape[0];
      this->_num_subdomains = 1;
      log::info("DomainDecomposer: no decomposition used");
    } else {
      // Fast copy for domain decomposition need we disable pitched memory
      // allocation
      log::info("ReduceMemoryFootprint set to 1");
      MemoryManager<DeviceType>::ReduceMemoryFootprint = true;
      if (config.domain_decomposition == domain_decomposition_type::MaxDim ||
          config.domain_decomposition ==
              domain_decomposition_type::TemporalDim) {
        this->_num_subdomains = (shape[this->_domain_decomposed_dim] - 1) /
                                    this->_domain_decomposed_size +
                                1;
        log::info(
            "Domain decomposed dim: " + std::to_string(_domain_decomposed_dim) +
            ", Domain decomposed size: " +
            std::to_string(_domain_decomposed_size));
        if (config.domain_decomposition == domain_decomposition_type::MaxDim) {
          log::info("DomainDecomposer: decomposed into " +
                    std::to_string(this->_num_subdomains) +
                    " subdomains using MaxDim method");
        } else if (config.domain_decomposition ==
                   domain_decomposition_type::TemporalDim) {
          log::info("DomainDecomposer: decomposed into " +
                    std::to_string(this->_num_subdomains) +
                    " subdomains using TemporalDim method");
        }
      } else if (config.domain_decomposition ==
                 domain_decomposition_type::Block) {
        this->_num_subdomains = 1;
        for (DIM d = 0; d < D; d++) {
          this->_num_subdomains *=
              (shape[d] - 1) / this->_domain_decomposed_size + 1;
        }
        log::info("Domain decomposed size: " +
                  std::to_string(_domain_decomposed_size));
        log::info("DomainDecomposer: decomposed into " +
                  std::to_string(this->_num_subdomains) +
                  " subdomains using Block method");
      } else {
        log::err("Wrong domain decomposition type.");
        exit(-1);
      }
    }

    uniform = false;
  }

  void calc_domain_decompose_parameter(SIZE subdomain_id,
                                       std::vector<SIZE> subdomain_shape,
                                       SIZE &subdomain_ld, SIZE &original_ld,
                                       SIZE &n1, SIZE &n2) {
    SIZE decomposed_size;
    if (subdomain_id <
        shape[_domain_decomposed_dim] / _domain_decomposed_size) {
      decomposed_size = _domain_decomposed_size;
    } else {
      decomposed_size = shape[_domain_decomposed_dim] % _domain_decomposed_size;
    }

    std::vector<SIZE> original_shape = original_data_shape(subdomain_shape);

    subdomain_ld = subdomain_shape[_domain_decomposed_dim]; // decomposed_size;
    for (int d = D - 1; d > (int)_domain_decomposed_dim; d--) {
      subdomain_ld *= subdomain_shape[d];
    }
    // std::cout << "dst_ld: " << dst_ld << "\n";
    original_ld = 1;
    for (int d = D - 1; d >= (int)_domain_decomposed_dim; d--) {
      original_ld *= original_shape[d];
    }
    // std::cout << "src_ld: " << src_ld << "\n";
    n1 = subdomain_shape[_domain_decomposed_dim];
    for (int d = D - 1; d > (int)_domain_decomposed_dim; d--) {
      n1 *= subdomain_shape[d];
    }
    // std::cout << "n1: " << n1 << "\n";
    n2 = 1;
    for (int d = 0; d < (int)_domain_decomposed_dim; d++) {
      n2 *= subdomain_shape[d];
    }
    // std::cout << "n2: " << n2 << "\n";
  }

  SIZE calc_offset(std::vector<SIZE> shape,
                   std::vector<SIZE> dim_subdomain_offset) {
    SIZE curr_stride = 1;
    SIZE ret_idx = 0;
    for (int d = D - 1; d >= 0; d--) {
      ret_idx += dim_subdomain_offset[d] * curr_stride;
      curr_stride *= shape[d];
    }
    return ret_idx;
  }

  bool check_shape(Array<D, T, DeviceType> &subdomain_data,
                   std::vector<SIZE> shape) {
    if (subdomain_data.data() == nullptr) {
      return false;
    }
    for (DIM d = 0; d < D; d++) {
      if (subdomain_data.shape(d) != shape[d]) {
        return false;
      }
    }
    return true;
  }

  T *original_data_ptr(int subdomain_id) {
    if (config.domain_decomposition == domain_decomposition_type::MaxDim ||
        config.domain_decomposition == domain_decomposition_type::TemporalDim) {
      if (!keep_original_data_decomposed) {
        SIZE offset = _domain_decomposed_size;
        for (int d = D - 1; d > (int)_domain_decomposed_dim; d--) {
          offset *= shape[d];
        }
        return original_data + offset * subdomain_id;
      } else {
        return decomposed_original_data[subdomain_id];
      }
    } else if (config.domain_decomposition ==
               domain_decomposition_type::Block) {
      if (!keep_original_data_decomposed) {
        return original_data +
               calc_offset(shape, dim_subdomain_offset(subdomain_id));
      } else {
        return decomposed_original_data[subdomain_id];
      }
    } else {
      log::err("Wrong domain decomposition type.");
      exit(-1);
    }
  }

  std::vector<SIZE> original_data_shape(std::vector<SIZE> subdomain_shape) {
    if (!keep_original_data_decomposed) {
      return shape;
    } else {
      return subdomain_shape;
    }
  }

  void copy_subdomain(Array<D, T, DeviceType> &subdomain_data, int subdomain_id,
                      enum subdomain_copy_direction direction, int queue_idx) {
    if (subdomain_id >= _num_subdomains) {
      log::err("DomainDecomposer::copy_subdomain wrong subdomain_id.");
      exit(-1);
    }

    if (!_domain_decomposed) {
      // if (keep_original_data_decomposed) {
      //   log::err("Do not support restoring to decomposed data when no domain
      //   decomposition was used."); exit(-1);
      // }

      if (direction == subdomain_copy_direction::OriginalToSubdomain) {
        subdomain_data.resize(shape);
      }
      SIZE linearized_width = 1;
      for (DIM d = 0; d < D - 1; d++)
        linearized_width *= subdomain_data.shape(d);
      if (direction == subdomain_copy_direction::OriginalToSubdomain) {
        MemoryManager<DeviceType>::CopyND(
            subdomain_data.data(), subdomain_data.ld(D - 1), original_data,
            subdomain_data.shape(D - 1), subdomain_data.shape(D - 1),
            linearized_width, queue_idx);
      } else {
        MemoryManager<DeviceType>::CopyND(
            original_data, subdomain_data.shape(D - 1), subdomain_data.data(),
            subdomain_data.ld(D - 1), subdomain_data.shape(D - 1),
            linearized_width, queue_idx);
      }
    } else {
      // Pitched memory allocation has to be disable for the correctness of the
      // following copies
      assert(MemoryManager<DeviceType>::ReduceMemoryFootprint == true);
      bool pitched = false;

      if (config.domain_decomposition == domain_decomposition_type::MaxDim ||
          config.domain_decomposition ==
              domain_decomposition_type::TemporalDim) {
        if (keep_original_data_decomposed) {
          log::err("Do not support restoring to decomposed data when using "
                   "MaxDim or TemporalDim");
          exit(-1);
        }
        T *data = original_data_ptr(subdomain_id);
        if (direction == subdomain_copy_direction::OriginalToSubdomain) {
          subdomain_data.resize(subdomain_shape(subdomain_id), pitched);
        }
        SIZE subdomain_ld, original_ld, n1, n2;
        // calc_domain_decompose_parameter(subdomain_id, subdomain_data.shape(),
        // subdomain_ld, original_ld, n1, n2); SIZE decomposed_size; if
        // (subdomain_id <
        //     shape[_domain_decomposed_dim] / _domain_decomposed_size) {
        //   decomposed_size = _domain_decomposed_size;
        // } else {
        //   decomposed_size = shape[_domain_decomposed_dim] %
        //   _domain_decomposed_size;
        // }

        std::vector<SIZE> original_shape =
            original_data_shape(subdomain_data.shape());

        subdomain_ld = subdomain_data.shape(_domain_decomposed_dim);
        for (int d = D - 1; d > (int)_domain_decomposed_dim; d--) {
          subdomain_ld *= subdomain_data.shape(d);
        }
        // std::cout << "dst_ld: " << dst_ld << "\n";
        original_ld = 1;
        for (int d = D - 1; d >= (int)_domain_decomposed_dim; d--) {
          original_ld *= original_shape[d];
        }
        // std::cout << "src_ld: " << src_ld << "\n";
        n1 = subdomain_data.shape(_domain_decomposed_dim);
        for (int d = D - 1; d > (int)_domain_decomposed_dim; d--) {
          n1 *= subdomain_data.shape(d);
        }
        // std::cout << "n1: " << n1 << "\n";
        n2 = 1;
        for (int d = 0; d < (int)_domain_decomposed_dim; d++) {
          n2 *= subdomain_data.shape(d);
        }

        if (direction == subdomain_copy_direction::OriginalToSubdomain) {
          MemoryManager<DeviceType>::CopyND(subdomain_data.data(), subdomain_ld,
                                            data, original_ld, n1, n2,
                                            queue_idx);
        } else {
          MemoryManager<DeviceType>::CopyND(data, original_ld,
                                            subdomain_data.data(), subdomain_ld,
                                            n1, n2, queue_idx);
        }
      } else if (config.domain_decomposition ==
                 domain_decomposition_type::Block) {
        std::vector<SIZE> chunck_shape = subdomain_shape(subdomain_id);
        if (direction == subdomain_copy_direction::OriginalToSubdomain) {
          subdomain_data.resize(chunck_shape, pitched);
        }

        T *data = original_data_ptr(subdomain_id);
        std::vector<SIZE> shape = original_data_shape(subdomain_data.shape());

        if (D == 1) {
          if (direction == subdomain_copy_direction::OriginalToSubdomain) {
            MemoryManager<DeviceType>::CopyND(
                subdomain_data.data(), subdomain_data.ld(D - 1), data,
                shape[D - 1], subdomain_data.shape(D - 1), 1, queue_idx);
          } else {
            MemoryManager<DeviceType>::CopyND(
                data, shape[D - 1], subdomain_data.data(),
                subdomain_data.ld(D - 1), subdomain_data.shape(D - 1), 1,
                queue_idx);
          }
        } else if (D == 2) {
          if (direction == subdomain_copy_direction::OriginalToSubdomain) {
            MemoryManager<DeviceType>::CopyND(
                subdomain_data.data(), subdomain_data.ld(D - 1), data,
                shape[D - 1], subdomain_data.shape(D - 1),
                subdomain_data.shape(D - 2), queue_idx);
          } else {
            MemoryManager<DeviceType>::CopyND(
                data, shape[D - 1], subdomain_data.data(),
                subdomain_data.ld(D - 1), subdomain_data.shape(D - 1),
                subdomain_data.shape(D - 2), queue_idx);
          }
        } else if (D == 3) {
          for (SIZE i = 0; i < subdomain_data.shape(D - 3); i++) {
            if (direction == subdomain_copy_direction::OriginalToSubdomain) {
              MemoryManager<DeviceType>::CopyND(
                  subdomain_data.data() +
                      calc_offset(subdomain_data.shape(), {i, 0, 0}),
                  subdomain_data.ld(D - 1),
                  data + calc_offset(shape, {i, 0, 0}), shape[D - 1],
                  subdomain_data.shape(D - 1), subdomain_data.shape(D - 2),
                  queue_idx);
            } else {
              MemoryManager<DeviceType>::CopyND(
                  data + calc_offset(shape, {i, 0, 0}), shape[D - 1],
                  subdomain_data.data() +
                      calc_offset(subdomain_data.shape(), {i, 0, 0}),
                  subdomain_data.ld(D - 1), subdomain_data.shape(D - 1),
                  subdomain_data.shape(D - 2), queue_idx);
            }
          }
        } else if (D == 4) {
          for (SIZE j = 0; j < subdomain_data.shape(D - 4); j++) {
            for (SIZE i = 0; i < subdomain_data.shape(D - 3); i++) {
              if (direction == subdomain_copy_direction::OriginalToSubdomain) {
                MemoryManager<DeviceType>::CopyND(
                    subdomain_data.data() +
                        calc_offset(subdomain_data.shape(), {j, i, 0, 0}),
                    subdomain_data.ld(D - 1),
                    data + calc_offset(shape, {j, i, 0, 0}), shape[D - 1],
                    subdomain_data.shape(D - 1), subdomain_data.shape(D - 2),
                    queue_idx);
              } else {
                MemoryManager<DeviceType>::CopyND(
                    data + calc_offset(shape, {j, i, 0, 0}), shape[D - 1],
                    subdomain_data.data() +
                        calc_offset(subdomain_data.shape(), {j, i, 0, 0}),
                    subdomain_data.ld(D - 1), subdomain_data.shape(D - 1),
                    subdomain_data.shape(D - 2), queue_idx);
              }
            }
          }
        } else if (D == 5) {
          for (SIZE k = 0; k < subdomain_data.shape(D - 5); k++) {
            for (SIZE j = 0; j < subdomain_data.shape(D - 4); j++) {
              for (SIZE i = 0; i < subdomain_data.shape(D - 3); i++) {
                if (direction ==
                    subdomain_copy_direction::OriginalToSubdomain) {
                  MemoryManager<DeviceType>::CopyND(
                      subdomain_data.data() +
                          calc_offset(subdomain_data.shape(), {k, j, i, 0, 0}),
                      subdomain_data.ld(D - 1),
                      data + calc_offset(shape, {k, j, i, 0, 0}), shape[D - 1],
                      subdomain_data.shape(D - 1), subdomain_data.shape(D - 2),
                      queue_idx);
                } else {
                  MemoryManager<DeviceType>::CopyND(
                      data + calc_offset(shape, {k, j, i, 0, 0}), shape[D - 1],
                      subdomain_data.data() +
                          calc_offset(subdomain_data.shape(), {k, j, i, 0, 0}),
                      subdomain_data.ld(D - 1), subdomain_data.shape(D - 1),
                      subdomain_data.shape(D - 2), queue_idx);
                }
              }
            }
          }
        } else {
          log::err("Copy subdomain does not support higher than 5D data.");
          exit(-1);
        }
      } else {
        log::err("Wrong domain decomposition type.");
        exit(-1);
      }
    }
  }

  bool domain_decomposed() { return _domain_decomposed; }

  DIM domain_decomposed_dim() { return _domain_decomposed_dim; }

  SIZE domain_decomposed_size() { return _domain_decomposed_size; }

  SIZE num_subdomains() { return _num_subdomains; }

  void set_original_data(T *original_data) {
    this->original_data = original_data;
    keep_original_data_decomposed = false;
  }

  void set_decomposed_original_data(std::vector<T *> decomposed_original_data) {
    this->decomposed_original_data = decomposed_original_data;
    original_data = decomposed_original_data[0];
    keep_original_data_decomposed = true;
  }

  std::vector<SIZE> shape;
  bool _domain_decomposed;
  DIM _domain_decomposed_dim;
  SIZE _domain_decomposed_size;
  SIZE _num_subdomains;
  std::vector<Hierarchy<D, T, DeviceType>> subdomain_hierarchies;
  T *original_data;
  std::vector<T *> decomposed_original_data;
  bool keep_original_data_decomposed;
  Config config;
  bool uniform;
  std::vector<T *> coords;
};

} // namespace mgard_x

#endif
