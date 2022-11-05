/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_DOMAIN_DECOMPOSER_HPP
#define MGARD_X_DOMAIN_DECOMPOSER_HPP

namespace mgard_x {

enum class subdomain_copy_direction : uint8_t { OriginalToSubdomain, SubdomainToOriginal };

template <DIM D, typename T, typename OperationType, typename DeviceType> class DomainDecomposer {
public:
  size_t estimate_memory_usgae(std::vector<SIZE> shape, double outlier_ratio,
                               double reduction_ratio, bool enable_prefetch) {
    size_t estimate_memory_usgae = 0;

    Array<1, T, DeviceType> array_with_pitch({1});
    size_t pitch_size = array_with_pitch.ld(0) * sizeof(T);

    Hierarchy<D, T, DeviceType> hierarchy;
    size_t hierarchy_space = hierarchy.estimate_memory_usgae(shape);
    // log::info("hierarchy_space: " +
    //           std::to_string((double)hierarchy_space / 1e9));

    size_t input_space = roundup(shape[D - 1] * sizeof(T), pitch_size);
    for (DIM d = 0; d < D - 1; d++) {
      input_space *= shape[d];
    }

    size_t output_space = (double)input_space * reduction_ratio;

    // log::info("input_space: " + std::to_string((double)input_space / 1e9));

    // CompressionLowLevelWorkspace<D, T, DeviceType> compression_workspace;
    estimate_memory_usgae =
          hierarchy_space + input_space + output_space;
    estimate_memory_usgae +=
        OperationType::EstimateMemoryFootprint(
            shape, hierarchy.l_target(), config.huff_dict_size,
            config.huff_block_size, outlier_ratio);

    // For prefetching
    if (enable_prefetch) {
      estimate_memory_usgae *= 2;
    }

    return estimate_memory_usgae;
  }

  bool need_domain_decomposition(std::vector<SIZE> shape,
                                 bool enable_prefetch) {
    size_t estm = estimate_memory_usgae(shape, 1, 1, enable_prefetch);
    size_t aval = std::min(DeviceRuntime<DeviceType>::GetAvailableMemory(),
                           config.max_memory_footprint);
    log::info("Estimated memory usage: " + std::to_string((double)estm / 1e9) +
              "GB, Available: " + std::to_string((double)aval / 1e9) + "GB");
    return estm >= aval;
  }

  std::vector<SIZE> subdomain_shape(int subdomain_id) {
    if (subdomain_id >= _num_subdomains) {
      log::err("DomainDecomposer::subdomain_shape wrong subdomain_id.");
      exit(-1);
    }
    if (!_domain_decomposed) {
      return shape;
    } else {
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
    }
  }

  bool generate_max_dim_domain_decomposition_strategy(std::vector<SIZE> shape,
                                              DIM &_domain_decomposed_dim,
                                              SIZE &_domain_decomposed_size,
                                              SIZE num_dev) {
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

    // First divide by the number of devices
    chunck_shape[_domain_decomposed_dim] =
        std::ceil((double)chunck_shape[_domain_decomposed_dim] / num_dev);

    SIZE curr_num_subdomains = (shape[_domain_decomposed_dim] - 1) /
                                   chunck_shape[_domain_decomposed_dim] +
                               1;

    // Need prefetch if there are more subdomains than devices
    bool need_prefetch = curr_num_subdomains > num_dev;
    // Then check if each chunk can fit into device memory
    while (need_domain_decomposition(chunck_shape, need_prefetch)) {
      // Divide by 2 and round up
      chunck_shape[_domain_decomposed_dim] =
          (chunck_shape[_domain_decomposed_dim] - 1) / 2 + 1;

      curr_num_subdomains = (shape[_domain_decomposed_dim] - 1) /
                                chunck_shape[_domain_decomposed_dim] +
                            1;
      need_prefetch = curr_num_subdomains > num_dev;
    }
    _domain_decomposed_size = chunck_shape[_domain_decomposed_dim];
    log::info(
        "Domain decomposed dim: " + std::to_string(_domain_decomposed_dim) +
        ", Domain decomposed size: " + std::to_string(_domain_decomposed_size));
    return true;
  }

  bool generate_block_domain_decomposition_strategy(std::vector<SIZE> shape,
                                              SIZE &_domain_decomposed_size,
                                              SIZE num_dev) {
    // determine max dimension
    SIZE max_dim = 0;
    for (DIM d = 0; d < D; d++) {
      if (shape[d] > max_dim) {
        max_dim = shape[d];
        _domain_decomposed_dim = d;
      }
    }

    // domain decomposition strategy
    std::vector<SIZE> chunck_shape(_domain_decomposed_size, D);

    int curr_num_subdomains = 1;
    for (DIM d = 0; d < D; d++){
      curr_num_subdomains *= (shape[d] - 1) / _domain_decomposed_size + 1;
    }

    // Need prefetch if there are more subdomains than devices
    bool need_prefetch = curr_num_subdomains > num_dev;
    // Then check if each chunk can fit into device memory
    while (need_domain_decomposition(chunck_shape, need_prefetch)) {
      // Divide by 2 and round up
      chunck_shape[_domain_decomposed_dim] =
          (chunck_shape[_domain_decomposed_dim] - 1) / 2 + 1;

      curr_num_subdomains = 1;
      for (DIM d = 0; d < D; d++){
        curr_num_subdomains *= (shape[d] - 1) / _domain_decomposed_size + 1;
      }

      need_prefetch = curr_num_subdomains > num_dev;
    }
    _domain_decomposed_size = chunck_shape[_domain_decomposed_dim];
    log::info("Domain decomposed size: " + std::to_string(_domain_decomposed_size));
    return true;
  }

  Hierarchy<D, T, DeviceType> subdomain_hierarchy(int subdomain_id) {
    if (uniform) {
      return Hierarchy<D, T, DeviceType>(subdomain_shape(subdomain_id), config);
    } else {
      std::vector<T *> chunck_coords = coords;
      std::vector<SIZE> shape = subdomain_shape(subdomain_id);
      T *decompose_dim_coord = new T[shape[_domain_decomposed_dim]];
      MemoryManager<DeviceType>::Copy1D(decompose_dim_coord,
                                        coords[_domain_decomposed_dim] +
                                            subdomain_id *
                                                _domain_decomposed_size,
                                        shape[_domain_decomposed_dim], 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      chunck_coords[_domain_decomposed_dim] = decompose_dim_coord;
      delete[] decompose_dim_coord;
      return Hierarchy<D, T, DeviceType>(shape, chunck_coords, config);
    }
  }

  std::vector<SIZE> subdomain_ids_for_device(int dev_id) {
    // Must assign consecutive subdomains to a device
    std::vector<SIZE> subdomain_ids;
    SIZE start = (dev_id * num_subdomains()) / num_devices();
    SIZE end = ((dev_id + 1) * num_subdomains()) / num_devices() - 1;
    for (SIZE subdomain_id = start; subdomain_id <= end; subdomain_id++) {
      subdomain_ids.push_back(subdomain_id);
    }
    return subdomain_ids;
  }

  SIZE total_subdomain_size_for_device(int dev_id) {
    SIZE total_size = 0;
    std::vector<SIZE> subdomain_ids = subdomain_ids_for_device(dev_id);
    for (int i = 0; i < subdomain_ids.size(); i++) {
      std::vector<SIZE> shape = subdomain_shape(subdomain_ids[i]);
      int num_elems = 1;
      for (DIM d = 0; d < D; d++) {
        num_elems *= shape[d];
      }
      total_size += num_elems * sizeof(T);
    }
    return total_size;
  }

  DomainDecomposer() : _domain_decomposed(false) {}

  // Find domain decomposion method
  DomainDecomposer(T *original_data, std::vector<SIZE> shape, int _num_devices,
                   Config config)
      : original_data(original_data), shape(shape), _num_devices(_num_devices),
        config(config) {
    if (!need_domain_decomposition(shape, false) && this->_num_devices == 1 &&
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
            shape, this->_domain_decomposed_dim, this->_domain_decomposed_size,
            this->_num_devices);
        this->_num_subdomains = (shape[this->_domain_decomposed_dim] - 1) /
                                this->_domain_decomposed_size +
                            1;
        log::info("DomainDecomposer: decomposed into " + std::to_string(this->_num_subdomains) + " subdomains using MaxDim method");
      } else if (config.domain_decomposition == domain_decomposition_type::Block) {
        this->_domain_decomposed_size = config.block_size;
        generate_block_domain_decomposition_strategy( shape,
                                              this->_domain_decomposed_size,
                                              this->_num_devices);
        this->_num_subdomains = 1;
        for (DIM d = 0; d < D; d++){
          this->_num_subdomains *= (shape[d] - 1) / config.block_size + 1;
        }
        log::info("DomainDecomposer: decomposed into " + std::to_string(this->_num_subdomains) + " subdomains using Block method");
      } else {
        log::err ("Wrong domain decomposition type.");
        exit(-1);
      }
    }
    
    uniform = true;
  }

  // Find domain decomposion method
  DomainDecomposer(T *original_data, std::vector<SIZE> shape, int _num_devices,
                   Config config, std::vector<T *> coords)
      : original_data(original_data), shape(shape), _num_devices(_num_devices),
        config(config), coords(coords) {
    if (!need_domain_decomposition(shape, false) && this->_num_devices == 1 && 
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
            shape, this->_domain_decomposed_dim, this->_domain_decomposed_size,
            this->_num_devices);
        this->_num_subdomains = (shape[this->_domain_decomposed_dim] - 1) /
                                  this->_domain_decomposed_size +
                              1;
        log::info("DomainDecomposer: decomposed into " + std::to_string(this->_num_subdomains) + " subdomains using MaxDim method");
      } else if (config.domain_decomposition == domain_decomposition_type::Block) {
        this->_domain_decomposed_size = config.block_size;
        generate_block_domain_decomposition_strategy( shape,
                                              this->_domain_decomposed_size,
                                              this->_num_devices);
        this->_num_subdomains = 1;
        for (DIM d = 0; d < D; d++){
          this->_num_subdomains *= (shape[d] - 1) / config.block_size + 1;
        }
        log::info("DomainDecomposer: decomposed into " + std::to_string(this->_num_subdomains) + " subdomains using Block method");
      } else {
        log::err ("Wrong domain decomposition type.");
        exit(-1);
      }
    }
    
    uniform = false;
  }

  // Force to use this domain decomposion method
  DomainDecomposer(T *original_data, std::vector<SIZE> shape, int _num_devices,
                   bool _domain_decomposed, DIM _domain_decomposed_dim,
                   SIZE _domain_decomposed_size, Config config)
      : original_data(original_data), shape(shape), _num_devices(_num_devices),
        _domain_decomposed_dim(_domain_decomposed_dim),
        _domain_decomposed_size(_domain_decomposed_size),
        _domain_decomposed(_domain_decomposed), config(config) {
    if (!this->_domain_decomposed) {
      this->_domain_decomposed_dim = 0;
      this->_domain_decomposed_size = this->shape[0];
      this->_num_subdomains = 1;
      log::info("DomainDecomposer: no decomposition used");
    } else {
      if (config.domain_decomposition == domain_decomposition_type::MaxDim) {
        this->_num_subdomains = (shape[this->_domain_decomposed_dim] - 1) /
                                  this->_domain_decomposed_size +
                              1;
        log::info("DomainDecomposer: decomposed into " + std::to_string(this->_num_subdomains) + " subdomains using MaxDim method");
      } else if (config.domain_decomposition == domain_decomposition_type::Block) {
        this->_domain_decomposed_size = config.block_size;
        this->_num_subdomains = 1;
        for (DIM d = 0; d < D; d++){
          this->_num_subdomains *= (shape[d] - 1) / config.block_size + 1;
        }
        log::info("DomainDecomposer: decomposed into " + std::to_string(this->_num_subdomains) + " subdomains using Block method");
      } else {
        log::err ("Wrong domain decomposition type.");
        exit(-1);
      }
    }

    uniform = true;
  }

  // Force to use this domain decomposion method
  DomainDecomposer(T *original_data, std::vector<SIZE> shape, int _num_devices,
                   bool _domain_decomposed, DIM _domain_decomposed_dim,
                   SIZE _domain_decomposed_size, Config config,
                   std::vector<T *> coords)
      : original_data(original_data), shape(shape), _num_devices(_num_devices),
        _domain_decomposed_dim(_domain_decomposed_dim),
        _domain_decomposed_size(_domain_decomposed_size),
        _domain_decomposed(_domain_decomposed), config(config), coords(coords) {
    if (!this->_domain_decomposed) {
      this->_domain_decomposed_dim = 0;
      this->_domain_decomposed_size = this->shape[0];
      this->_num_subdomains = 1;
      log::info("DomainDecomposer: no decomposition used");
    } else {
      if (config.domain_decomposition == domain_decomposition_type::MaxDim) {
        this->_num_subdomains = (shape[this->_domain_decomposed_dim] - 1) /
                                  this->_domain_decomposed_size +
                              1;
        log::info("DomainDecomposer: decomposed into " + std::to_string(this->_num_subdomains) + " subdomains using MaxDim method");
      } else if (config.domain_decomposition == domain_decomposition_type::Block) {
        this->_domain_decomposed_size = config.block_size;
        this->_num_subdomains = 1;
        for (DIM d = 0; d < D; d++){
          this->_num_subdomains *= (shape[d] - 1) / config.block_size + 1;
        }
        log::info("DomainDecomposer: decomposed into " + std::to_string(this->_num_subdomains) + " subdomains using Block method");
      } else {
        log::err ("Wrong domain decomposition type.");
        exit(-1);
      }
    }

    uniform = false;
  }

  void calc_domain_decompose_parameter(std::vector<SIZE> shape,
                                       DIM _domain_decomposed_dim,
                                       SIZE _domain_decomposed_size,
                                       SIZE &dst_ld, SIZE &src_ld, SIZE &n1,
                                       SIZE &n2) {
    dst_ld = _domain_decomposed_size;
    for (int d = D - 1; d > (int)_domain_decomposed_dim; d--) {
      dst_ld *= shape[d];
    }
    // std::cout << "dst_ld: " << dst_ld << "\n";
    src_ld = 1;
    for (int d = D - 1; d >= (int)_domain_decomposed_dim; d--) {
      src_ld *= shape[d];
    }
    // std::cout << "src_ld: " << src_ld << "\n";
    n1 = _domain_decomposed_size;
    for (int d = D - 1; d > (int)_domain_decomposed_dim; d--) {
      n1 *= shape[d];
    }
    // std::cout << "n1: " << n1 << "\n";
    n2 = 1;
    for (int d = 0; d < (int)_domain_decomposed_dim; d++) {
      n2 *= shape[d];
    }
    // std::cout << "n2: " << n2 << "\n";
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

  void copy_subdomain(Array<D, T, DeviceType> &subdomain_data, int subdomain_id,
                      enum subdomain_copy_direction direction, int queue_idx) {
    if (subdomain_id >= _num_subdomains) {
      log::err("DomainDecomposer::copy_subdomain wrong subdomain_id.");
      exit(-1);
    }

    // Timer timer;
    // if (log::level & log::TIME) {
    //   DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    //   timer.start();
    // }
    if (!_domain_decomposed) {
      SIZE linearized_width = 1;
      for (DIM d = 0; d < D - 1; d++)
        linearized_width *= shape[d];
      if (direction == subdomain_copy_direction::OriginalToSubdomain) {
        subdomain_data = Array<D, T, DeviceType>(shape);
        // subdomain_data.load(original_data);
        MemoryManager<DeviceType>::CopyND(
            subdomain_data.data(), subdomain_data.ld(D - 1), original_data,
            shape[D - 1], shape[D - 1], linearized_width, queue_idx);
      } else {
        MemoryManager<DeviceType>::CopyND(
            original_data, shape[D - 1], subdomain_data.data(),
            subdomain_data.ld(D - 1), shape[D - 1], linearized_width,
            queue_idx);
      }
    } else {
      // Pitched memory allocation has to be disable for the correctness of the
      // following copies
      assert(MemoryManager<DeviceType>::ReduceMemoryFootprint == true);
      bool pitched = false;

      SIZE dst_ld, src_ld, n1, n2;
      calc_domain_decompose_parameter(shape, _domain_decomposed_dim,
                                      _domain_decomposed_size, dst_ld, src_ld,
                                      n1, n2);
      T *data = original_data + n1 * subdomain_id;
      if (subdomain_id <
          shape[_domain_decomposed_dim] / _domain_decomposed_size) {
        if (direction == subdomain_copy_direction::OriginalToSubdomain) {
          subdomain_data.resize(subdomain_shape(subdomain_id), pitched);
        }
      } else {
        SIZE leftover_dim_size =
            shape[_domain_decomposed_dim] % _domain_decomposed_size;
        calc_domain_decompose_parameter(shape, _domain_decomposed_dim,
                                        leftover_dim_size, dst_ld, src_ld, n1,
                                        n2);
        if (direction == subdomain_copy_direction::OriginalToSubdomain) {
          subdomain_data.resize(subdomain_shape(subdomain_id), pitched);
        }
      }

      if (direction == subdomain_copy_direction::OriginalToSubdomain) {
        MemoryManager<DeviceType>::CopyND(subdomain_data.data(), dst_ld, data,
                                          src_ld, n1, n2, queue_idx);
      } else {
        MemoryManager<DeviceType>::CopyND(data, src_ld, subdomain_data.data(),
                                          dst_ld, n1, n2, queue_idx);
      }
    }
    // if (log::level & log::TIME) {
    //   timer.end();
    //   timer.print("Copy subdomain " + std::to_string(subdomain_id));
    //   timer.clear();
    // }
  }

  bool domain_decomposed() { return _domain_decomposed; }

  DIM domain_decomposed_dim() { return _domain_decomposed_dim; }

  SIZE domain_decomposed_size() { return _domain_decomposed_size; }

  SIZE num_subdomains() { return _num_subdomains; }

  SIZE num_devices() { return _num_devices; }

  std::vector<SIZE> shape;
  int _num_devices;
  bool _domain_decomposed;
  DIM _domain_decomposed_dim;
  SIZE _domain_decomposed_size;
  SIZE _num_subdomains;
  std::vector<Hierarchy<D, T, DeviceType>> subdomain_hierarchies;
  T *original_data;
  Config config;
  bool uniform;
  std::vector<T *> coords;
};

} // namespace mgard_x

#endif