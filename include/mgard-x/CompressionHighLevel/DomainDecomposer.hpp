/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_DOMAIN_DECOMPOSER_HPP
#define MGARD_X_DOMAIN_DECOMPOSER_HPP

#define ORIGINAL_TO_SUBDOMAIN 0
#define SUBDOMAIN_TO_ORIGINAL 1

namespace mgard_x {

template <DIM D, typename T, typename DeviceType> class DomainDecomposer {
public:
  size_t estimate_memory_usgae(std::vector<SIZE> shape, double outlier_ratio,
                               double reduction_ratio, bool enable_prefetch) {
    size_t estimate_memory_usgae = 0;
    size_t total_elem = 1;
    for (DIM d = 0; d < D; d++) {
      total_elem *= shape[d];
    }

    Array<1, T, DeviceType> array_with_pitch({1});
    size_t pitch_size = array_with_pitch.ld(0) * sizeof(T);

    // log::info("pitch_size: " + std::to_string(pitch_size));

    size_t hierarchy_space = 0;
    // space need for hiearachy
    int nlevel = std::numeric_limits<int>::max();
    for (DIM i = 0; i < shape.size(); i++) {
      int n = shape[i];
      int l = 0;
      while (n > 2) {
        n = n / 2 + 1;
        l++;
      }
      nlevel = std::min(nlevel, l);
    }
    nlevel--;
    for (DIM d = 0; d < D; d++) {
      hierarchy_space += shape[d] * 2 * sizeof(T); // dist
      hierarchy_space += shape[d] * 2 * sizeof(T); // ratio
    }
    SIZE max_dim = 0;
    for (DIM d = 0; d < D; d++) {
      max_dim = std::max(max_dim, shape[d]);
    }

    hierarchy_space +=
        D * (nlevel + 1) * roundup(max_dim * sizeof(T), pitch_size); // volume
    for (DIM d = 0; d < D; d++) {
      hierarchy_space += shape[d] * 2 * sizeof(T); // am
      hierarchy_space += shape[d] * 2 * sizeof(T); // bm
    }

    // log::info("hierarchy_space: " +
    // std::to_string((double)hierarchy_space/1e9));

    size_t input_space = roundup(shape[D - 1] * sizeof(T), pitch_size);
    for (DIM d = 0; d < D - 1; d++) {
      input_space *= shape[d];
    }

    size_t output_space = (double)input_space * reduction_ratio;

    // For prefetching
    if (enable_prefetch) {
      input_space *= 2;
      output_space *= 2;
    }

    // log::info("input_space: " + std::to_string((double)input_space/1e9));

    CompressionLowLevelWorkspace<D, T, DeviceType> compression_workspace;

    estimate_memory_usgae =
        hierarchy_space + input_space + output_space +
        compression_workspace.estimate_size(shape, 64, outlier_ratio);

    return estimate_memory_usgae;
  }

  bool need_domain_decomposition(std::vector<SIZE> shape, bool enable_prefetch) {
    size_t estm = estimate_memory_usgae(shape, 0.1, 1, enable_prefetch);
    size_t aval = DeviceRuntime<DeviceType>::GetAvailableMemory();
    log::info("Estimated memory usage: " + std::to_string((double)estm / 1e9) +
              "GB, Available: " + std::to_string((double)aval / 1e9) + "GB");
    return estm >= aval;
  }

  bool generate_domain_decomposition_strategy(std::vector<SIZE> shape,
                                              DIM &_domain_decomposed_dim,
                                              SIZE &_domain_decomposed_size,
                                              SIZE num_dev) {
    // determine max dimension
    DIM max_dim = 0;
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
                                chunck_shape[_domain_decomposed_dim] + 1;

    // Need prefetch if there are more subdomains than devices         
    bool need_prefetch = curr_num_subdomains > num_dev;
    // Then check if each chunk can fit into device memory
    while (need_domain_decomposition(chunck_shape, need_prefetch)) {
      // Divide by 2 and round up
      chunck_shape[_domain_decomposed_dim] =
          (chunck_shape[_domain_decomposed_dim] - 1) / 2 + 1;

      curr_num_subdomains = (shape[_domain_decomposed_dim] - 1) /
                                chunck_shape[_domain_decomposed_dim] + 1;
      need_prefetch = curr_num_subdomains > num_dev;
    }
    _domain_decomposed_size = chunck_shape[_domain_decomposed_dim];
    log::info(
        "_domain_decomposed_dim: " + std::to_string(_domain_decomposed_dim) +
        ", _domain_decomposed_size: " +
        std::to_string(_domain_decomposed_size));
    return true;
  }

  DomainDecomposer() : _domain_decomposed(false) {}

  // Find domain decomposion method
  DomainDecomposer(T *original_data, std::vector<SIZE> shape, int _num_devices)
      : original_data(original_data), shape(shape), _num_devices(_num_devices) {
    if (!need_domain_decomposition(shape, false) && this->_num_devices == 1) {
      this->_domain_decomposed = false;
      this->_domain_decomposed_dim = 0;
      this->_domain_decomposed_size = this->shape[0];
    } else {
      this->_domain_decomposed = true;
      generate_domain_decomposition_strategy(
          shape, this->_domain_decomposed_dim, this->_domain_decomposed_size,
          this->_num_devices);
    }
    this->_num_subdomains = (shape[this->_domain_decomposed_dim] - 1) /
                                this->_domain_decomposed_size +
                            1;
  }

  // Force to use this domain decomposion method
  DomainDecomposer(T *original_data, std::vector<SIZE> shape, int _num_devices,
                   bool _domain_decomposed, DIM _domain_decomposed_dim,
                   SIZE _domain_decomposed_size)
      : original_data(original_data), shape(shape), _num_devices(_num_devices),
        _domain_decomposed_dim(_domain_decomposed_dim),
        _domain_decomposed_size(_domain_decomposed_size),
        _domain_decomposed(_domain_decomposed) {
    if (!this->_domain_decomposed) {
      this->_domain_decomposed_dim = 0;
      this->_domain_decomposed_size = this->shape[0];
    }
    this->_num_subdomains = (this->shape[this->_domain_decomposed_dim] - 1) /
                                this->_domain_decomposed_size +
                            1;
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

  bool check_shape(Array<D, T, DeviceType> &subdomain_data, std::vector<SIZE> shape) {
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
                      int option, int queue_idx) {
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
      if (option == ORIGINAL_TO_SUBDOMAIN) {
        subdomain_data = Array<D, T, DeviceType>(shape);
        // subdomain_data.load(original_data);
        MemoryManager<DeviceType>::CopyND(
            subdomain_data.data(), subdomain_data.ld(D - 1), original_data,
            shape[D - 1], shape[D - 1], linearized_width, queue_idx);
      } else {
        MemoryManager<DeviceType>::CopyND(
            original_data, shape[D - 1], subdomain_data.data(),
            subdomain_data.ld(D - 1), shape[D - 1], linearized_width, queue_idx);
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
        if (option == ORIGINAL_TO_SUBDOMAIN) {
          subdomain_data.resize(subdomain_shape(subdomain_id), pitched);
        }
      } else {
        SIZE leftover_dim_size =
            shape[_domain_decomposed_dim] % _domain_decomposed_size;
        calc_domain_decompose_parameter(shape, _domain_decomposed_dim,
                                        leftover_dim_size, dst_ld, src_ld, n1,
                                        n2);
        if (option == ORIGINAL_TO_SUBDOMAIN) {
          subdomain_data.resize(subdomain_shape(subdomain_id), pitched);
        }
      }

      if (option == ORIGINAL_TO_SUBDOMAIN) {
        MemoryManager<DeviceType>::CopyND(subdomain_data.data(), dst_ld, data,
                                          src_ld, n1, n2, queue_idx);
      } else if (option == SUBDOMAIN_TO_ORIGINAL) {
        MemoryManager<DeviceType>::CopyND(data, src_ld, subdomain_data.data(),
                                          dst_ld, n1, n2, queue_idx);
      } else {
        log::err("copy_subdomain: wrong option.");
        exit(-1);
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

  T *original_data;
};

} // namespace mgard_x

#endif