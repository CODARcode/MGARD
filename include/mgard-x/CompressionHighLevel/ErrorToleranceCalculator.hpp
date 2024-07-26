namespace mgard_x {
template <DIM D, typename T, typename DeviceType, typename CompressorType>
T calc_subdomain_norm_series_w_prefetch(
    DomainDecomposer<D, T, CompressorType, DeviceType> &domain_decomposer,
    T s) {
  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();

  DeviceRuntime<DeviceType>::SyncQueue(0);
  Array<1, T, DeviceType> norm_array({1});
  SubArray<1, T, DeviceType> norm_subarray(norm_array);
  T norm = 0;

  // Two buffers one for current and one for next
  Array<D, T, DeviceType> device_subdomain_buffer[2];
  // Pre-allocate to the size of the first subdomain
  // Following subdomains should be no bigger than the first one
  // We shouldn't need to reallocate in the future
  device_subdomain_buffer[0].resize(domain_decomposer.subdomain_shape(0));
  device_subdomain_buffer[1].resize(domain_decomposer.subdomain_shape(0));

  // Pre-fetch the first subdomain to one buffer
  int current_buffer = 0;
  domain_decomposer.copy_subdomain(
      device_subdomain_buffer[current_buffer], 0,
      subdomain_copy_direction::OriginalToSubdomain, 0);

  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {
    SIZE next_subdomain_id;
    int next_buffer = (current_buffer + 1) % 2;
    if (curr_subdomain_id + 1 < domain_decomposer.num_subdomains()) {
      // Prefetch the next subdomain
      next_subdomain_id = curr_subdomain_id + 1;
      // Copy data
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[next_buffer], next_subdomain_id,
          subdomain_copy_direction::OriginalToSubdomain, 1);
    }

    // device_input_buffer has to be not pitched to avoid copy for
    // linearization
    assert(!device_subdomain_buffer[current_buffer].isPitched());
    // Disable normalize_coordinate since we do not want to void dividing
    // total_elems
    T local_norm =
        norm_calculator(device_subdomain_buffer[current_buffer],
                        SubArray<1, T, DeviceType>(), norm_subarray, s, false);
    if (s == std::numeric_limits<T>::infinity()) {
      norm = std::max(norm, local_norm);
    } else {
      norm += local_norm * local_norm;
    }
    current_buffer = next_buffer;
    DeviceRuntime<DeviceType>::SyncQueue(1);
  }
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Calculate subdomains norm series");
    timer_series.clear();
  }

  DeviceRuntime<DeviceType>::SyncDevice();
  return norm;
}

template <DIM D, typename T, typename DeviceType, typename CompressorType>
T calc_norm_decomposed_w_prefetch(
    DomainDecomposer<D, T, CompressorType, DeviceType> &domain_decomposer, T s,
    bool normalize_coordinates, SIZE total_num_elem) {

  // Process a series of subdomains according to the subdomain id list
  T norm = calc_subdomain_norm_series_w_prefetch(domain_decomposer, s);
  if (s != std::numeric_limits<T>::infinity()) {
    if (!normalize_coordinates) {
      norm = std::sqrt(norm);
    } else {
      norm = std::sqrt(norm / total_num_elem);
    }
  }
  if (s == std::numeric_limits<T>::infinity()) {
    log::info("L_inf norm: " + std::to_string(norm));
  } else {
    log::info("L_2 norm: " + std::to_string(norm));
  }
  return norm;
}

template <DIM D, typename T, typename DeviceType, typename CompressorType>
T calc_norm_decomposed(
    DomainDecomposer<D, T, CompressorType, DeviceType> &domain_decomposer, T s,
    bool normalize_coordinates, SIZE total_num_elem) {
  Array<D, T, DeviceType> device_input_buffer;
  T norm = 0;
  for (int subdomain_id = 0; subdomain_id < domain_decomposer.num_subdomains();
       subdomain_id++) {
    domain_decomposer.copy_subdomain(
        device_input_buffer, subdomain_id,
        subdomain_copy_direction::OriginalToSubdomain, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    Array<1, T, DeviceType> norm_array({1});
    SubArray<1, T, DeviceType> norm_subarray(norm_array);
    // device_input_buffer has to be not pitched to avoid copy for
    // linearization
    assert(!device_input_buffer.isPitched());
    // Disable normalize_coordinate since we do not want to void dividing
    // total_elems
    T local_norm =
        norm_calculator(device_input_buffer, SubArray<1, T, DeviceType>(),
                        norm_subarray, s, false);
    if (s == std::numeric_limits<T>::infinity()) {
      norm = std::max(norm, local_norm);
    } else {
      norm += local_norm * local_norm;
    }
  }
  if (s != std::numeric_limits<T>::infinity()) {
    if (!normalize_coordinates) {
      norm = std::sqrt(norm);
    } else {
      norm = std::sqrt(norm / total_num_elem);
    }
  }
  if (s == std::numeric_limits<T>::infinity()) {
    log::info("L_inf norm: " + std::to_string(norm));
  } else {
    log::info("L_2 norm: " + std::to_string(norm));
  }
  return norm;
}

template <typename T>
T calc_local_abs_tol(enum error_bound_type ebtype, T norm, T tol, T s,
                     SIZE num_subdomain) {
  T local_abs_tol;
  if (ebtype == error_bound_type::REL) {
    if (s == std::numeric_limits<T>::infinity()) {
      log::info("L_inf norm: " + std::to_string(norm));
      local_abs_tol = tol * norm;
    } else {
      log::info("L_2 norm: " + std::to_string(norm));
      local_abs_tol = std::sqrt((tol * norm) * (tol * norm) / num_subdomain);
    }
  } else {
    if (s == std::numeric_limits<T>::infinity()) {
      local_abs_tol = tol;
    } else {
      local_abs_tol = std::sqrt((tol * tol) / num_subdomain);
    }
  }
  log::info("local abs tol: " + std::to_string(local_abs_tol));
  return local_abs_tol;
}
} // namespace mgard_x