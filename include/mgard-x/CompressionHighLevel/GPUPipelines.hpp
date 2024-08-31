namespace mgard_x {
template <DIM D, typename T, typename DeviceType, typename CompressorType>
enum compress_status_type compress_pipeline_gpu(
    DomainDecomposer<D, T, CompressorType, DeviceType> &domain_decomposer,
    T local_tol, T s, T &norm, enum error_bound_type local_ebtype,
    Config &config, Byte *compressed_subdomain_data,
    SIZE &compressed_subdomain_size) {
  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();

  using Cache = CompressorCache<D, T, DeviceType, CompressorType>;
  using HierarchyType = typename CompressorType::HierarchyType;
  CompressorType &compressor = Cache::cache.compressor[0];
  Array<D, T, DeviceType> *device_subdomain_buffer =
      Cache::cache.device_subdomain_buffer;
  Array<1, Byte, DeviceType> *device_compressed_buffer =
      Cache::cache.device_compressed_buffer;

  if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(0),
                                     domain_decomposer.uniform)) {
    Cache::cache.ClearHierarchyCache();
  }

  for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
    if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(id),
                                       domain_decomposer.uniform)) {
      Cache::cache.InsertHierarchyCache(
          domain_decomposer.subdomain_hierarchy(id));
    }
  }
  log::info("Adjust device buffers");
  std::vector<SIZE> shape =
      domain_decomposer.subdomain_shape(domain_decomposer.largest_subdomain());
  SIZE num_elements = 1;
  for (int i = 0; i < shape.size(); i++)
    num_elements *= shape[i];
  device_subdomain_buffer[0].resize(shape);
  device_subdomain_buffer[1].resize(shape);
  device_compressed_buffer[0].resize(
      {domain_decomposer.subdomain_compressed_buffer_size(
          domain_decomposer.largest_subdomain())});
  device_compressed_buffer[1].resize(
      {domain_decomposer.subdomain_compressed_buffer_size(
          domain_decomposer.largest_subdomain())});

  HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(
      domain_decomposer.subdomain_shape(domain_decomposer.largest_subdomain()));
  log::info("Adapt Compressor to hierarchy");
  compressor.Adapt(hierarchy, config, 0);

  DeviceRuntime<DeviceType>::SyncDevice();

  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Prepare device environment");
    timer_series.clear();
    timer_series.start();
  }

  Timer timer_profile;
  std::vector<float> h2d, d2h, comp, size;
  bool profile = false;
  bool profile_e2e = false;

  // For serialization
  SIZE byte_offset = 0;

  // Pre-fetch the first subdomain to one buffer
  int current_buffer = 0;
  int current_queue = 0;

  if (profile || profile_e2e) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer_profile.clear();
    timer_profile.start();
  }
  domain_decomposer.copy_subdomain(
      device_subdomain_buffer[current_buffer], 0,
      subdomain_copy_direction::OriginalToSubdomain, current_queue);

  if (profile) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer_profile.end();
    h2d.push_back(timer_profile.get());
  }

  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {
    SIZE next_subdomain_id;
    int next_buffer = (current_buffer + 1) % 2;
    int next_queue = (current_queue + 1) % 3;
    HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(
        domain_decomposer.subdomain_shape(curr_subdomain_id));
    log::info("Adapt Compressor to hierarchy");
    compressor.Adapt(hierarchy, config, current_queue);

    if (curr_subdomain_id + 1 < domain_decomposer.num_subdomains()) {
      // Prefetch the next subdomain
      next_subdomain_id = curr_subdomain_id + 1;
      // Copy data
      if (profile) {
        DeviceRuntime<DeviceType>::SyncDevice();
        timer_profile.clear();
        timer_profile.start();
      }
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[next_buffer], next_subdomain_id,
          subdomain_copy_direction::OriginalToSubdomain, next_queue);
      if (profile) {
        DeviceRuntime<DeviceType>::SyncDevice();
        timer_profile.end();
        h2d.push_back(timer_profile.get());
      }
    }

    std::stringstream ss;
    for (DIM d = 0; d < D; d++) {
      ss << compressor.hierarchy->level_shape(compressor.hierarchy->l_target(),
                                              d)
         << " ";
    }
    log::info("Compressing subdomain " + std::to_string(curr_subdomain_id) +
              " with shape: " + ss.str());
    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.clear();
      timer_profile.start();
    }
    compressor.Compress(
        device_subdomain_buffer[current_buffer], local_ebtype, local_tol, s,
        norm, device_compressed_buffer[current_buffer], current_queue);

    SIZE compressed_size = device_compressed_buffer[current_buffer].shape(0);
    double CR = (double)compressor.hierarchy->total_num_elems() * sizeof(T) /
                compressed_size;
    log::info("Subdomain CR: " + std::to_string(CR));
    if (CR < 1.0) {
      log::info("Using uncompressed data instead");
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[current_buffer], curr_subdomain_id,
          subdomain_copy_direction::OriginalToSubdomain, current_queue);
      SIZE linearized_width = 1;
      for (DIM d = 0; d < D - 1; d++)
        linearized_width *= device_subdomain_buffer[current_buffer].shape(d);
      MemoryManager<DeviceType>::CopyND(
          device_compressed_buffer[current_buffer].data(),
          device_subdomain_buffer[current_buffer].shape(D - 1) * sizeof(T),
          (Byte *)device_subdomain_buffer[current_buffer].data(),
          device_subdomain_buffer[current_buffer].ld(D - 1) * sizeof(T),
          device_subdomain_buffer[current_buffer].shape(D - 1) * sizeof(T),
          linearized_width, current_queue);
      compressed_size = compressor.hierarchy->total_num_elems() * sizeof(T);
    }

    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.end();
      comp.push_back(timer_profile.get());
    }

    if (profile || profile_e2e) {
      size.push_back(compressor.hierarchy->total_num_elems() * sizeof(T) /
                     1.0e9);
    }

    // Check if we have enough space
    if (compressed_size >
        compressed_subdomain_size - byte_offset - sizeof(SIZE)) {
      log::err("Output too large (original size: " +
               std::to_string((double)compressor.hierarchy->total_num_elems() *
                              sizeof(T) / 1e9) +
               " GB, compressed size: " +
               std::to_string((double)compressed_size / 1e9) +
               " GB, leftover buffer space: " +
               std::to_string((double)(compressed_subdomain_size - byte_offset -
                                       sizeof(SIZE)) /
                              1e9) +
               " GB)");
      return compress_status_type::OutputTooLargeFailure;
    }

    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.clear();
      timer_profile.start();
    }
    Serialize<SIZE, DeviceType>(compressed_subdomain_data, &compressed_size, 1,
                                byte_offset, current_queue);
    Serialize<Byte, DeviceType>(compressed_subdomain_data,
                                device_compressed_buffer[current_buffer].data(),
                                compressed_size, byte_offset, current_queue);
    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.end();
      d2h.push_back(timer_profile.get());
    }

    if (config.compress_with_dryrun) {
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[current_buffer], curr_subdomain_id,
          subdomain_copy_direction::SubdomainToOriginal, current_queue);
    }
    current_buffer = next_buffer;
    current_queue = next_queue;
  }

  if (profile_e2e) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer_profile.end();
    timer_profile.print("end to end");
    float s = 0;
    for (float t : size)
      s += t;
    timer_profile.print_throughput("end to end", s * 1e9);
  }

  if (profile) {
    // double total_size = domain_decomposer.shape[0] *
    // domain_decomposer.shape[1] * domain_decomposer.shape[2] * sizeof(T) /
    // 1e9; std::cout << "comp: " << comp / domain_decomposer.num_subdomains()
    // << "(" << total_size / comp << " GB/s)"<< "\n"; std::cout << "h2d: " <<
    // h2d / domain_decomposer.num_subdomains() << "(" << total_size / h2d << "
    // GB/s)"<< "\n"; std::cout << "d2h: " << d2h /
    // domain_decomposer.num_subdomains() << "(" << byte_offset/ 1e9 / d2h << "
    // GB/s)"<< "\n";
    std::cout << "comp: "
              << "\n";
    for (float t : comp)
      std::cout << t << ", ";
    std::cout << "\n";

    std::cout << "h2d: "
              << "\n";
    for (float t : h2d)
      std::cout << t << ", ";
    std::cout << "\n";

    std::cout << "d2h: "
              << "\n";
    for (float t : d2h)
      std::cout << t << ", ";
    std::cout << "\n";

    std::cout << "size: "
              << "\n";
    for (float t : size)
      std::cout << t << ", ";
    std::cout << "\n";

    std::cout << "comp_speed: "
              << "\n";
    for (int i = 0; i < comp.size(); i++)
      std::cout << size[i] / comp[i] << ", ";
    std::cout << "\n";
  }

  compressed_subdomain_size = byte_offset;
  DeviceRuntime<DeviceType>::SyncDevice();
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Compress subdomains series with prefetch");
    timer_series.clear();
  }
  return compress_status_type::Success;
}

template <DIM D, typename T, typename DeviceType, typename CompressorType>
enum compress_status_type decompress_pipeline_gpu(
    DomainDecomposer<D, T, CompressorType, DeviceType> &domain_decomposer,
    T local_tol, T s, T norm, enum error_bound_type local_ebtype,
    Config &config, Byte *compressed_subdomain_data) {
  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();

  SIZE byte_offset = 0;
  using Cache = CompressorCache<D, T, DeviceType, CompressorType>;
  using HierarchyType = typename CompressorType::HierarchyType;
  CompressorType &compressor = Cache::cache.compressor[0];
  Array<D, T, DeviceType> *device_subdomain_buffer =
      Cache::cache.device_subdomain_buffer;
  Array<1, Byte, DeviceType> *device_compressed_buffer =
      Cache::cache.device_compressed_buffer;

  if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(0),
                                     domain_decomposer.uniform)) {
    Cache::cache.ClearHierarchyCache();
  }

  for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
    if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(id),
                                       domain_decomposer.uniform)) {
      Cache::cache.InsertHierarchyCache(
          domain_decomposer.subdomain_hierarchy(id));
    }
  }

  log::info("Adjust device buffers");
  std::vector<SIZE> shape =
      domain_decomposer.subdomain_shape(domain_decomposer.largest_subdomain());
  SIZE num_elements = 1;
  for (int i = 0; i < shape.size(); i++)
    num_elements *= shape[i];
  device_subdomain_buffer[0].resize(shape);
  device_subdomain_buffer[1].resize(shape);
  device_compressed_buffer[0].resize(
      {domain_decomposer.subdomain_compressed_buffer_size(
          domain_decomposer.largest_subdomain())});
  device_compressed_buffer[1].resize(
      {domain_decomposer.subdomain_compressed_buffer_size(
          domain_decomposer.largest_subdomain())});

  HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(
      domain_decomposer.subdomain_shape(domain_decomposer.largest_subdomain()));
  log::info("Adapt Compressor to hierarchy");
  compressor.Adapt(hierarchy, config, 0);

  DeviceRuntime<DeviceType>::SyncDevice();

  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Prepare device environment");
    timer_series.clear();
    timer_series.start();
  }

  Timer timer_profile;
  std::vector<float> h2d, d2h, comp, size;
  bool profile = false;
  bool profile_e2e = false;

  // Pre-fetch the first subdomain on queue 0
  int current_buffer = 0;
  int current_queue = 0;

  if (profile || profile_e2e) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer_profile.clear();
    timer_profile.start();
  }

  SIZE compressed_size;
  SIZE *compressed_size_ptr = &compressed_size;
  Byte *compressed_data;

  Deserialize<SIZE, DeviceType>(compressed_subdomain_data, compressed_size_ptr,
                                1, byte_offset, false, current_queue);
  DeviceRuntime<DeviceType>::SyncQueue(current_queue);
  Deserialize<Byte, DeviceType>(compressed_subdomain_data, compressed_data,
                                compressed_size, byte_offset, true,
                                current_queue);

  device_compressed_buffer[current_buffer].resize({compressed_size});

  MemoryManager<DeviceType>::Copy1D(
      device_compressed_buffer[current_buffer].data(), compressed_data,
      compressed_size, current_queue);

  if (profile) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer_profile.end();
    h2d.push_back(timer_profile.get());
  }

  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {
    SIZE next_subdomain_id;
    int next_buffer = (current_buffer + 1) % 2;
    int next_queue = (current_queue + 1) % 3;
    HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(
        domain_decomposer.subdomain_shape(curr_subdomain_id));
    log::info("Adapt Compressor to hierarchy");
    compressor.Adapt(hierarchy, config, current_queue);

    if (curr_subdomain_id + 1 < domain_decomposer.num_subdomains()) {
      // Prefetch the next subdomain
      next_subdomain_id = curr_subdomain_id + 1;

      if (profile) {
        DeviceRuntime<DeviceType>::SyncDevice();
        timer_profile.clear();
        timer_profile.start();
      }

      // Deserialize and copy next compressed data on queue 1
      SIZE compressed_size;
      SIZE *compressed_size_ptr = &compressed_size;
      Byte *compressed_data;

      Deserialize<SIZE, DeviceType>(compressed_subdomain_data,
                                    compressed_size_ptr, 1, byte_offset, false,
                                    next_queue);
      DeviceRuntime<DeviceType>::SyncQueue(next_queue);
      Deserialize<Byte, DeviceType>(compressed_subdomain_data, compressed_data,
                                    compressed_size, byte_offset, true,
                                    next_queue);

      device_compressed_buffer[next_buffer].resize({compressed_size});

      MemoryManager<DeviceType>::Copy1D(
          device_compressed_buffer[next_buffer].data(), compressed_data,
          compressed_size, next_queue);

      if (profile) {
        DeviceRuntime<DeviceType>::SyncDevice();
        timer_profile.end();
        h2d.push_back(timer_profile.get());
      }
    }

    double CR = (double)compressor.hierarchy->total_num_elems() * sizeof(T) /
                compressed_size;
    log::info("Subdomain CR: " + std::to_string(CR));
    if (CR > 1.0) {
      std::stringstream ss;
      for (DIM d = 0; d < D; d++) {
        ss << compressor.hierarchy->level_shape(
                  compressor.hierarchy->l_target(), d)
           << " ";
      }
      log::info("Decompressing subdomain " + std::to_string(curr_subdomain_id) +
                " with shape: " + ss.str());
      compressor.Deserialize(device_compressed_buffer[current_buffer],
                             current_queue);
    }

    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.clear();
      timer_profile.start();
    }

    if (curr_subdomain_id > 0) {
      // We delay D2H since since it can delay the D2H in lossless decompession
      // and dequantization
      int previous_buffer = std::abs((current_buffer - 1) % 2);
      int previous_queue = std::abs((current_queue - 1) % 3);
      SIZE prev_subdomain_id = curr_subdomain_id - 1;
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[previous_buffer], prev_subdomain_id,
          subdomain_copy_direction::SubdomainToOriginal, previous_queue);
    }

    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.end();
      d2h.push_back(timer_profile.get());
    }

    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.clear();
      timer_profile.start();
    }
    if (CR > 1.0) {
      compressor.LosslessDecompress(device_compressed_buffer[current_buffer],
                                    current_queue);
      compressor.Dequantize(device_subdomain_buffer[current_buffer],
                            local_ebtype, local_tol, s, norm, current_queue);
      compressor.Recompose(device_subdomain_buffer[current_buffer],
                           current_queue);
    } else {
      log::info("Skipping decompression as original data was saved instead");
      device_subdomain_buffer[current_buffer].resize(
          {compressor.hierarchy->level_shape(
              compressor.hierarchy->l_target())});
      SIZE linearized_width = 1;
      for (DIM d = 0; d < D - 1; d++)
        linearized_width *= device_subdomain_buffer[current_buffer].shape(d);
      MemoryManager<DeviceType>::CopyND(
          device_subdomain_buffer[current_buffer].data(),
          device_subdomain_buffer[current_buffer].ld(D - 1),
          (T *)device_compressed_buffer[current_buffer].data(),
          device_subdomain_buffer[current_buffer].shape(D - 1),
          device_subdomain_buffer[current_buffer].shape(D - 1),
          linearized_width, current_queue);
    }

    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.end();
      comp.push_back(timer_profile.get());
    }

    if (profile || profile_e2e) {
      size.push_back(compressor.hierarchy->total_num_elems() * sizeof(T) /
                     1.0e9);
    }

    // Need to ensure decompession is complete without blocking other operations
    DeviceRuntime<DeviceType>::SyncQueue(current_queue);
    current_buffer = next_buffer;
    current_queue = next_queue;
  }

  if (profile) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer_profile.clear();
    timer_profile.start();
  }

  // Copy the last subdomain
  int previous_buffer = std::abs((current_buffer - 1) % 2);
  int previous_queue = previous_buffer;
  SIZE prev_subdomain_id = domain_decomposer.num_subdomains() - 1;
  domain_decomposer.copy_subdomain(
      device_subdomain_buffer[previous_buffer], prev_subdomain_id,
      subdomain_copy_direction::SubdomainToOriginal, previous_queue);

  if (profile) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer_profile.end();
    d2h.push_back(timer_profile.get());
  }

  if (profile_e2e) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer_profile.end();
    timer_profile.print("end to end");
    float s = 0;
    for (float t : size)
      s += t;
    timer_profile.print_throughput("end to end", s * 1e9);
  }

  if (profile) {
    // double total_size = domain_decomposer.shape[0] *
    // domain_decomposer.shape[1] * domain_decomposer.shape[2] * sizeof(T) /
    // 1e9; std::cout << "comp: " << comp / domain_decomposer.num_subdomains()
    // << "(" << total_size / comp << " GB/s)"<< "\n"; std::cout << "h2d: " <<
    // h2d / domain_decomposer.num_subdomains() << "(" << total_size / h2d << "
    // GB/s)"<< "\n"; std::cout << "d2h: " << d2h /
    // domain_decomposer.num_subdomains() << "(" << byte_offset/ 1e9 / d2h << "
    // GB/s)"<< "\n";
    std::cout << "comp: "
              << "\n";
    for (float t : comp)
      std::cout << t << ", ";
    std::cout << "\n";

    std::cout << "h2d: "
              << "\n";
    for (float t : h2d)
      std::cout << t << ", ";
    std::cout << "\n";

    std::cout << "d2h: "
              << "\n";
    for (float t : d2h)
      std::cout << t << ", ";
    std::cout << "\n";

    std::cout << "size: "
              << "\n";
    for (float t : size)
      std::cout << t << ", ";
    std::cout << "\n";

    std::cout << "comp_speed: "
              << "\n";
    for (int i = 0; i < comp.size(); i++)
      std::cout << size[i] / comp[i] << ", ";
    std::cout << "\n";
  }

  DeviceRuntime<DeviceType>::SyncDevice();
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Decompress subdomains series with prefetch");
    timer_series.clear();
  }
  return compress_status_type::Success;
}
} // namespace mgard_x
