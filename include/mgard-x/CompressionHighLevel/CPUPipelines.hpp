namespace mgard_x {

template <DIM D, typename T, typename DeviceType, typename CompressorType>
enum compress_status_type compress_pipeline_cpu(
    DomainDecomposer<D, T, CompressorType, DeviceType> &domain_decomposer,
    T local_tol, T s, T &norm, enum error_bound_type local_ebtype,
    Config &config, Byte *compressed_subdomain_data,
    SIZE &compressed_subdomain_size) {
  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();

  // { // test openmp
  //   omp_set_nested(1);
  //   #pragma omp parallel for num_threads(2)
  //   for (int i = 0; i < 2; i++) {

  //     omp_set_num_threads(4);
  //     #pragma omp parallel for
  //     for (int i = 0; i < 4; i++) {
  //       printf("thread: %d of %d\n", omp_get_thread_num(),
  //       omp_get_num_threads());
  //     }
  //   }
  //   // int n = omp_get_num_threads();

  // }

  // // exit(0);

  using Cache = CompressorCache<D, T, DeviceType, CompressorType>;
  using HierarchyType = typename CompressorType::HierarchyType;
  CompressorType *compressor = Cache::cache.compressor;
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
  log::info("Adjust device buffers and hierarchy");
  for (int i = 0; i < domain_decomposer.num_subdomains(); i++) {
    std::vector<SIZE> shape = domain_decomposer.subdomain_shape(i);
    device_subdomain_buffer[i].resize(shape);
    device_compressed_buffer[i].resize(
        {domain_decomposer.subdomain_compressed_buffer_size(
            domain_decomposer.largest_subdomain())});

    HierarchyType &hierarchy =
        Cache::cache.GetHierarchyCache(domain_decomposer.subdomain_shape(i));
    compressor[i].Adapt(hierarchy, config, 0);
  }

  std::vector<SIZE> compressed_size(domain_decomposer.num_subdomains(), 0);

  DeviceRuntime<DeviceType>::SyncDevice();

  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Prepare device environment");
    timer_series.clear();
    timer_series.start();
  }

  Timer timer_profile;
  std::vector<float> h2d, d2h, comp, size;
  bool profile = true;
  bool profile_e2e = false;

  // For serialization
  SIZE byte_offset = 0;

  if (profile_e2e) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer_profile.clear();
    timer_profile.start();
  }

  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {

    if (profile || profile_e2e) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.clear();
      timer_profile.start();
    }

    domain_decomposer.copy_subdomain(
        device_subdomain_buffer[curr_subdomain_id], curr_subdomain_id,
        subdomain_copy_direction::OriginalToSubdomain, 0);

    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.end();
      h2d.push_back(timer_profile.get());
    }
  }

  omp_set_nested(1);
  omp_set_dynamic(0);
  int total_threads = DeviceRuntime<DeviceType>::GetNumSMs();
#pragma omp parallel for num_threads(config.openmp_num_groups)
  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {

    omp_set_num_threads(total_threads / config.openmp_num_groups);

    std::stringstream ss;
    for (DIM d = 0; d < D; d++) {
      ss << compressor[curr_subdomain_id].hierarchy->level_shape(
                compressor[curr_subdomain_id].hierarchy->l_target(), d)
         << " ";
    }
    log::info("Compressing subdomain " + std::to_string(curr_subdomain_id) +
              " with shape: " + ss.str());
    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.clear();
      timer_profile.start();
    }

    compressor[curr_subdomain_id].Compress(
        device_subdomain_buffer[curr_subdomain_id], local_ebtype, local_tol, s,
        norm, device_compressed_buffer[curr_subdomain_id], 0);

    compressed_size[curr_subdomain_id] =
        device_compressed_buffer[curr_subdomain_id].shape(0);
    double CR =
        (double)(compressor[curr_subdomain_id].hierarchy->total_num_elems() *
                 sizeof(T)) /
        compressed_size[curr_subdomain_id];
    log::info("Subdomain CR: " + std::to_string(CR));
    if (CR < 1.0) {
      log::info("Using uncompressed data instead");
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[curr_subdomain_id], curr_subdomain_id,
          subdomain_copy_direction::OriginalToSubdomain, 0);
      SIZE linearized_width = 1;
      for (DIM d = 0; d < D - 1; d++)
        linearized_width *= device_subdomain_buffer[curr_subdomain_id].shape(d);
      MemoryManager<DeviceType>::CopyND(
          device_compressed_buffer[curr_subdomain_id].data(),
          device_subdomain_buffer[curr_subdomain_id].shape(D - 1) * sizeof(T),
          (Byte *)device_subdomain_buffer[curr_subdomain_id].data(),
          device_subdomain_buffer[curr_subdomain_id].ld(D - 1) * sizeof(T),
          device_subdomain_buffer[curr_subdomain_id].shape(D - 1) * sizeof(T),
          linearized_width, 0);
      compressed_size[curr_subdomain_id] =
          compressor[curr_subdomain_id].hierarchy->total_num_elems() *
          sizeof(T);
    }

    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.end();
      comp.push_back(timer_profile.get());
    }

    if (profile || profile_e2e) {
      size.push_back(
          compressor[curr_subdomain_id].hierarchy->total_num_elems() *
          sizeof(T) / 1.0e9);
    }
  }

  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {
    // Check if we have enough space
    if (compressed_size[curr_subdomain_id] >
        compressed_subdomain_size - byte_offset - sizeof(SIZE)) {
      log::err(
          "Output too large (original size: " +
          std::to_string((double)(compressor[curr_subdomain_id]
                                      .hierarchy->total_num_elems()) *
                         sizeof(T) / 1e9) +
          " GB, compressed size: " +
          std::to_string((double)(compressed_size[curr_subdomain_id]) / 1e9) +
          " GB, leftover buffer space: " +
          std::to_string(
              (double)(compressed_subdomain_size - byte_offset - sizeof(SIZE)) /
              1e9) +
          " GB)");
      return compress_status_type::OutputTooLargeFailure;
    }

    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.clear();
      timer_profile.start();
    }
    Serialize<SIZE, DeviceType>(compressed_subdomain_data,
                                &compressed_size[curr_subdomain_id], 1,
                                byte_offset, 0);
    Serialize<Byte, DeviceType>(
        compressed_subdomain_data,
        device_compressed_buffer[curr_subdomain_id].data(),
        compressed_size[curr_subdomain_id], byte_offset, 0);
    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.end();
      d2h.push_back(timer_profile.get());
    }
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
    timer_series.print("Compress subdomains series");
    timer_series.clear();
  }
  return compress_status_type::Success;
}

template <DIM D, typename T, typename DeviceType, typename CompressorType>
enum compress_status_type decompress_pipeline_cpu(
    DomainDecomposer<D, T, CompressorType, DeviceType> &domain_decomposer,
    T local_tol, T s, T norm, enum error_bound_type local_ebtype,
    Config &config, Byte *compressed_subdomain_data) {
  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();

  SIZE byte_offset = 0;
  using Cache = CompressorCache<D, T, DeviceType, CompressorType>;
  using HierarchyType = typename CompressorType::HierarchyType;
  CompressorType *compressor = Cache::cache.compressor;
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
  for (int i = 0; i < domain_decomposer.num_subdomains(); i++) {
    std::vector<SIZE> shape = domain_decomposer.subdomain_shape(i);
    device_subdomain_buffer[i].resize(shape);
    device_compressed_buffer[i].resize(
        {domain_decomposer.subdomain_compressed_buffer_size(
            domain_decomposer.largest_subdomain())});

    HierarchyType &hierarchy =
        Cache::cache.GetHierarchyCache(domain_decomposer.subdomain_shape(i));
    compressor[i].Adapt(hierarchy, config, 0);
  }
  std::vector<SIZE> compressed_size(domain_decomposer.num_subdomains(), 0);
  DeviceRuntime<DeviceType>::SyncDevice();

  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Prepare device environment");
    timer_series.clear();
    timer_series.start();
  }

  Timer timer_profile;
  std::vector<float> h2d, d2h, comp, size;
  bool profile = true;
  bool profile_e2e = false;

  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {
    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.clear();
      timer_profile.start();
    }

    // Deserialize and copy next compressed data on queue 1
    SIZE *compressed_size_ptr = &compressed_size[curr_subdomain_id];
    Byte *compressed_data;

    Deserialize<SIZE, DeviceType>(compressed_subdomain_data,
                                  compressed_size_ptr, 1, byte_offset, false,
                                  0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    Deserialize<Byte, DeviceType>(compressed_subdomain_data, compressed_data,
                                  compressed_size[curr_subdomain_id],
                                  byte_offset, true, 0);

    device_compressed_buffer[curr_subdomain_id].resize(
        {compressed_size[curr_subdomain_id]});

    MemoryManager<DeviceType>::Copy1D(
        device_compressed_buffer[curr_subdomain_id].data(), compressed_data,
        compressed_size[curr_subdomain_id], 0);

    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.end();
      h2d.push_back(timer_profile.get());
    }
  }

  omp_set_nested(1);
  omp_set_dynamic(0);
  int total_threads = DeviceRuntime<DeviceType>::GetNumSMs();
#pragma omp parallel for num_threads(config.openmp_num_groups)
  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {

    omp_set_num_threads(total_threads / config.openmp_num_groups);

    double CR =
        (double)(compressor[curr_subdomain_id].hierarchy->total_num_elems() *
                 sizeof(T)) /
        compressed_size[curr_subdomain_id];
    log::info("Subdomain CR: " + std::to_string(CR));
    if (CR > 1.0) {
      std::stringstream ss;
      for (DIM d = 0; d < D; d++) {
        ss << compressor[curr_subdomain_id].hierarchy->level_shape(
                  compressor[curr_subdomain_id].hierarchy->l_target(), d)
           << " ";
      }
      log::info("Decompressing subdomain " + std::to_string(curr_subdomain_id) +
                " with shape: " + ss.str());
      compressor[curr_subdomain_id].Deserialize(
          device_compressed_buffer[curr_subdomain_id], 0);
    }

    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.clear();
      timer_profile.start();
    }
    if (CR > 1.0) {
      compressor[curr_subdomain_id].LosslessDecompress(
          device_compressed_buffer[curr_subdomain_id], 0);
      compressor[curr_subdomain_id].Dequantize(
          device_subdomain_buffer[curr_subdomain_id], local_ebtype, local_tol,
          s, norm, 0);
      compressor[curr_subdomain_id].Recompose(
          device_subdomain_buffer[curr_subdomain_id], 0);
    } else {
      log::info("Skipping decompression as original data was saved instead");
      device_subdomain_buffer[curr_subdomain_id].resize(
          {compressor[curr_subdomain_id].hierarchy->level_shape(
              compressor[curr_subdomain_id].hierarchy->l_target())});
      SIZE linearized_width = 1;
      for (DIM d = 0; d < D - 1; d++)
        linearized_width *= device_subdomain_buffer[curr_subdomain_id].shape(d);
      MemoryManager<DeviceType>::CopyND(
          device_subdomain_buffer[curr_subdomain_id].data(),
          device_subdomain_buffer[curr_subdomain_id].ld(D - 1),
          (T *)device_compressed_buffer[curr_subdomain_id].data(),
          device_subdomain_buffer[curr_subdomain_id].shape(D - 1),
          device_subdomain_buffer[curr_subdomain_id].shape(D - 1),
          linearized_width, 0);
    }

    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.end();
      comp.push_back(timer_profile.get());
    }

    if (profile || profile_e2e) {
      size.push_back(
          compressor[curr_subdomain_id].hierarchy->total_num_elems() *
          sizeof(T) / 1.0e9);
    }
  }

  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {

    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.clear();
      timer_profile.start();
    }

    domain_decomposer.copy_subdomain(
        device_subdomain_buffer[curr_subdomain_id], curr_subdomain_id,
        subdomain_copy_direction::SubdomainToOriginal, 0);

    if (profile) {
      DeviceRuntime<DeviceType>::SyncDevice();
      timer_profile.end();
      d2h.push_back(timer_profile.get());
    }
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
