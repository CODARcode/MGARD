namespace mgard_x {
template <typename T> int max_dim(std::vector<T> &shape) {
  int max_d = 0;
  T max_n = 0;
  for (int i = 0; i < shape.size(); i++) {
    if (max_n < shape[i]) {
      max_d = i;
      max_n = shape[i];
    }
  }
  return max_d;
}

template <typename T> int min_dim(std::vector<T> &shape) {
  int min_d = 0;
  T min_n = SIZE_MAX;
  for (int i = 0; i < shape.size(); i++) {
    if (min_n > shape[i]) {
      min_d = i;
      min_n = shape[i];
    }
  }
  return min_d;
}

template <typename T> std::vector<T> find_refactors(T n) {
  std::vector<T> factors;
  T z = 2;
  while (z * z <= n) {
    if (n % z == 0) {
      factors.push_back(z);
      n /= z;
    } else {
      z++;
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  return factors;
}

template <typename T> void adjust_shape(std::vector<T> &shape, Config config) {
  log::info("Using shape adjustment");
  int num_timesteps;
  if (config.domain_decomposition == domain_decomposition_type::TemporalDim) {
    // If do shape adjustment with temporal dim domain decomposition
    // the temporal dim has to be the first dim
    assert(config.temporal_dim == 0);
    num_timesteps = shape[0] / config.temporal_dim_size;
    shape[0] = config.temporal_dim_size;
  }
  int max_d = max_dim(shape);
  SIZE max_n = shape[max_d];
  std::vector<SIZE> factors = find_refactors(max_n);
  // std::cout << "factors: ";
  // for (SIZE f : factors) std::cout << f << " ";
  // std::cout << "\n";
  shape[max_d] = 1;
  for (int i = factors.size() - 1; i >= 0; i--) {
    int min_d = min_dim(shape);
    shape[min_d] *= factors[i];
    // std::cout << "multiple " << factors[i] <<
    // " to dim " << min_d << ": " << shape[min_d] << "\n";
  }
  if (config.domain_decomposition == domain_decomposition_type::TemporalDim) {
    shape[0] *= num_timesteps;
  }
  // std::cout << "shape: ";
  // for (SIZE n : shape) {
  //   std::cout <<  n << "\n";
  // }
  std::stringstream ss;
  for (DIM d = 0; d < shape.size(); d++) {
    ss << shape[d] << " ";
  }
  log::info("Shape adjusted to " + ss.str());
}
} // namespace mgard_x