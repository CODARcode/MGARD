/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_ERROR_CALCULATOR
#define MGARD_X_ERROR_CALCULATOR

// #include "../../TensorMeshHierarchy.hpp"
// #include "../../TensorNorms.hpp"
// #include "../../shuffle.hpp"
#include "Types.h"

#include <cmath>
#include <limits>
#include <vector>


namespace mgard_x {

template <typename T> T L_inf_norm(size_t n, const T *data) {
  T L_inf = 0;
  for (size_t i = 0; i < n; ++i) {
    T temp = fabs(data[i]);
    if (temp > L_inf)
      L_inf = temp;
  }
  // Avoiding issue with norm == 0
  if (L_inf == 0)
    L_inf = std::numeric_limits<T>::epsilon();
  return L_inf;
}

template <typename T>
T L_2_norm(std::vector<SIZE> shape, const T *data, bool normalize_coordinates) {
  SIZE n = 1;
  for (DIM d = 0; d < shape.size(); d++)
    n *= shape[d];
  T L_2 = 0;
  for (size_t i = 0; i < n; ++i) {
    T temp = fabs(data[i]);
    L_2 += temp * temp;
  }
  // Avoiding issue with norm == 0
  if (L_2 == 0)
    L_2 = std::numeric_limits<T>::epsilon();
  if (!normalize_coordinates) {
    return std::sqrt(L_2);
  } else {
    return std::sqrt(L_2 / n);
  }
}

template <typename T>
T L_inf_error(size_t n, const T *original_data, const T *decompressed_data,
              enum error_bound_type mode) {
  T error_L_inf_norm = 0;
  for (size_t i = 0; i < n; ++i) {
    T temp = fabs(original_data[i] - decompressed_data[i]);
    if (temp > error_L_inf_norm)
      error_L_inf_norm = temp;
  }
  if (mode == error_bound_type::ABS) {
    return error_L_inf_norm;
  } else if (mode == error_bound_type::REL) {
    return error_L_inf_norm / L_inf_norm(n, original_data);
  } else {
    return 0;
  }
}

template <typename T>
T L_2_error(std::vector<SIZE> shape, const T *original_data,
            const T *decompressed_data, enum error_bound_type mode,
            bool normalize_coordinates) {
  SIZE n = 1;
  for (DIM d = 0; d < shape.size(); d++)
    n *= shape[d];
  T *const error = new T[n];
  for (size_t i = 0; i < n; ++i) {
    T temp = fabs(original_data[i] - decompressed_data[i]);
    error[i] = temp;
  }
  T org_norm = L_2_norm<T>(shape, original_data, normalize_coordinates);
  T err_norm = L_2_norm<T>(shape, error, normalize_coordinates);
  delete[] error;

  if (mode == error_bound_type::ABS) {
    return err_norm;
  } else if (mode == error_bound_type::REL) {
    return err_norm / org_norm;
  } else {
    return 0;
  }
}

template <typename T>
T MSE(size_t n, const T *original_data, const T *decompressed_data) {
  T mse = 0;
  for (size_t i = 0; i < n; ++i) {
    T temp = fabs(original_data[i] - decompressed_data[i]);
    mse += temp * temp;
  }
  return mse / n;
}

template <typename T>
T PSNR(size_t n, const T *original_data, const T *decompressed_data) {
  T mse = MSE(n, original_data, decompressed_data);
  T max = 0, min = std::numeric_limits<T>::max();
  for (size_t i = 0; i < n; ++i) {
    if (max < original_data[i])
      max = original_data[i];
    if (min > original_data[i])
      min = original_data[i];
  }
  T range = max - min;
  return 20 * std::log10(range / std::sqrt(mse));
}

} // namespace mgard_x

#endif
