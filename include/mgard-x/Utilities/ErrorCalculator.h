/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_ERROR_CALCULATOR
#define MGARD_X_ERROR_CALCULATOR

#include "../Types.h"

namespace mgard_x {

template <typename T>
T L_inf_norm(size_t n, T * data) {
  T L_inf = 0;
  for (size_t i = 0; i < n; ++i) {
    T temp = fabs(data[i]);
    if (temp > L_inf)
      L_inf = temp;
  }
  return L_inf;
}

template <typename T>
T L_2_norm(size_t n, T * data) {
  T L_2 = 0;
  for (size_t i = 0; i < n; ++i) {
    T temp = fabs(data[i]);
      L_2 += temp * temp;
  }
  return std::sqrt(L_2);
}


template <typename T>
T L_inf_error(size_t n, T * original_data, T * decompressed_data, enum error_bound_type mode) {
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
T L_2_error(size_t n, T * original_data, T * decompressed_data, enum error_bound_type mode) {
  T error_L_2_norm = 0;
  for (size_t i = 0; i < n; ++i) {
    T temp = fabs(original_data[i] - decompressed_data[i]);
    error_L_2_norm += temp * temp;
  }
  if (mode == error_bound_type::ABS) {
    return std::sqrt(error_L_2_norm);
  } else if (mode == error_bound_type::REL) {
    return std::sqrt(error_L_2_norm) / L_2_norm(n, original_data);
  } else {
    return 0;
  }
}

template <typename T>
T MSE(size_t n, T * original_data, T * decompressed_data) {
  T mse = 0;
  for (size_t i = 0; i < n; ++i) {
    T temp = fabs(original_data[i] - decompressed_data[i]);
    mse += temp * temp;
  }
  return mse / n;
}

template <typename T>
T PSNR(size_t n, T * original_data, T * decompressed_data) {
  T mse = MSE(n, original_data, decompressed_data);
  T max = 0, min = std::numeric_limits<T>::max();
  for (size_t i = 0; i < n; ++i) {
    if (max < original_data[i]) max = original_data[i];
    if (min > original_data[i]) min = original_data[i];
  }
  T range = max - min;
  return 20*std::log10(range)-10*std::log10(mse);
}

}

#endif