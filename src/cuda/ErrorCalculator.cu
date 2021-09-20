/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <cuda/Common.h>
#include <math.h>

namespace mgard_cuda {

template <typename T> T L_inf_norm(size_t n, T *data) {
  T L_inf = 0;
  for (int i = 0; i < n; ++i) {
    T temp = fabs(data[i]);
    if (temp > L_inf)
      L_inf = temp;
  }
  return L_inf;
}

template <typename T> T L_2_norm(size_t n, T *data) {
  T L_2 = 0;
  for (int i = 0; i < n; ++i) {
    T temp = fabs(data[i]);
    L_2 += temp * temp;
  }
  return std::sqrt(L_2);
}

template <typename T>
T L_inf_error(size_t n, T *original_data, T *decompressed_data,
              enum error_bound_type mode) {
  T error_L_inf_norm = 0;
  for (int i = 0; i < n; ++i) {
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
T L_2_error(size_t n, T *original_data, T *decompressed_data,
            enum error_bound_type mode) {
  T error_L_2_norm = 0;
  for (int i = 0; i < n; ++i) {
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

template <typename T> T MSE(size_t n, T *original_data, T *decompressed_data) {
  T mse = 0;
  for (int i = 0; i < n; ++i) {
    T temp = fabs(original_data[i] - decompressed_data[i]);
    mse += temp * temp;
  }
  return mse / n;
}

template <typename T> T PSNR(size_t n, T *original_data, T *decompressed_data) {
  T mse = MSE(n, original_data, decompressed_data);
  T max = 0, min = std::numeric_limits<T>::max();
  for (int i = 0; i < n; ++i) {
    if (max < original_data[i])
      max = original_data[i];
    if (min > original_data[i])
      min = original_data[i];
  }
  T range = max - min;
  return 20 * std::log10(range) - 10 * std::log10(mse);
}

// double max = 0, min = std::numeric_limits<double>::max(), range = 0;
// double error_sum = 0, mse = 0, psnr = 0;
// for (int i = 0; i < num_double; ++i) {
//   if (max < in_buff[i]) max = in_buff[i];
//   if (min > in_buff[i]) min = in_buff[i];
//   double err = fabs(in_buff[i] - mgard_out_buff[i]);
//   error_sum += err * err;
// }
// range = max - min;
// mse = error_sum / num_double;
// psnr = 20*log::log10(range)-10*log::log10(mse);

template float L_inf_norm<float>(size_t n, float *data);
template double L_inf_norm<double>(size_t n, double *data);
template float L_2_norm<float>(size_t n, float *data);
template double L_2_norm<double>(size_t n, double *data);

template float L_inf_error<float>(size_t n, float *original_data,
                                  float *decompressed_data,
                                  enum error_bound_type mode);
template double L_inf_error<double>(size_t n, double *original_data,
                                    double *decompressed_data,
                                    enum error_bound_type mode);
template float L_2_error<float>(size_t n, float *original_data,
                                float *decompressed_data,
                                enum error_bound_type mode);
template double L_2_error<double>(size_t n, double *original_data,
                                  double *decompressed_data,
                                  enum error_bound_type mode);
template float MSE<float>(size_t n, float *original_data,
                          float *decompressed_data);
template double MSE<double>(size_t n, double *original_data,
                            double *decompressed_data);
template float PSNR<float>(size_t n, float *original_data,
                           float *decompressed_data);
template double PSNR<double>(size_t n, double *original_data,
                             double *decompressed_data);
} // namespace mgard_cuda