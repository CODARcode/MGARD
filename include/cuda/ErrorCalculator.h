/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_ERROR_CALCULATOR
#define MGARD_X_ERROR_CALCULATOR

namespace mgard_x {

template <typename T>
T L_inf_norm(size_t n, T * data);

template <typename T>
T L_2_norm(size_t n, T * data);

template <typename T>
T L_inf_error(size_t n, T * original_data, T * decompressed_data, enum error_bound_type mode);

template <typename T>
T L_2_error(size_t n, T * original_data, T * decompressed_data, enum error_bound_type mode);

template <typename T>
T MSE(size_t n, T * original_data, T * decompressed_data);

template <typename T>
T PSNR(size_t n, T * original_data, T * decompressed_data);

}

#endif