/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_ERROR_CALCULATOR
#define MGRAD_CUDA_ERROR_CALCULATOR

namespace mgard_cuda {

template <typename T> T L_inf_norm(size_t n, T *data);

template <typename T> T L_2_norm(size_t n, T *data);

template <typename T>
T L_inf_error(size_t n, T *original_data, T *decompressed_data,
              enum error_bound_type mode);

template <typename T>
T L_2_error(size_t n, T *original_data, T *decompressed_data,
            enum error_bound_type mode);

template <typename T> T MSE(size_t n, T *original_data, T *decompressed_data);

template <typename T> T PSNR(size_t n, T *original_data, T *decompressed_data);

} // namespace mgard_cuda

#endif