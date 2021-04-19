/* 
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGARD_CUDA_LOSSLESS
#define MGARD_CUDA_LOSSLESS

#include "mgard_cuda_common.h"
#include "cascaded.hpp"
#include "lz4.hpp"
#include "nvcomp.hpp"

namespace mgard_cuda {


template <typename T, int D, typename C>
void cascaded_compress(mgard_cuda_handle<T, D> &handle, C * input_data, 
             size_t intput_count, void * &output_data, size_t &output_size, 
             int n_rle, int n_de, bool bitpack, int queue_idx);

template <typename T, int D, typename C>
void cascaded_decompress(mgard_cuda_handle<T, D> &handle, void * input_data, size_t input_size, 
             C * &output_data, int queue_idx);

template <typename T, int D, typename C>
void lz4_compress(mgard_cuda_handle<T, D> &handle, C * input_data, 
             size_t intput_count, void * &output_data, size_t &output_size, 
             size_t chunk_size, int queue_idx);

template <typename T, int D, typename C>
void lz4_decompress(mgard_cuda_handle<T, D> &handle, void * input_data, size_t input_size, 
              C * &output_data, size_t& output_size, int queue_idx);

template <typename T, int D, typename S, typename Q>
void SeparateOutlierAndPrimary(mgard_cuda_handle<T, D> &handle, 
                               S * dqv, size_t n, size_t * outlier_idx, 
                               size_t outlier_count, size_t primary_count, 
                               S * doutlier, Q * dprimary,
                               int queue_idx);
template <typename T, int D, typename S, typename Q>
void CombineOutlierAndPrimary(mgard_cuda_handle<T, D> &handle, 
                               S * dqv, size_t n, size_t * outlier_idx, 
                               size_t outlier_count, size_t primary_count, 
                               S * doutlier, Q * dprimary,
                               int queue_idx);

template <typename T, int D, typename S, typename Q, typename H>
void huffman_compress(mgard_cuda_handle<T, D> &handle, S * input_data, size_t input_count, std::vector<size_t>& outlier_idx,
                      H * &out_meta, size_t &out_meta_size, H * &out_data, size_t &out_data_size, 
                      int chunk_size, int dict_size, int queue_idx);
template <typename T, int D, typename S, typename Q, typename H>
void huffman_decompress(mgard_cuda_handle<T, D> &handle, 
                        H * in_meta, size_t in_meta_size, H * in_data, size_t in_data_size, 
                        S * &output_data, size_t &output_count, int queue_idx);

}

#endif