/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGARD_CUDA_LOSSLESS
#define MGARD_CUDA_LOSSLESS

#include "Common.h"
#include "cascaded.hpp"
#include "lz4.hpp"
#include "nvcomp.hpp"

namespace mgard_cuda {

template <uint32_t D, typename T, typename C>
void cascaded_compress(Handle<D, T> &handle, C *input_data, size_t intput_count,
                       void *&output_data, size_t &output_size, int n_rle,
                       int n_de, bool bitpack, int queue_idx);

template <uint32_t D, typename T, typename C>
void cascaded_decompress(Handle<D, T> &handle, void *input_data,
                         size_t input_size, C *&output_data, int queue_idx);

template <uint32_t D, typename T, typename C>
void lz4_compress(Handle<D, T> &handle, C *input_data, size_t intput_count,
                  void *&output_data, size_t &output_size, size_t chunk_size,
                  int queue_idx);

template <uint32_t D, typename T, typename C>
void lz4_decompress(Handle<D, T> &handle, void *input_data, size_t input_size,
                    C *&output_data, size_t &output_size, int queue_idx);

template <uint32_t D, typename T, typename S, typename Q>
void SeparateOutlierAndPrimary(Handle<D, T> &handle, S *dqv, size_t n,
                               size_t *outlier_idx, size_t outlier_count,
                               size_t primary_count, S *doutlier, Q *dprimary,
                               int queue_idx);
template <uint32_t D, typename T, typename S, typename Q>
void CombineOutlierAndPrimary(Handle<D, T> &handle, S *dqv, size_t n,
                              size_t *outlier_idx, size_t outlier_count,
                              size_t primary_count, S *doutlier, Q *dprimary,
                              int queue_idx);

template <uint32_t D, typename T, typename S, typename Q, typename H>
void huffman_compress(Handle<D, T> &handle, S *input_data, size_t input_count,
                      std::vector<size_t> &outlier_idx, H *&out_meta,
                      size_t &out_meta_size, H *&out_data,
                      size_t &out_data_size, int chunk_size, int dict_size,
                      int queue_idx);
template <uint32_t D, typename T, typename S, typename Q, typename H>
void huffman_decompress(Handle<D, T> &handle, H *in_meta, size_t in_meta_size,
                        H *in_data, size_t in_data_size, S *&output_data,
                        size_t &output_count, int queue_idx);

} // namespace mgard_cuda

#endif