/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/mgard_cuda_common.h"
#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda_lossless.h"
#include "cuda/parallel_huffman/huffman_workflow.cuh"
#include <typeinfo>

namespace mgard_cuda {

template <typename T, int D, typename C>
void cascaded_compress(mgard_cuda_handle<T, D> &handle, C *input_data,
                       size_t intput_count, void *&output_data,
                       size_t &output_size, int n_rle, int n_de, bool bitpack,
                       int queue_idx) {
  nvcomp::CascadedCompressor<C> compressor(input_data, intput_count, n_rle,
                                           n_de, bitpack);

  void *temp_space;
  size_t temp_size = compressor.get_temp_size();
  cudaMallocHelper(&temp_space, temp_size);

  output_size = compressor.get_max_output_size(temp_space, temp_size);
  cudaMallocHelper(&output_data, output_size);

  compressor.compress_async(temp_space, temp_size, output_data, &output_size,
                            *(cudaStream_t *)handle.get(queue_idx));
  cudaFreeHelper(temp_space);
}

// template void cascaded_compress<double, int>(mgard_cuda_handle<double>
// &handle, int * input_data,
//              size_t intput_count, void * &output_data, size_t &output_size,
//              int n_rle, int n_de, bool bitpack, int queue_idx);
// template void cascaded_compress<float, int>(mgard_cuda_handle<float> &handle,
// int * input_data,
//              size_t intput_count, void * &output_data, size_t &output_size,
//              int n_rle, int n_de, bool bitpack, int queue_idx);
// template void cascaded_compress<double, uint32_t>(mgard_cuda_handle<double>
// &handle, uint32_t * input_data,
//              size_t intput_count, void * &output_data, size_t &output_size,
//              int n_rle, int n_de, bool bitpack, int queue_idx);
// template void cascaded_compress<float, uint32_t>(mgard_cuda_handle<float>
// &handle, uint32_t * input_data,
//              size_t intput_count, void * &output_data, size_t &output_size,
//              int n_rle, int n_de, bool bitpack, int queue_idx);

template <typename T, int D, typename C>
void cascaded_decompress(mgard_cuda_handle<T, D> &handle, void *input_data,
                         size_t input_size, C *&output_data, int queue_idx) {
  nvcomp::Decompressor<C> decompressor(input_data, input_size,
                                       *(cudaStream_t *)handle.get(queue_idx));

  void *temp_space;
  size_t temp_size = decompressor.get_temp_size();
  cudaMallocHelper(&temp_space, temp_size);

  size_t output_count = decompressor.get_num_elements();
  cudaMallocHelper((void **)&output_data, output_count * sizeof(C));

  decompressor.decompress_async(temp_space, temp_size, output_data,
                                output_count,
                                *(cudaStream_t *)handle.get(queue_idx));
}

// template void cascaded_decompress<double, int>(mgard_cuda_handle<double>
// &handle, void * input_data,
//                 size_t input_size, int * &output_data,  int queue_idx);
// template void cascaded_decompress<float, int>(mgard_cuda_handle<float>
// &handle, void * input_data,
//              size_t input_size, int * &output_data, int queue_idx);
// template void cascaded_decompress<double, uint32_t>(mgard_cuda_handle<double>
// &handle, void * input_data,
//                 size_t input_size, uint32_t * &output_data,  int queue_idx);
// template void cascaded_decompress<float, uint32_t>(mgard_cuda_handle<float>
// &handle, void * input_data,
//              size_t input_size, uint32_t * &output_data, int queue_idx);

template <typename T, int D, typename C>
void lz4_compress(mgard_cuda_handle<T, D> &handle, C *input_data,
                  size_t input_count, void *&output_data, size_t &output_size,
                  size_t chunk_size, int queue_idx) {

  nvcomp::LZ4Compressor<C> compressor(input_data, input_count, chunk_size);

  void *temp_space;
  size_t temp_size = compressor.get_temp_size();
  cudaMallocHelper(&temp_space, temp_size);

  output_size = compressor.get_max_output_size(temp_space, temp_size);
  cudaMallocHelper(&output_data, output_size);

  compressor.compress_async(temp_space, temp_size, output_data, &output_size,
                            *(cudaStream_t *)handle.get(queue_idx));

  cudaStreamSynchronize(*(cudaStream_t *)handle.get(queue_idx));
  cudaFreeHelper(temp_space);
}

template <typename T, int D, typename C>
void lz4_decompress(mgard_cuda_handle<T, D> &handle, void *input_data,
                    size_t input_size, C *&output_data, size_t &output_size,
                    int queue_idx) {

  nvcomp::Decompressor<C> decompressor(input_data, input_size,
                                       *(cudaStream_t *)handle.get(queue_idx));

  void *temp_space;
  size_t temp_size = decompressor.get_temp_size();
  cudaMallocHelper(&temp_space, temp_size);

  output_size = decompressor.get_output_size();
  cudaMallocHelper((void **)&output_data, output_size);

  decompressor.decompress_async(temp_space, temp_size, output_data, output_size,
                                *(cudaStream_t *)handle.get(queue_idx));
  cudaStreamSynchronize(*(cudaStream_t *)handle.get(queue_idx));
  cudaFreeHelper(temp_space);
}

// template void lz4_compress<double, int>(mgard_cuda_handle<double> &handle,
// int * input_data,
//              size_t intput_count, void * &output_data, size_t &output_size,
//              size_t chunk_size, int queue_idx);
// template void lz4_compress<float, int>(mgard_cuda_handle<float> &handle, int
// * input_data,
//              size_t intput_count, void * &output_data, size_t &output_size,
//              size_t chunk_size, int queue_idx);
// template void lz4_compress<double, uint32_t>(mgard_cuda_handle<double>
// &handle, uint32_t * input_data,
//              size_t intput_count, void * &output_data, size_t &output_size,
//              size_t chunk_size, int queue_idx);
// template void lz4_compress<float, uint32_t>(mgard_cuda_handle<float> &handle,
// uint32_t * input_data,
//              size_t intput_count, void * &output_data, size_t &output_size,
//              size_t chunk_size, int queue_idx);

#define KERNELS(T, D, C)                                                       \
  template void cascaded_compress<T, D, C>(                                    \
      mgard_cuda_handle<T, D> & handle, C * input_data, size_t intput_count,   \
      void *&output_data, size_t &output_size, int n_rle, int n_de,            \
      bool bitpack, int queue_idx);                                            \
  template void cascaded_decompress<T, D, C>(                                  \
      mgard_cuda_handle<T, D> & handle, void *input_data, size_t input_size,   \
      C *&output_data, int queue_idx);                                         \
  template void lz4_compress<T, D, C>(mgard_cuda_handle<T, D> & handle,        \
                                      C * input_data, size_t input_count,      \
                                      void *&output_data, size_t &output_size, \
                                      size_t chunk_size, int queue_idx);       \
  template void lz4_decompress<T, D, C>(                                       \
      mgard_cuda_handle<T, D> & handle, void *input_data, size_t input_size,   \
      C *&output_data, size_t &output_count, int queue_idx);

KERNELS(double, 1, uint8_t)
KERNELS(float, 1, uint8_t)
KERNELS(double, 2, uint8_t)
KERNELS(float, 2, uint8_t)
KERNELS(double, 3, uint8_t)
KERNELS(float, 3, uint8_t)
KERNELS(double, 4, uint8_t)
KERNELS(float, 4, uint8_t)
KERNELS(double, 5, uint8_t)
KERNELS(float, 5, uint8_t)
KERNELS(double, 1, uint32_t)
KERNELS(float, 1, uint32_t)
KERNELS(double, 2, uint32_t)
KERNELS(float, 2, uint32_t)
KERNELS(double, 3, uint32_t)
KERNELS(float, 3, uint32_t)
KERNELS(double, 4, uint32_t)
KERNELS(float, 4, uint32_t)
KERNELS(double, 5, uint32_t)
KERNELS(float, 5, uint32_t)
KERNELS(double, 1, uint64_t)
KERNELS(float, 1, uint64_t)
KERNELS(double, 2, uint64_t)
KERNELS(float, 2, uint64_t)
KERNELS(double, 3, uint64_t)
KERNELS(float, 3, uint64_t)
KERNELS(double, 4, uint64_t)
KERNELS(float, 4, uint64_t)
KERNELS(double, 5, uint64_t)
KERNELS(float, 5, uint64_t)
#undef KERNELS

template <typename T, int D, typename S, typename Q>
void SeparateOutlierAndPrimary(mgard_cuda_handle<T, D> &handle, S *dqv,
                               size_t n, size_t *outlier_idx,
                               size_t outlier_count, size_t primary_count,
                               S *doutlier, Q *dprimary, int queue_idx) {

  // printf("compress outlier_idx: "); for(int i = 0; i < outlier_count; i++)
  // {printf("%llu ", outlier_idx[i]);} printf("\n");
  printf("compress outlier_count: %llu\n", outlier_count);
  printf("compress primary_count: %llu\n", primary_count);
  printf("start separating primary and outlier\n");

  size_t p = 0;
  size_t pp = 0;
  size_t op = 0;
  size_t size = outlier_idx[0] - 0;
  // printf("copy primary\n");
  if (size > 0) {
    mgard_cuda::cudaMemcpyAsyncHelper(handle, dprimary + pp, dqv + p,
                                      size * sizeof(Q), mgard_cuda::D2D,
                                      queue_idx);
  }
  pp += size;
  p += size;

  for (int i = 0; i < outlier_count - 1; i++) {
    size = 1;
    // printf("copy outlier\n");
    mgard_cuda::cudaMemcpyAsyncHelper(handle, doutlier + op, dqv + p,
                                      size * sizeof(S), mgard_cuda::D2D,
                                      queue_idx);
    op += size;
    p += size;
    size = outlier_idx[i + 1] - outlier_idx[i] - 1;
    // printf("copy primary %d %d %d\n", p, size, outlier_idx[outlier_idx.size()
    // - 1]);
    if (size > 0) {
      mgard_cuda::cudaMemcpyAsyncHelper(handle, dprimary + pp, dqv + p,
                                        size * sizeof(Q), mgard_cuda::D2D,
                                        queue_idx);
    }
    pp += size;
    p += size;
  }
  size = 1;
  // printf("copy outlier\n");
  mgard_cuda::cudaMemcpyAsyncHelper(handle, doutlier + op, dqv + p,
                                    size * sizeof(S), mgard_cuda::D2D,
                                    queue_idx);
  op += size;
  p += size;
  size = n - outlier_idx[outlier_count - 1] - 1;
  // printf("copy primary %d %d %d\n", p, size, outlier_idx[outlier_idx.size() -
  // 1]);
  if (size > 0) {
    mgard_cuda::cudaMemcpyAsyncHelper(handle, dprimary + pp, dqv + p,
                                      size * sizeof(Q), mgard_cuda::D2D,
                                      queue_idx);
  }
  // printf("done copy primary\n");
  pp += size;
  p += size;

  if (pp != primary_count || op != outlier_count) {
    printf("Primary or outlier size mismatch!\n");
  }
  printf("done separating primary and outlier\n");
}

template <typename T, int D, typename S, typename Q>
void CombineOutlierAndPrimary(mgard_cuda_handle<T, D> &handle, S *dqv, size_t n,
                              size_t *outlier_idx, size_t outlier_count,
                              size_t primary_count, S *doutlier, Q *dprimary,
                              int queue_idx) {
  size_t p = 0;
  size_t pp = 0;
  size_t op = 0;
  size_t size = outlier_idx[0] - 0;
  // printf("copy primary\n");
  if (size > 0) {
    mgard_cuda::cudaMemcpyAsyncHelper(handle, dqv + p, dprimary + pp,
                                      size * sizeof(Q), mgard_cuda::D2D,
                                      queue_idx);
  }
  pp += size;
  p += size;

  for (int i = 0; i < outlier_count - 1; i++) {
    size = 1;
    // printf("copy outlier\n");
    mgard_cuda::cudaMemcpyAsyncHelper(handle, dqv + p, doutlier + op,
                                      size * sizeof(S), mgard_cuda::D2D,
                                      queue_idx);
    op += size;
    p += size;
    size = outlier_idx[i + 1] - outlier_idx[i] - 1;
    // printf("copy primary %d %d %d\n", p, size, outlier_idx[outlier_idx.size()
    // - 1]);
    if (size > 0) {
      mgard_cuda::cudaMemcpyAsyncHelper(handle, dqv + p, dprimary + pp,
                                        size * sizeof(Q), mgard_cuda::D2D,
                                        queue_idx);
    }
    pp += size;
    p += size;
  }
  size = 1;
  // printf("copy outlier\n");
  mgard_cuda::cudaMemcpyAsyncHelper(handle, dqv + p, doutlier + op,
                                    size * sizeof(S), mgard_cuda::D2D,
                                    queue_idx);
  op += size;
  p += size;
  size = n - outlier_idx[outlier_count - 1] - 1;
  // printf("copy primary %d %d %d\n", p, size, outlier_idx[outlier_idx.size() -
  // 1]);
  if (size > 0) {
    mgard_cuda::cudaMemcpyAsyncHelper(handle, dqv + p, dprimary + pp,
                                      size * sizeof(Q), mgard_cuda::D2D,
                                      queue_idx);
  }
  // printf("done copy primary\n");
  pp += size;
  p += size;
}

#define KERNELS(T, D, S, Q)                                                    \
  template void SeparateOutlierAndPrimary<T, D, S, Q>(                         \
      mgard_cuda_handle<T, D> & handle, S * dqv, size_t n,                     \
      size_t * outlier_idx,\ 
          size_t outlier_count,                                                \
      size_t primary_count,\ 
          S * doutlier,                                                        \
      Q * dprimary, int queue_idx);                                            \
  template void CombineOutlierAndPrimary<T, D, S, Q>(                          \
      mgard_cuda_handle<T, D> & handle, S * dqv, size_t n,                     \
      size_t * outlier_idx,\ 
          size_t outlier_count,                                                \
      size_t primary_count,\ 
          S * doutlier,                                                        \
      Q * dprimary, int queue_idx);

KERNELS(double, 1, int, uint32_t)
KERNELS(float, 1, int, uint32_t)
KERNELS(double, 2, int, uint32_t)
KERNELS(float, 2, int, uint32_t)
KERNELS(double, 3, int, uint32_t)
KERNELS(float, 3, int, uint32_t)
KERNELS(double, 4, int, uint32_t)
KERNELS(float, 4, int, uint32_t)
KERNELS(double, 5, int, uint32_t)
KERNELS(float, 5, int, uint32_t)
#undef KERNELS

template <typename T, int D, typename S, typename Q, typename H>
void huffman_compress(mgard_cuda_handle<T, D> &handle, S *input_data,
                      size_t input_count, std::vector<size_t> &outlier_idx,
                      H *&out_meta, size_t &out_meta_size, H *&out_data,
                      size_t &out_data_size, int chunk_size, int dict_size,
                      int queue_idx) {

  HuffmanEncode<T, D, S, Q, H>(handle, input_data, input_count, outlier_idx,
                               out_meta, out_meta_size, out_data, out_data_size,
                               chunk_size, dict_size);
}

template <typename T, int D, typename S, typename Q, typename H>
void huffman_decompress(mgard_cuda_handle<T, D> &handle, H *in_meta,
                        size_t in_meta_size, H *in_data, size_t in_data_size,
                        S *&output_data, size_t &output_count, int queue_idx) {
  HuffmanDecode<T, D, S, Q, H>(handle, output_data, output_count, in_meta,
                               in_meta_size, in_data, in_data_size);
}

#define KERNELS(T, D, S, Q, H)                                                 \
  template void huffman_compress<T, D, S, Q, H>(                               \
      mgard_cuda_handle<T, D> & handle, S * input_data, size_t input_count,    \
      std::vector<size_t> & outlier_idx, H * &out_meta,                        \
      size_t & out_meta_size, H * &out_data, size_t & out_data_size,           \
      int chunk_size, int dict_size, int queue_idx);                           \
  template void huffman_decompress<T, D, S, Q, H>(                             \
      mgard_cuda_handle<T, D> & handle, H * in_meta, size_t in_meta_size,      \
      H * in_data, size_t in_data_size, S * &output_data,                      \
      size_t & output_count, int queue_idx);

KERNELS(double, 1, int, uint32_t, uint32_t)
KERNELS(float, 1, int, uint32_t, uint32_t)
KERNELS(double, 2, int, uint32_t, uint32_t)
KERNELS(float, 2, int, uint32_t, uint32_t)
KERNELS(double, 3, int, uint32_t, uint32_t)
KERNELS(float, 3, int, uint32_t, uint32_t)
KERNELS(double, 4, int, uint32_t, uint32_t)
KERNELS(float, 4, int, uint32_t, uint32_t)
KERNELS(double, 5, int, uint32_t, uint32_t)
KERNELS(float, 5, int, uint32_t, uint32_t)
KERNELS(double, 1, int, uint32_t, uint64_t)
KERNELS(float, 1, int, uint32_t, uint64_t)
KERNELS(double, 2, int, uint32_t, uint64_t)
KERNELS(float, 2, int, uint32_t, uint64_t)
KERNELS(double, 3, int, uint32_t, uint64_t)
KERNELS(float, 3, int, uint32_t, uint64_t)
KERNELS(double, 4, int, uint32_t, uint64_t)
KERNELS(float, 4, int, uint32_t, uint64_t)
KERNELS(double, 5, int, uint32_t, uint64_t)
KERNELS(float, 5, int, uint32_t, uint64_t)

} // namespace mgard_cuda