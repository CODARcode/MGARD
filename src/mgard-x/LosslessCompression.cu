/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "compressors.hpp"
#include "cuda/Common.h"
#include "cuda/CommonInternal.h"
#include "cuda/LosslessCompression.h"
#include "cuda/ParallelHuffman/huffman_workflow.cuh"
// #include "cuda/ParallelHuffman/Huffman.hpp"

#include <typeinfo>

namespace mgard_x {

template <uint32_t D, typename T, typename C>
void cascaded_compress(Handle<D, T> &handle, C *input_data, size_t intput_count,
                       void *&output_data, size_t &output_size, int n_rle,
                       int n_de, bool bitpack, int queue_idx) {

  nvcomp::CascadedCompressor compressor(nvcomp::TypeOf<C>(), n_rle, n_de,
                                        bitpack);

  size_t *temp_bytes;
  cudaMallocHostHelper((void **)&temp_bytes, sizeof(size_t));
  size_t *output_bytes;
  cudaMallocHostHelper((void **)&output_bytes, sizeof(size_t));

  compressor.configure(intput_count * sizeof(C), temp_bytes, output_bytes);

  void *temp_space;
  cudaMallocHelper(handle, &temp_space, *temp_bytes);
  cudaMallocHelper(handle, &output_data, *output_bytes);

  compressor.compress_async(input_data, intput_count * sizeof(C), temp_space,
                            *temp_bytes, output_data, output_bytes,
                            *(cudaStream_t *)handle.get(queue_idx));
  handle.sync(queue_idx);
  output_size = *output_bytes;
  cudaFreeHelper(temp_space);
  cudaFreeHostHelper(temp_bytes);
  cudaFreeHostHelper(output_bytes);
}

template <uint32_t D, typename T, typename C>
void cascaded_decompress(Handle<D, T> &handle, void *input_data,
                         size_t input_size, C *&output_data, int queue_idx) {

  // nvcomp::Decompressor<C> decompressor(input_data, input_size,
  //                                      *(cudaStream_t
  //                                      *)handle.get(queue_idx));

  nvcomp::CascadedDecompressor decompressor;

  size_t *temp_bytes;
  cudaMallocHostHelper((void **)&temp_bytes, sizeof(size_t));
  size_t *output_bytes;
  cudaMallocHostHelper((void **)&output_bytes, sizeof(size_t));

  decompressor.configure(input_data, input_size, temp_bytes, output_bytes,
                         *(cudaStream_t *)handle.get(queue_idx));

  void *temp_space;
  cudaMallocHelper(handle, (void **)&temp_space, *temp_bytes);
  cudaMallocHelper(handle, (void **)&output_data, *output_bytes);

  decompressor.decompress_async(input_data, input_size, temp_space, *temp_bytes,
                                output_data, *output_bytes,
                                *(cudaStream_t *)handle.get(queue_idx));
  handle.sync(queue_idx);
  cudaFreeHelper(temp_space);
  cudaFreeHostHelper(temp_bytes);
  cudaFreeHostHelper(output_bytes);
}

template <uint32_t D, typename T, typename C>
void lz4_compress(Handle<D, T> &handle, C *input_data, size_t input_count,
                  void *&output_data, size_t &output_size, size_t chunk_size,
                  int queue_idx) {
  nvcompType_t dtype = NVCOMP_TYPE_UCHAR;
  nvcomp::LZ4Compressor compressor(chunk_size, dtype);

  size_t *temp_bytes;
  cudaMallocHostHelper((void **)&temp_bytes, sizeof(size_t));
  size_t *output_bytes;
  cudaMallocHostHelper((void **)&output_bytes, sizeof(size_t));

  compressor.configure(input_count * sizeof(C), temp_bytes, output_bytes);

  void *temp_space;
  cudaMallocHelper(handle, &temp_space, *temp_bytes);
  cudaMallocHelper(handle, &output_data, *output_bytes);

  compressor.compress_async(input_data, input_count * sizeof(C), temp_space,
                            *temp_bytes, output_data, output_bytes,
                            *(cudaStream_t *)handle.get(queue_idx));

  handle.sync(queue_idx);
  output_size = *output_bytes;
  cudaFreeHelper(temp_space);
  cudaFreeHostHelper(temp_bytes);
  cudaFreeHostHelper(output_bytes);
}

template <uint32_t D, typename T, typename C>
void lz4_decompress(Handle<D, T> &handle, void *input_data, size_t input_size,
                    C *&output_data, size_t &output_size, int queue_idx) {

  nvcomp::LZ4Decompressor decompressor;

  size_t *temp_bytes;
  cudaMallocHostHelper((void **)&temp_bytes, sizeof(size_t));
  size_t *output_bytes;
  cudaMallocHostHelper((void **)&output_bytes, sizeof(size_t));

  decompressor.configure(input_data, input_size, temp_bytes, output_bytes,
                         *(cudaStream_t *)handle.get(queue_idx));

  void *temp_space;
  cudaMallocHelper(handle, (void **)&temp_space, *temp_bytes);
  cudaMallocHelper(handle, (void **)&output_data, *output_bytes);

  decompressor.decompress_async(input_data, input_size, temp_space, *temp_bytes,
                                output_data, *output_bytes,
                                *(cudaStream_t *)handle.get(queue_idx));
  handle.sync(queue_idx);
  output_size = *output_bytes;
  cudaFreeHelper(temp_space);
  cudaFreeHostHelper(temp_bytes);
  cudaFreeHostHelper(output_bytes);
}

#define KERNELS(D, T, C)                                                       \
  template void cascaded_compress<D, T, C>(                                    \
      Handle<D, T> & handle, C * input_data, size_t intput_count,              \
      void *&output_data, size_t &output_size, int n_rle, int n_de,            \
      bool bitpack, int queue_idx);                                            \
  template void cascaded_decompress<D, T, C>(                                  \
      Handle<D, T> & handle, void *input_data, size_t input_size,              \
      C *&output_data, int queue_idx);                                         \
  template void lz4_compress<D, T, C>(Handle<D, T> & handle, C * input_data,   \
                                      size_t input_count, void *&output_data,  \
                                      size_t &output_size, size_t chunk_size,  \
                                      int queue_idx);                          \
  template void lz4_decompress<D, T, C>(                                       \
      Handle<D, T> & handle, void *input_data, size_t input_size,              \
      C *&output_data, size_t &output_count, int queue_idx);

KERNELS(1, double, uint8_t)
KERNELS(1, float, uint8_t)
KERNELS(2, double, uint8_t)
KERNELS(2, float, uint8_t)
KERNELS(3, double, uint8_t)
KERNELS(3, float, uint8_t)
KERNELS(4, double, uint8_t)
KERNELS(4, float, uint8_t)
KERNELS(5, double, uint8_t)
KERNELS(5, float, uint8_t)
KERNELS(1, double, uint32_t)
KERNELS(1, float, uint32_t)
KERNELS(2, double, uint32_t)
KERNELS(2, float, uint32_t)
KERNELS(3, double, uint32_t)
KERNELS(3, float, uint32_t)
KERNELS(4, double, uint32_t)
KERNELS(4, float, uint32_t)
KERNELS(5, double, uint32_t)
KERNELS(5, float, uint32_t)
KERNELS(1, double, uint64_t)
KERNELS(1, float, uint64_t)
KERNELS(2, double, uint64_t)
KERNELS(2, float, uint64_t)
KERNELS(3, double, uint64_t)
KERNELS(3, float, uint64_t)
KERNELS(4, double, uint64_t)
KERNELS(4, float, uint64_t)
KERNELS(5, double, uint64_t)
KERNELS(5, float, uint64_t)
#undef KERNELS

template <uint32_t D, typename T, typename S, typename Q>
void SeparateOutlierAndPrimary(Handle<D, T> &handle, S *dqv, size_t n,
                               size_t *outlier_idx, size_t outlier_count,
                               size_t primary_count, S *doutlier, Q *dprimary,
                               int queue_idx) {

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
    mgard_x::cudaMemcpyAsyncHelper(handle, dprimary + pp, dqv + p,
                                   size * sizeof(Q), mgard_x::D2D, queue_idx);
  }
  pp += size;
  p += size;

  for (int i = 0; i < outlier_count - 1; i++) {
    size = 1;
    // printf("copy outlier\n");
    mgard_x::cudaMemcpyAsyncHelper(handle, doutlier + op, dqv + p,
                                   size * sizeof(S), mgard_x::D2D, queue_idx);
    op += size;
    p += size;
    size = outlier_idx[i + 1] - outlier_idx[i] - 1;
    // printf("copy primary %d %d %d\n", p, size, outlier_idx[outlier_idx.size()
    // - 1]);
    if (size > 0) {
      mgard_x::cudaMemcpyAsyncHelper(handle, dprimary + pp, dqv + p,
                                     size * sizeof(Q), mgard_x::D2D, queue_idx);
    }
    pp += size;
    p += size;
  }
  size = 1;
  // printf("copy outlier\n");
  mgard_x::cudaMemcpyAsyncHelper(handle, doutlier + op, dqv + p,
                                 size * sizeof(S), mgard_x::D2D, queue_idx);
  op += size;
  p += size;
  size = n - outlier_idx[outlier_count - 1] - 1;
  // printf("copy primary %d %d %d\n", p, size, outlier_idx[outlier_idx.size() -
  // 1]);
  if (size > 0) {
    mgard_x::cudaMemcpyAsyncHelper(handle, dprimary + pp, dqv + p,
                                   size * sizeof(Q), mgard_x::D2D, queue_idx);
  }
  // printf("done copy primary\n");
  pp += size;
  p += size;

  if (pp != primary_count || op != outlier_count) {
    printf("Primary or outlier size mismatch!\n");
  }
  printf("done separating primary and outlier\n");
}

template <uint32_t D, typename T, typename S, typename Q>
void CombineOutlierAndPrimary(Handle<D, T> &handle, S *dqv, size_t n,
                              size_t *outlier_idx, size_t outlier_count,
                              size_t primary_count, S *doutlier, Q *dprimary,
                              int queue_idx) {
  size_t p = 0;
  size_t pp = 0;
  size_t op = 0;
  size_t size = outlier_idx[0] - 0;
  // printf("copy primary\n");
  if (size > 0) {
    mgard_x::cudaMemcpyAsyncHelper(handle, dqv + p, dprimary + pp,
                                   size * sizeof(Q), mgard_x::D2D, queue_idx);
  }
  pp += size;
  p += size;

  for (int i = 0; i < outlier_count - 1; i++) {
    size = 1;
    // printf("copy outlier\n");
    mgard_x::cudaMemcpyAsyncHelper(handle, dqv + p, doutlier + op,
                                   size * sizeof(S), mgard_x::D2D, queue_idx);
    op += size;
    p += size;
    size = outlier_idx[i + 1] - outlier_idx[i] - 1;
    // printf("copy primary %d %d %d\n", p, size, outlier_idx[outlier_idx.size()
    // - 1]);
    if (size > 0) {
      mgard_x::cudaMemcpyAsyncHelper(handle, dqv + p, dprimary + pp,
                                     size * sizeof(Q), mgard_x::D2D, queue_idx);
    }
    pp += size;
    p += size;
  }
  size = 1;
  // printf("copy outlier\n");
  mgard_x::cudaMemcpyAsyncHelper(handle, dqv + p, doutlier + op,
                                 size * sizeof(S), mgard_x::D2D, queue_idx);
  op += size;
  p += size;
  size = n - outlier_idx[outlier_count - 1] - 1;
  // printf("copy primary %d %d %d\n", p, size, outlier_idx[outlier_idx.size() -
  // 1]);
  if (size > 0) {
    mgard_x::cudaMemcpyAsyncHelper(handle, dqv + p, dprimary + pp,
                                   size * sizeof(Q), mgard_x::D2D, queue_idx);
  }
  // printf("done copy primary\n");
  pp += size;
  p += size;
}

#define KERNELS(D, T, S, Q)                                                    \
  template void SeparateOutlierAndPrimary<D, T, S, Q>(                         \
      Handle<D, T> & handle, S * dqv, size_t n, size_t * outlier_idx,\ 
          size_t outlier_count,                                                \
      size_t primary_count,\ 
          S * doutlier,                                                        \
      Q * dprimary, int queue_idx);                                            \
  template void CombineOutlierAndPrimary<D, T, S, Q>(                          \
      Handle<D, T> & handle, S * dqv, size_t n, size_t * outlier_idx,\ 
          size_t outlier_count,                                                \
      size_t primary_count,\ 
          S * doutlier,                                                        \
      Q * dprimary, int queue_idx);

KERNELS(1, double, int, uint32_t)
KERNELS(1, float, int, uint32_t)
KERNELS(2, double, int, uint32_t)
KERNELS(2, float, int, uint32_t)
KERNELS(3, double, int, uint32_t)
KERNELS(3, float, int, uint32_t)
KERNELS(4, double, int, uint32_t)
KERNELS(4, float, int, uint32_t)
KERNELS(5, double, int, uint32_t)
KERNELS(5, float, int, uint32_t)
#undef KERNELS

template <uint32_t D, typename T, typename S, typename Q, typename H,
          typename DeviceType>
void huffman_compress(Handle<D, T> &handle, S *input_data, size_t input_count,
                      std::vector<size_t> &outlier_idx, H *&out_meta,
                      size_t &out_meta_size, H *&out_data,
                      size_t &out_data_size, int chunk_size, int dict_size,
                      int queue_idx) {

  // HuffmanEncode<D, T, S, Q, H, DeviceType>(handle, input_data, input_count,
  // outlier_idx,
  //                              out_meta, out_meta_size, out_data,
  //                              out_data_size, chunk_size, dict_size);
}

template <uint32_t D, typename T, typename S, typename Q, typename H,
          typename DeviceType>
void huffman_decompress(Handle<D, T> &handle, H *in_meta, size_t in_meta_size,
                        H *in_data, size_t in_data_size, S *&output_data,
                        size_t &output_count, int queue_idx) {
  // HuffmanDecode<D, T, S, Q, H, DeviceType>(handle, output_data, output_count,
  // in_meta,
  //                              in_meta_size, in_data, in_data_size);
}

#define KERNELS(D, T, S, Q, H)                                                 \
  template void huffman_compress<D, T, S, Q, H, CUDA>(                         \
      Handle<D, T> & handle, S * input_data, size_t input_count,               \
      std::vector<size_t> & outlier_idx, H * &out_meta,                        \
      size_t & out_meta_size, H * &out_data, size_t & out_data_size,           \
      int chunk_size, int dict_size, int queue_idx);                           \
  template void huffman_decompress<D, T, S, Q, H, CUDA>(                       \
      Handle<D, T> & handle, H * in_meta, size_t in_meta_size, H * in_data,    \
      size_t in_data_size, S * &output_data, size_t & output_count,            \
      int queue_idx);

KERNELS(1, double, int, uint32_t, uint32_t)
KERNELS(1, float, int, uint32_t, uint32_t)
KERNELS(2, double, int, uint32_t, uint32_t)
KERNELS(2, float, int, uint32_t, uint32_t)
KERNELS(3, double, int, uint32_t, uint32_t)
KERNELS(3, float, int, uint32_t, uint32_t)
KERNELS(4, double, int, uint32_t, uint32_t)
KERNELS(4, float, int, uint32_t, uint32_t)
KERNELS(5, double, int, uint32_t, uint32_t)
KERNELS(5, float, int, uint32_t, uint32_t)
KERNELS(1, double, int, uint32_t, uint64_t)
KERNELS(1, float, int, uint32_t, uint64_t)
KERNELS(2, double, int, uint32_t, uint64_t)
KERNELS(2, float, int, uint32_t, uint64_t)
KERNELS(3, double, int, uint32_t, uint64_t)
KERNELS(3, float, int, uint32_t, uint64_t)
KERNELS(4, double, int, uint32_t, uint64_t)
KERNELS(4, float, int, uint32_t, uint64_t)
KERNELS(5, double, int, uint32_t, uint64_t)
KERNELS(5, float, int, uint32_t, uint64_t)

template <uint32_t D, typename T, typename S, typename H>
void cpu_lossless_compression(Handle<D, T> &handle, S *input_data,
                              size_t input_count, H *&out_data,
                              size_t &out_data_size) {

  int *int_vector = new int[input_count];

  cudaMemcpyAsyncHelper(handle, int_vector, input_data, input_count * sizeof(S),
                        AUTO, 0);
  handle.sync(0);

  std::vector<long int> input_vector(input_count);
  for (int i = 0; i < input_count; i++)
    input_vector[i] = int_vector[i];

  // printf("%u %u\n", sizeof(long int), sizeof(int));
  // printf("dqv\n");
  // print_matrix_cuda(1, input_count, input_data, input_count);

  // printf("input_vector: ");
  // for (int i = 0; i < input_vector.size(); i++) printf("%d ",
  // input_vector[i]); printf("\n"); Compress an array of data using `zstd`.
  std::size_t zstd_outsize;

  void *const buffer =
      mgard::compress_memory_huffman(input_vector, zstd_outsize);

  out_data_size = zstd_outsize;

  cudaMallocHelper(handle, (void **)&out_data, out_data_size);
  cudaMemcpyAsyncHelper(handle, out_data, buffer, out_data_size, AUTO, 0);
  handle.sync(0);
  delete[] int_vector;
}

template <uint32_t D, typename T, typename S, typename H>
void cpu_lossless_decompression(Handle<D, T> &handle, H *input_data,
                                size_t input_count, S *&out_data,
                                size_t output_count) {

  // printf("cpu decompression: %llu\n", input_count);
  std::vector<unsigned char> input_vector(input_count);
  cudaMemcpyAsyncHelper(handle, input_vector.data(), input_data, input_count,
                        AUTO, 0);
  handle.sync(0);
  // printf("copy done\n");

  long int *output_vector = new long int[output_count];
  int *int_vector = new int[output_count];

  mgard::decompress_memory_huffman(
      reinterpret_cast<unsigned char *>(input_vector.data()),
      input_vector.size(), output_vector,
      output_count * sizeof(*output_vector));

  for (int i = 0; i < output_count; i++)
    int_vector[i] = output_vector[i];
  cudaMallocHelper(handle, (void **)&out_data, output_count * sizeof(S));
  cudaMemcpyAsyncHelper(handle, out_data, int_vector, output_count * sizeof(S),
                        AUTO, 0);
  handle.sync(0);
  delete[] output_vector;
  delete[] int_vector;

  // printf("dqv\n");
  // print_matrix_cuda(1, output_count, out_data, output_count);
}

#define KERNELS(D, T, S, H)                                                    \
  template void cpu_lossless_compression<D, T, S, H>(                          \
      Handle<D, T> & handle, S * input_data, size_t input_count,               \
      H * &out_data, size_t & out_data_size);                                  \
  template void cpu_lossless_decompression<D, T, S, H>(                        \
      Handle<D, T> & handle, H * input_data, size_t input_count,               \
      S * &out_data, size_t output_count);

KERNELS(1, double, int, unsigned char)
KERNELS(1, float, int, unsigned char)
KERNELS(2, double, int, unsigned char)
KERNELS(2, float, int, unsigned char)
KERNELS(3, double, int, unsigned char)
KERNELS(3, float, int, unsigned char)
KERNELS(4, double, int, unsigned char)
KERNELS(4, float, int, unsigned char)
KERNELS(5, double, int, unsigned char)
KERNELS(5, float, int, unsigned char)

} // namespace mgard_x