/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

// #include "compressors.hpp"
#include <zstd.h>
#include "cuda/Common.h"
#include "cuda/CommonInternal.h"
#include "cuda/LosslessCompression.h"
#include "cuda/ParallelHuffman/huffman_workflow.cuh"
#include <typeinfo>

namespace mgard {
  void huffman_encoding(long int *quantized_data, const std::size_t n,
                      unsigned char **out_data_hit, size_t *out_data_hit_size,
                      unsigned char **out_data_miss, size_t *out_data_miss_size,
                      unsigned char **out_tree, size_t *out_tree_size);
  void huffman_decoding(long int *quantized_data,
                      const std::size_t quantized_data_size,
                      unsigned char *out_data_hit, size_t out_data_hit_size,
                      unsigned char *out_data_miss, size_t out_data_miss_size,
                      unsigned char *out_tree, size_t out_tree_size);
}

namespace mgard_cuda {

/*! CHECK
 * Check that the condition holds. If it doesn't print a message and die.
 */
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d CHECK(%s) failed: ", __FILE__, __LINE__, #cond);  \
      fprintf(stderr, "" __VA_ARGS__);                                         \
      fprintf(stderr, "\n");                                                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/*! CHECK_ZSTD
 * Check the zstd error code and die if an error occurred after printing a
 * message.
 */
/*! CHECK_ZSTD
 * Check the zstd error code and die if an error occurred after printing a
 * message.
 */
#define CHECK_ZSTD(fn, ...)                                                    \
  do {                                                                         \
    size_t const err = (fn);                                                   \
    CHECK(!ZSTD_isError(err), "%s", ZSTD_getErrorName(err));                   \
  } while (0)

unsigned char * compress_memory_huffman(long int *const src,
                                                    const std::size_t srcLen,
                                                    std::size_t outsize) {
  unsigned char *out_data_hit = 0;
  size_t out_data_hit_size;
  unsigned char *out_data_miss = 0;
  size_t out_data_miss_size;
  unsigned char *out_tree = 0;
  size_t out_tree_size;
  mgard::huffman_encoding(src, srcLen, &out_data_hit, &out_data_hit_size,
                   &out_data_miss, &out_data_miss_size, &out_tree,
                   &out_tree_size);

  const size_t total_size =
      out_data_hit_size / 8 + 4 + out_data_miss_size + out_tree_size;
  unsigned char *payload = (unsigned char *)malloc(total_size);
  unsigned char *bufp = payload;

  if (out_tree_size) {
    std::memcpy(bufp, out_tree, out_tree_size);
    bufp += out_tree_size;
  }

  std::memcpy(bufp, out_data_hit, out_data_hit_size / 8 + 4);
  bufp += out_data_hit_size / 8 + 4;

  if (out_data_miss_size) {
    std::memcpy(bufp, out_data_miss, out_data_miss_size);
    bufp += out_data_miss_size;
  }

  free(out_tree);
  free(out_data_hit);
  free(out_data_miss);

  // const MemoryBuffer<unsigned char> out_data =
  //     compress_memory_zstd(payload, total_size);

  const size_t cBuffSize = ZSTD_compressBound(total_size);
  unsigned char *const zstd_buffer = new unsigned char[cBuffSize];
  const std::size_t cSize = ZSTD_compress(zstd_buffer, cBuffSize, payload, total_size, 1);
  CHECK_ZSTD(cSize);
  // return MemoryBuffer<unsigned char>(buffer, cSize);

  free(payload);
  payload = 0;

  const std::size_t bufferLen = 3 * sizeof(size_t) + cSize;
  unsigned char *const buffer = new unsigned char[bufferLen];
  outsize = bufferLen;

  bufp = buffer;
  *(size_t *)bufp = out_tree_size;
  bufp += sizeof(size_t);

  *(size_t *)bufp = out_data_hit_size;
  bufp += sizeof(size_t);

  *(size_t *)bufp = out_data_miss_size;
  bufp += sizeof(size_t);

  {
    unsigned char const *const p = zstd_buffer;
    std::copy(p, p + cSize, bufp);
  }
  // return MemoryBuffer<unsigned char>(buffer, bufferLen);
  return buffer;
}


void decompress_memory_huffman(unsigned char *const src,
                               const std::size_t srcLen, long int *const dst,
                               const std::size_t dstLen) {
  unsigned char *out_data_hit = 0;
  size_t out_data_hit_size;
  unsigned char *out_data_miss = 0;
  size_t out_data_miss_size;
  unsigned char *out_tree = 0;
  size_t out_tree_size;

  unsigned char *buf = src;

  out_tree_size = *(size_t *)buf;
  buf += sizeof(size_t);

  out_data_hit_size = *(size_t *)buf;
  buf += sizeof(size_t);

  out_data_miss_size = *(size_t *)buf;
  buf += sizeof(size_t);
  size_t total_huffman_size =
      out_tree_size + out_data_hit_size / 8 + 4 + out_data_miss_size;
  unsigned char *huffman_encoding_p =
      (unsigned char *)malloc(total_huffman_size);
  // decompress_memory_zstd(buf, srcLen - 3 * sizeof(size_t), huffman_encoding_p,
  //                        total_huffman_size);

  size_t const dSize = ZSTD_decompress(huffman_encoding_p, total_huffman_size, buf, srcLen - 3 * sizeof(size_t));
  CHECK_ZSTD(dSize);

  /* When zstd knows the content size, it will error if it doesn't match. */
  CHECK(dstLen == dSize, "Impossible because zstd will check this condition!");


  out_tree = huffman_encoding_p;
  out_data_hit = huffman_encoding_p + out_tree_size;
  out_data_miss =
      huffman_encoding_p + out_tree_size + out_data_hit_size / 8 + 4;

  mgard::huffman_decoding(dst, dstLen, out_data_hit, out_data_hit_size, out_data_miss,
                   out_data_miss_size, out_tree, out_tree_size);

  free(huffman_encoding_p);
}


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

template <uint32_t D, typename T, typename S, typename Q, typename H>
void huffman_compress(Handle<D, T> &handle, S *input_data, size_t input_count,
                      std::vector<size_t> &outlier_idx, H *&out_meta,
                      size_t &out_meta_size, H *&out_data,
                      size_t &out_data_size, int chunk_size, int dict_size,
                      int queue_idx) {

  HuffmanEncode<D, T, S, Q, H>(handle, input_data, input_count, outlier_idx,
                               out_meta, out_meta_size, out_data, out_data_size,
                               chunk_size, dict_size);
}

template <uint32_t D, typename T, typename S, typename Q, typename H>
void huffman_decompress(Handle<D, T> &handle, H *in_meta, size_t in_meta_size,
                        H *in_data, size_t in_data_size, S *&output_data,
                        size_t &output_count, int queue_idx) {
  HuffmanDecode<D, T, S, Q, H>(handle, output_data, output_count, in_meta,
                               in_meta_size, in_data, in_data_size);
}

#define KERNELS(D, T, S, Q, H)                                                 \
  template void huffman_compress<D, T, S, Q, H>(                               \
      Handle<D, T> & handle, S * input_data, size_t input_count,               \
      std::vector<size_t> & outlier_idx, H * &out_meta,                        \
      size_t & out_meta_size, H * &out_data, size_t & out_data_size,           \
      int chunk_size, int dict_size, int queue_idx);                           \
  template void huffman_decompress<D, T, S, Q, H>(                             \
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

  unsigned char * buffer =
      compress_memory_huffman(input_vector.data(), input_vector.size() * sizeof(long int), zstd_outsize);

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

  decompress_memory_huffman(
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

} // namespace mgard_cuda