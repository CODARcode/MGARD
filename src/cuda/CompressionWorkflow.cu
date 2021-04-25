/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <vector>

#include "cuda/CompressionWorkflow.h"

#include "cuda/CommonInternal.h"
#include "cuda/MemoryManagement.h"

#include "cuda/DataRefactoring.h"
#include "cuda/LinearQuantization.h"
#include "cuda/LosslessCompression.h"

using namespace std::chrono;

namespace mgard_cuda {

template <typename T>
struct linf_norm : public thrust::binary_function<T, T, T> {
  __host__ __device__ T operator()(T x, T y) { return max(abs(x), abs(y)); }
};

template <uint32_t D, typename T>
Array<1, unsigned char> compress(Handle<D, T> &handle, Array<D, T> &in_array,
                                 enum error_bound_type type, T tol, T s) {

  for (int i = 0; i < D; i++) {
    if (handle.shapes_h[0][i] != in_array.getShape()[i]) {
      std::cout << log_err
                << "The shape of input array does not match the shape "
                   "initilized in handle!\n";
      std::vector<size_t> empty_shape;
      empty_shape.push_back(1);
      Array<1, unsigned char> empty(empty_shape);
      return empty;
    }
  }
  // handle.l_target = 3;
  high_resolution_clock::time_point t1, t2, start, end;
  duration<double> time_span;
  size_t free, total;

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9); cudaMemGetInfo(&free,
  // &total); printf("Mem: %f/%f\n", (double)(total-free)/1e9,
  // (double)total/1e9);

  T norm = (T)1.0;

  if (type == REL) {
    t1 = high_resolution_clock::now();
    thrust::device_vector<T> v_vec(handle.dofs[0][0] * handle.dofs[1][0] *
                                   handle.linearized_depth);
    cudaMemcpy3DAsyncHelper(
        handle, thrust::raw_pointer_cast(v_vec.data()),
        handle.dofs[0][0] * sizeof(T), handle.dofs[0][0] * sizeof(T),
        handle.dofs[1][0], in_array.get_dv(),
        in_array.get_ldvs_h()[0] * sizeof(T), handle.dofs[0][0] * sizeof(T),
        handle.dofs[1][0], handle.dofs[0][0] * sizeof(T), handle.dofs[1][0],
        handle.linearized_depth, AUTO, 0);
    handle.sync(0);
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    // printf("copy time: %.6f s \n", time_span.count());

    t1 = high_resolution_clock::now();
    norm = thrust::reduce(v_vec.begin(), v_vec.end(), (T)0, linf_norm<T>());
    // printf("norm %f\n", norm);
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
  }
  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);
  handle.allocate_workspace();
  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  start = high_resolution_clock::now();
  // Decomposition
  t1 = high_resolution_clock::now();
  decompose<D, T>(handle, in_array.get_dv(), in_array.get_ldvs_h(),
                  handle.l_target);
  // printf("sync_all 1\n");
  handle.sync_all();
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("Decomposition time: %.6f s\n", time_span.count());

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);
  // print_matrix_cuda(handle.nrow, handle.ncol, handle.nfib, dv, lddv1, lddv2,
  // handle.nfib);
  // printf("sync_all 2\n");
  handle.sync_all();
  handle.free_workspace();

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  /**** refactoring ****/
  // outsize = (handle.linearized_depth * handle.dofs[1][0] * handle.dofs[0][0])
  // * sizeof(T); unsigned char *buffer = (unsigned char *)malloc(outsize);
  // cudaMemcpy3DAsyncHelper(
  //       handle,
  //       buffer, handle.dofs[0][0] * sizeof(T), handle.dofs[0][0] * sizeof(T),
  //       handle.dofs[1][0], dv, lddv1 * sizeof(T), handle.dofs[0][0] *
  //       sizeof(T), handle.dofs[1][0], handle.dofs[0][0] * sizeof(T),
  //       handle.dofs[1][0], handle.linearized_depth, D2H, 0);

  // Quantization
  bool huffman = true;
  int dict_size = handle.huff_dict_size, block_size = handle.huff_block_size;
  size_t quantized_count =
      handle.dofs[0][0] * handle.dofs[1][0] * handle.linearized_depth;
  int *dqv;
  cudaMallocHelper((void **)&dqv, (handle.dofs[0][0] * handle.dofs[1][0] *
                                   handle.linearized_depth) *
                                      sizeof(int));

  thrust::device_vector<int> ldqvs(handle.D_padded);
  ldqvs[0] = handle.dofs[0][0];
  for (int i = 1; i < handle.D_padded; i++) {
    ldqvs[i] = handle.dofs[i][0];
  }

  t1 = high_resolution_clock::now();

  int *hshapes = new int[D * (handle.l_target + 2)];
  for (int d = 0; d < D; d++) {
    hshapes[d * (handle.l_target + 2)] = 0;
    for (int l = 1; l < handle.l_target + 2; l++) {
      hshapes[d * (handle.l_target + 2) + l] =
          handle.dofs[d][handle.l_target + 1 - l];
    }
    // printf("hshapes[%d]: ", d);
    // for (int l = 0; l < handle.l_target+2; l++) { printf("%d ", hshapes[d *
    // (handle.l_target+2)+l]); } printf("\n");
  }
  int *dshapes;
  cudaMallocHelper((void **)&dshapes, D * (handle.l_target + 2) * sizeof(int));
  cudaMemcpyAsyncHelper(handle, dshapes, hshapes,
                        D * (handle.l_target + 2) * sizeof(int), H2D, 0);

  size_t estimate_outlier_count = (double)handle.dofs[0][0] *
                                  handle.dofs[1][0] * handle.linearized_depth *
                                  1;
  // printf("estimate_outlier_count: %llu\n", estimate_outlier_count);
  size_t *outlier_count_d;
  unsigned int *outlier_idx_d;
  int *outliers;
  cudaMallocHelper((void **)&outliers, estimate_outlier_count * sizeof(int));
  cudaMallocHelper((void **)&outlier_count_d, sizeof(size_t));
  cudaMallocHelper((void **)&outlier_idx_d,
                   estimate_outlier_count * sizeof(unsigned int));
  size_t zero = 0, outlier_count, *outlier_idx_h;
  cudaMemcpyAsyncHelper(handle, outlier_count_d, &zero, sizeof(size_t), H2D, 0);

  quant_meta<T> m;
  m.norm = norm;
  m.s = s;
  m.tol = tol;
  m.dict_size = dict_size;
  m.enable_lz4 = handle.enable_lz4;
  m.l_target = handle.l_target;

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  levelwise_linear_quantize<D, T>(
      handle, dshapes, handle.l_target, m, in_array.get_dv(),
      in_array.get_ldvs_d(), dqv, thrust::raw_pointer_cast(ldqvs.data()),
      huffman, handle.shapes_d[0], outlier_count_d, outlier_idx_d, outliers, 0);

  cudaMemcpyAsyncHelper(handle, &outlier_count, outlier_count_d, sizeof(size_t),
                        D2H, 0);

  // printf("outlier_count: %llu\n", outlier_count);

  // printf("dqv\n");
  // print_matrix_cuda(1, quantized_counD, Tqv, quantized_count);

  // printf("outlier_idx_d\n");
  // print_matrix_cuda(1, outlier_count, outlier_idx_d, quantized_count);

  // printf("outliers\n");
  // print_matrix_cuda(1, outlier_count, outliers, quantized_count);

  std::vector<size_t> outlier_idx;

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("Quantization time: %.6f s\n", time_span.count());

  // cudaFreeHelper(dv);

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  // Huffman compression
  t1 = high_resolution_clock::now();
  uint64_t *hufmeta;
  uint64_t *hufdata;
  size_t hufmeta_size;
  size_t hufdata_size;
  huffman_compress<D, T, int, uint32_t, uint64_t>(
      handle, dqv, quantized_count, outlier_idx, hufmeta, hufmeta_size, hufdata,
      hufdata_size, block_size, dict_size, 0);
  // printf("sync_all 3\n");
  handle.sync_all();
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("Parallel Huffman time: %.6f s\n", time_span.count());

  cudaFreeHelper(dqv);

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  // LZ4 compression
  void *lz4_hufmeta;
  size_t lz4_hufmeta_size;
  void *lz4_hufdata;
  size_t lz4_hufdata_size;

  if (handle.enable_lz4) {
    t1 = high_resolution_clock::now();
    lz4_compress(handle, hufdata, hufdata_size / sizeof(uint64_t), lz4_hufdata,
                 lz4_hufdata_size, handle.lz4_block_size, 0);
    // printf("sync_all 4\n");
    handle.sync_all();
    cudaFreeHelper(hufdata);
    hufdata = (uint64_t *)lz4_hufdata;
    hufdata_size = lz4_hufdata_size;
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    // printf("NVComp::LZ4 time: %.6f s\n", time_span.count());

    // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
    // (double)(total-free)/1e9, (double)total/1e9);
  }

  end = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(end - start);
  // printf("Overall compression time: %.6f s (%.6f GB/s)\n", time_span.count(),
  // (double)(handle.dofs[0][0] * handle.dofs[1][0] *handle.linearized_depth
  // *sizeof(T))/time_span.count()/1e9);

  // Output serilization
  t1 = high_resolution_clock::now();

  size_t outsize = 0;
  outsize += sizeof(quant_meta<T>);
  outsize += sizeof(size_t) + outlier_count * sizeof(size_t) +
             outlier_count * sizeof(int);
  outsize += sizeof(size_t) + hufmeta_size;
  outsize += sizeof(size_t) + hufdata_size;

  std::vector<size_t> out_shape(1);
  out_shape[0] = outsize;
  Array<1, unsigned char> compressed_array(out_shape);

  unsigned char *buffer = compressed_array.get_dv();
  // cudaMallocHostHelper((void**)&buffer, outsize);
  // else cudaMallocHelper((void**)&buffer, outsize);
  // unsigned char *buffer = (unsigned char *)malloc(outsize);

  void *buffer_p = (void *)buffer;

  // memcpy(buffer_p, &m, sizeof(quant_meta<T>));
  cudaMemcpyAsyncHelper(handle, buffer_p, &m, sizeof(quant_meta<T>), AUTO, 0);
  buffer_p = buffer_p + sizeof(quant_meta<T>);

  cudaMemcpyAsyncHelper(handle, buffer_p, outlier_count_d, sizeof(size_t), AUTO,
                        0);
  buffer_p = buffer_p + sizeof(size_t);
  cudaMemcpyAsyncHelper(handle, buffer_p, outlier_idx_d,
                        outlier_count * sizeof(unsigned int), AUTO, 0);
  buffer_p = buffer_p + outlier_count * sizeof(unsigned int);
  cudaMemcpyAsyncHelper(handle, buffer_p, outliers, outlier_count * sizeof(int),
                        AUTO, 0);
  buffer_p = buffer_p + outlier_count * sizeof(int);

  // memcpy(buffer_p, &hufmeta_size, sizeof(size_t));
  cudaMemcpyAsyncHelper(handle, buffer_p, &hufmeta_size, sizeof(size_t), AUTO,
                        0);

  buffer_p = buffer_p + sizeof(size_t);
  cudaMemcpyAsyncHelper(handle, buffer_p, hufmeta, hufmeta_size, AUTO, 0);
  buffer_p = buffer_p + hufmeta_size;
  // memcpy(buffer_p, &lz4_hufmeta_size, sizeof(size_t));
  // buffer_p = buffer_p + sizeof(size_t);
  // memcpy(buffer_p, &hufdata_size, sizeof(size_t));
  cudaMemcpyAsyncHelper(handle, buffer_p, &hufdata_size, sizeof(size_t), AUTO,
                        0);
  buffer_p = buffer_p + sizeof(size_t);
  // cudaMemcpyAsyncHelper(handle, buffer_p, lz4_hufmeta, lz4_hufmeta_size, D2H,
  // 0); buffer_p = buffer_p + lz4_hufmeta_size;
  cudaMemcpyAsyncHelper(handle, buffer_p, hufdata, hufdata_size, AUTO, 0);
  buffer_p = buffer_p + hufdata_size;
  // printf("sync_all 5\n");
  handle.sync_all();
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("seralization time: %.6f\n", time_span.count());

  cudaFreeHelper(outlier_count_d);
  cudaFreeHelper(outlier_idx_d);
  cudaFreeHelper(outliers);
  cudaFreeHelper(hufmeta);
  cudaFreeHelper(hufdata);

  cudaMemGetInfo(&free, &total);
  // printf("Mem: %f/%f\n", (double)(total - free) / 1e9, (double)total / 1e9);
  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);
  return compressed_array;
}

template <uint32_t D, typename T>
Array<D, T> decompress(Handle<D, T> &handle,
                       Array<1, unsigned char> &compressed_array) {

  high_resolution_clock::time_point t1, t2, start, end;
  duration<double> time_span;

  size_t free, total;

  quant_meta<T> m;

  size_t outlier_count;
  unsigned int *outlier_idx_d;
  int *outliers;

  void *lz4_hufmeta;
  size_t lz4_hufmeta_size;
  void *lz4_hufdata;
  size_t lz4_hufdata_size;

  uint8_t *hufmeta;
  uint64_t *hufdata;
  size_t hufmeta_size;
  size_t hufdata_size;
  size_t outsize;
  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  void *data_p = compressed_array.get_dv(); //(void *)data;
  cudaMemcpyAsyncHelper(handle, &m, data_p, sizeof(quant_meta<T>), AUTO, 0);
  data_p = data_p + sizeof(quant_meta<T>);
  cudaMemcpyAsyncHelper(handle, &outlier_count, data_p, sizeof(size_t), AUTO,
                        0);
  data_p = data_p + sizeof(size_t);
  handle.sync(0);
  cudaMallocHelper((void **)&outlier_idx_d,
                   outlier_count * sizeof(unsigned int));
  cudaMemcpyAsyncHelper(handle, outlier_idx_d, data_p,
                        outlier_count * sizeof(unsigned int), AUTO, 0);
  data_p = data_p + outlier_count * sizeof(unsigned int);
  cudaMallocHelper((void **)&outliers, outlier_count * sizeof(int));
  cudaMemcpyAsyncHelper(handle, outliers, data_p, outlier_count * sizeof(int),
                        AUTO, 0);
  data_p = data_p + outlier_count * sizeof(int);
  cudaMemcpyAsyncHelper(handle, &hufmeta_size, data_p, sizeof(size_t), AUTO, 0);
  data_p = data_p + sizeof(size_t);
  handle.sync(0);
  cudaMallocHelper((void **)&hufmeta, hufmeta_size);
  cudaMemcpyAsyncHelper(handle, hufmeta, data_p, hufmeta_size, AUTO, 0);
  data_p = data_p + hufmeta_size;
  cudaMemcpyAsyncHelper(handle, &hufdata_size, data_p, sizeof(size_t), AUTO, 0);
  data_p = data_p + sizeof(size_t);
  handle.sync(0);
  cudaMallocHelper((void **)&hufdata, hufdata_size);
  cudaMemcpyAsyncHelper(handle, hufdata, data_p, hufdata_size, H2D, 0);
  data_p = data_p + hufdata_size;
  handle.sync(0);

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  start = high_resolution_clock::now();

  if (m.enable_lz4) {
    if (!handle.enable_lz4)
      printf("Warning: This data was compressed with LZ4, but handler is "
             "configed to disable LZ4!\n");
    t1 = high_resolution_clock::now();

    uint64_t *lz4_decompressed_hufdata;
    size_t lz4_decompressed_hufdata_size;
    lz4_decompress(handle, (void *)hufdata, hufdata_size,
                   lz4_decompressed_hufdata, lz4_decompressed_hufdata_size, 0);
    // printf("sync_all 6\n");
    handle.sync_all();
    cudaFreeHelper(hufdata);
    hufdata = lz4_decompressed_hufdata;
    hufdata_size = lz4_decompressed_hufdata_size;
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    // printf("NVComp::LZ4 time: %.6f s \n", time_span.count());
    // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
    // (double)(total-free)/1e9, (double)total/1e9);
  }

  size_t quantized_count =
      handle.dofs[0][0] * handle.dofs[1][0] * handle.linearized_depth;
  int *dqv;

  t1 = high_resolution_clock::now();
  huffman_decompress<D, T, int, uint32_t, uint64_t>(
      handle, (uint64_t *)hufmeta, hufmeta_size, hufdata, hufdata_size, dqv,
      outsize, 0);
  // printf("sync_all 7\n");
  handle.sync_all();
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("Parallel Huffman time: %.6f s\n", time_span.count());

  cudaFreeHelper(hufmeta);
  cudaFreeHelper(hufdata);
  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  thrust::device_vector<int> ldqvs(handle.D_padded);
  ldqvs[0] = handle.dofs[0][0];
  for (int i = 1; i < handle.D_padded; i++) {
    ldqvs[i] = handle.dofs[i][0];
  }

  thrust::device_vector<int> shape(handle.D_padded);
  for (int d = 0; d < handle.D_padded; d++) {
    shape[d] = handle.dofs[d][0];
  }

  // T *dv;
  // size_t dv_pitch;
  // cudaMalloc3DHelper((void **)&dv, &dv_pitch, handle.dofs[0][0] * sizeof(T),
  //                    handle.dofs[1][0], handle.linearized_depth);
  // int lddv1 = dv_pitch / sizeof(T);
  // int lddv2 = handle.dofs[1][0];

  // thrust::device_vector<int> ldvs(handle.D_padded);
  // ldvs[0] = lddv1;
  // for (int i = 1; i < handle.D_padded; i++) { ldvs[i] = handle.dofs[i][0]; }

  // std::vector<int> ldvs_h(handle.D_padded);
  // ldvs_h[0] = lddv1;
  // for (int i = 1; i < handle.D_padded; i++) { ldvs_h[i] = handle.dofs[i][0];
  // } int * ldvs_d; cudaMallocHelper((void **)&ldvs_d, handle.D_padded *
  // sizeof(int)); cudaMemcpyAsyncHelper(handle, ldvs_d, ldvs_h.data(),
  //   handle.D_padded * sizeof(int), H2D, 0);

  std::vector<size_t> decompressed_shape(D);
  for (int i = 0; i < D; i++)
    decompressed_shape[i] = handle.shapes_h[0][i];
  std::reverse(decompressed_shape.begin(), decompressed_shape.end());
  Array<D, T> decompressed_data(decompressed_shape);

  int *hshapes = new int[D * (handle.l_target + 2)];
  for (int d = 0; d < D; d++) {
    hshapes[d * (handle.l_target + 2)] = 0;
    for (int l = 1; l < handle.l_target + 2; l++) {
      hshapes[d * (handle.l_target + 2) + l] =
          handle.dofs[d][handle.l_target + 1 - l];
    }
    // printf("hshapes[%d]: ", d);
    // for (int l = 0; l < handle.l_target+2; l++) { printf("%d ", hshapes[d *
    // (handle.l_target+2)+l]); } printf("\n");
  }
  int *dshapes;
  cudaMallocHelper((void **)&dshapes, D * (handle.l_target + 2) * sizeof(int));
  cudaMemcpyAsyncHelper(handle, dshapes, hshapes,
                        D * (handle.l_target + 2) * sizeof(int), H2D, 0);

  // printf("sync_all 7.5\n");
  handle.sync_all();

  t1 = high_resolution_clock::now();
  levelwise_linear_dequantize<D, T>(handle, dshapes, handle.l_target, m, dqv,
                                    thrust::raw_pointer_cast(ldqvs.data()),
                                    decompressed_data.get_dv(),
                                    decompressed_data.get_ldvs_d(),
                                    outlier_count, outlier_idx_d, outliers, 0);
  // printf("sync_all 8\n");
  handle.sync_all();
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("Dequantization time: %.6f s\n", time_span.count());

  cudaFreeHelper(dqv);
  cudaFreeHelper(outlier_idx_d);
  cudaFreeHelper(outliers);
  cudaMemGetInfo(&free, &total);
  // printf("Mem: %f/%f\n", (double)(total-free)/1e9, (double)total/1e9);

  // printf("dv:\n");
  // print_matrix_cuda(1, quantized_counD, Tv, quantized_count);

  /**** refactoring ****/

  // cudaMemcpy3DAsyncHelper( handle,
  //   dv, lddv1 * sizeof(T), handle.dofs[0][0] * sizeof(T), handle.dofs[1][0],
  //     data, handle.dofs[0][0] * sizeof(T), handle.dofs[0][0] * sizeof(T),
  //     handle.dofs[1][0], handle.dofs[0][0] * sizeof(T), handle.dofs[1][0],
  //     handle.linearized_depth, H2D, 0);

  handle.allocate_workspace();

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  t1 = high_resolution_clock::now();
  recompose<D, T>(handle, decompressed_data.get_dv(),
                  decompressed_data.get_ldvs_h(), m.l_target);
  // printf("sync_all 9\n");
  handle.sync_all();
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("Recomposition time: %.6f s\n", time_span.count());

  handle.free_workspace();

  // printf("sync_all 10\n");
  handle.sync_all();
  end = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(end - start);
  // printf("Overall decompression time: %.6f s (%.6f GB/s)\n",
  // time_span.count(), (double)(handle.dofs[0][0] * handle.dofs[1][0]
  // *handle.linearized_depth *sizeof(T))/time_span.count()/1e9);

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  // T *v;
  // cudaMallocHostHelper((void **)&v, handle.dofs[0][0] * handle.dofs[1][0] *
  // handle.linearized_depth * sizeof(T));
  // // = (T *)malloc(handle.dofs[0][0] * handle.dofs[1][0] *
  // handle.linearized_depth * sizeof(T));

  // cudaMemcpy3DAsyncHelper(
  //     handle, v, handle.dofs[0][0] * sizeof(T), handle.dofs[0][0] *
  //     sizeof(T), handle.dofs[1][0], dv, lddv1 * sizeof(T), handle.dofs[0][0]
  //     * sizeof(T), handle.dofs[1][0], handle.dofs[0][0] * sizeof(T),
  //     handle.dofs[1][0], handle.linearized_depth, D2H, 0);

  // cudaFreeHelper(dv);

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);
  return decompressed_data;
}

#define KERNELS(D, T)                                                          \
  template Array<1, unsigned char> compress<D, T>(                             \
      Handle<D, T> & handle, Array<D, T> & in_array,                           \
      enum error_bound_type type, T tol, T s);                                 \
  template Array<D, T> decompress<D, T>(                                       \
      Handle<D, T> & handle, Array<1, unsigned char> & compressed_array);

KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)
KERNELS(4, double)
KERNELS(4, float)
KERNELS(5, double)
KERNELS(5, float)
#undef KERNELS

} // namespace mgard_cuda
