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

#include "cuda/mgard_cuda_compression_workflow.h"

#include "cuda/linear_quantization.h"
#include "cuda/mgard_cuda_data_refactoring.h"
#include "cuda/mgard_cuda_lossless.h"

using namespace std::chrono;

namespace mgard_cuda {

template <typename T>
struct linf_norm : public thrust::binary_function<T, T, T> {
  __host__ __device__ T operator()(T x, T y) { return max(abs(x), abs(y)); }
};

template <typename T, int D>
unsigned char *refactor_qz_cuda(mgard_cuda_handle<T, D> &handle, T *u,
                                size_t &outsize, T tol, T s) {

  // printf("l: %d\n", handle.l_target);

  // handle.l_target = 3;

  // int size_ratio = sizeof(T) / sizeof(int);
  high_resolution_clock::time_point t1, t2, start, end;
  duration<double> time_span;
  size_t free, total;

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  // t1 = high_resolution_clock::now();

  T *dv;
  size_t dv_pitch;
  cudaMalloc3DHelper((void **)&dv, &dv_pitch, handle.dofs[0][0] * sizeof(T),
                     handle.dofs[1][0], handle.linearized_depth);
  int lddv1 = dv_pitch / sizeof(T);
  int lddv2 = handle.dofs[1][0];
  thrust::device_vector<int> ldvs(handle.D_padded);
  ldvs[0] = lddv1;
  for (int i = 1; i < handle.D_padded; i++) {
    ldvs[i] = handle.dofs[i][0];
  }
  // printf("size of ldvs: %d\n", handle.D_padded);

  // printf("Allocating dqv: %llu * %llu * %llu\n", handle.dofs[0][0],
  // handle.dofs[1][0],
  //                                                 handle.linearized_depth);

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  // t2 = high_resolution_clock::now();
  // time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("init time: %.6f s \n", time_span.count());

  T norm;
  {
    // t1 = high_resolution_clock::now();
    thrust::device_vector<T> v_vec(handle.dofs[0][0] * handle.dofs[1][0] *
                                   handle.linearized_depth);
    cudaMemcpyAsyncHelper(handle, thrust::raw_pointer_cast(v_vec.data()), u,
                          handle.dofs[0][0] * handle.dofs[1][0] *
                              handle.linearized_depth * sizeof(T),
                          H2D, 0);
    handle.sync_all();
    // t2 = high_resolution_clock::now();
    // time_span = duration_cast<duration<double>>(t2 - t1);
    // printf("copy time: %.6f s \n", time_span.count());

    // t1 = high_resolution_clock::now();
    norm = thrust::reduce(v_vec.begin(), v_vec.end(), (T)0, linf_norm<T>());
    // printf("norm %f\n", norm);

    // t2 = high_resolution_clock::now();
    // time_span = duration_cast<duration<double>>(t2 - t1);
    // printf("max_norm_cuda time: %.6f s \n", time_span.count());

    // t1 = high_resolution_clock::now();
    cudaMemcpy3DAsyncHelper(
        handle, dv, lddv1 * sizeof(T), handle.dofs[0][0] * sizeof(T),
        handle.dofs[1][0], thrust::raw_pointer_cast(v_vec.data()),
        handle.dofs[0][0] * sizeof(T), handle.dofs[0][0] * sizeof(T),
        handle.dofs[1][0], handle.dofs[0][0] * sizeof(T), handle.dofs[1][0],
        handle.linearized_depth, D2D, 0);
    handle.sync_all();
    // t2 = high_resolution_clock::now();
    // time_span = duration_cast<duration<double>>(t2 - t1);
    // printf("copy time: %.6f s \n", time_span.count());
    // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
    // (double)(total-free)/1e9, (double)total/1e9);
  }

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  handle.allocate_workspace();

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  // start = high_resolution_clock::now();
  // Decomposition
  // t1 = high_resolution_clock::now();
  refactor_reo<T, D>(handle, dv, ldvs, handle.l_target);
  handle.sync_all();
  // t2 = high_resolution_clock::now();
  // time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("Decomposition time: %.6f s\n", time_span.count());

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);
  // print_matrix_cuda(handle.nrow, handle.ncol, handle.nfib, dv, lddv1, lddv2,
  // handle.nfib);

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

  // t1 = high_resolution_clock::now();

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

  levelwise_linear_quantize<T, D>(handle, dshapes, handle.l_target, m, dv,
                                  thrust::raw_pointer_cast(ldvs.data()), dqv,
                                  thrust::raw_pointer_cast(ldqvs.data()),
                                  huffman, handle.shapes_d[0], outlier_count_d,
                                  outlier_idx_d, outliers, 0);

  cudaMemcpyAsyncHelper(handle, &outlier_count, outlier_count_d, sizeof(size_t),
                        D2H, 0);

  // printf("outlier_count: %llu\n", outlier_count);

  // printf("dqv\n");
  // print_matrix_cuda(1, quantized_count, dqv, quantized_count);

  // printf("outlier_idx_d\n");
  // print_matrix_cuda(1, outlier_count, outlier_idx_d, quantized_count);

  // printf("outliers\n");
  // print_matrix_cuda(1, outlier_count, outliers, quantized_count);

  std::vector<size_t> outlier_idx;

  // t2 = high_resolution_clock::now();
  // time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("Quantization time: %.6f s\n", time_span.count());

  cudaFreeHelper(dv);

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  // Huffman compression
  // t1 = high_resolution_clock::now();
  uint64_t *hufmeta;
  uint64_t *hufdata;
  size_t hufmeta_size;
  size_t hufdata_size;
  huffman_compress<T, D, int, uint32_t, uint64_t>(
      handle, dqv, quantized_count, outlier_idx, hufmeta, hufmeta_size, hufdata,
      hufdata_size, block_size, dict_size, 0);
  handle.sync_all();
  // t2 = high_resolution_clock::now();
  // time_span = duration_cast<duration<double>>(t2 - t1);
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
    handle.sync_all();
    cudaFreeHelper(hufdata);
    hufdata = (uint64_t *)lz4_hufdata;
    hufdata_size = lz4_hufdata_size;
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    printf("NVComp::LZ4 time: %.6f s\n", time_span.count());

    // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
    // (double)(total-free)/1e9, (double)total/1e9);
  }

  // end = high_resolution_clock::now();
  // time_span = duration_cast<duration<double>>(end - start);
  // printf("Overall compression time: %.6f s (%.6f GB/s)\n", time_span.count(),
  //        (double)(handle.dofs[0][0] * handle.dofs[1][0] *
  //                 handle.linearized_depth * sizeof(T)) /
  //            time_span.count() / 1e9);

  // Output serilization
  // t1 = high_resolution_clock::now();

  outsize = 0;
  outsize += sizeof(quant_meta<T>);
  outsize += sizeof(size_t) + outlier_count * sizeof(size_t) +
             outlier_count * sizeof(int);
  outsize += sizeof(size_t) + hufmeta_size;
  outsize += sizeof(size_t) + hufdata_size;

  unsigned char *buffer = (unsigned char *)malloc(outsize);

  void *buffer_p = (void *)buffer;

  memcpy(buffer_p, &m, sizeof(quant_meta<T>));
  buffer_p = buffer_p + sizeof(quant_meta<T>);

  cudaMemcpyAsyncHelper(handle, buffer_p, outlier_count_d, sizeof(size_t), D2H,
                        0);
  buffer_p = buffer_p + sizeof(size_t);
  cudaMemcpyAsyncHelper(handle, buffer_p, outlier_idx_d,
                        outlier_count * sizeof(unsigned int), D2H, 0);
  buffer_p = buffer_p + outlier_count * sizeof(unsigned int);
  cudaMemcpyAsyncHelper(handle, buffer_p, outliers, outlier_count * sizeof(int),
                        D2H, 0);
  buffer_p = buffer_p + outlier_count * sizeof(int);

  memcpy(buffer_p, &hufmeta_size, sizeof(size_t));
  buffer_p = buffer_p + sizeof(size_t);
  cudaMemcpyAsyncHelper(handle, buffer_p, hufmeta, hufmeta_size, D2H, 0);
  buffer_p = buffer_p + hufmeta_size;
  // memcpy(buffer_p, &lz4_hufmeta_size, sizeof(size_t));
  // buffer_p = buffer_p + sizeof(size_t);
  memcpy(buffer_p, &hufdata_size, sizeof(size_t));
  buffer_p = buffer_p + sizeof(size_t);
  // cudaMemcpyAsyncHelper(handle, buffer_p, lz4_hufmeta, lz4_hufmeta_size, D2H,
  // 0); buffer_p = buffer_p + lz4_hufmeta_size;
  cudaMemcpyAsyncHelper(handle, buffer_p, hufdata, hufdata_size, D2H, 0);
  buffer_p = buffer_p + hufdata_size;
  handle.sync_all();
  // t2 = high_resolution_clock::now();
  // time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("seralization time: %.6f\n", time_span.count());

  cudaFreeHelper(outlier_count_d);
  cudaFreeHelper(outlier_idx_d);
  cudaFreeHelper(outliers);
  cudaFreeHelper(hufmeta);
  cudaFreeHelper(hufdata);

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  return (unsigned char *)buffer;
}

template <typename T, int D>
T *recompose_udq_cuda(mgard_cuda_handle<T, D> &handle, unsigned char *data,
                      size_t data_len) {

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

  void *data_p = (void *)data;
  memcpy(&m, data_p, sizeof(quant_meta<T>));
  data_p = data_p + sizeof(quant_meta<T>);

  memcpy(&outlier_count, data_p, sizeof(size_t));
  data_p = data_p + sizeof(size_t);
  cudaMallocHelper((void **)&outlier_idx_d,
                   outlier_count * sizeof(unsigned int));
  cudaMemcpyAsyncHelper(handle, outlier_idx_d, data_p,
                        outlier_count * sizeof(unsigned int), H2D, 0);
  data_p = data_p + outlier_count * sizeof(unsigned int);
  cudaMallocHelper((void **)&outliers, outlier_count * sizeof(int));
  cudaMemcpyAsyncHelper(handle, outliers, data_p, outlier_count * sizeof(int),
                        H2D, 0);
  data_p = data_p + outlier_count * sizeof(int);

  memcpy(&hufmeta_size, data_p, sizeof(size_t));
  data_p = data_p + sizeof(size_t);
  cudaMallocHelper((void **)&hufmeta, hufmeta_size);
  cudaMemcpyAsyncHelper(handle, hufmeta, data_p, hufmeta_size, H2D, 0);
  data_p = data_p + hufmeta_size;

  memcpy(&hufdata_size, data_p, sizeof(size_t));
  data_p = data_p + sizeof(size_t);

  cudaMallocHelper((void **)&hufdata, hufdata_size);
  cudaMemcpyAsyncHelper(handle, hufdata, data_p, hufdata_size, H2D, 0);
  data_p = data_p + hufdata_size;

  handle.sync_all();

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  // start = high_resolution_clock::now();

  if (m.enable_lz4) {
    if (!handle.enable_lz4)
      printf("Warning: This data was compressed with LZ4, but handler is "
             "configed to disable LZ4!\n");
    // t1 = high_resolution_clock::now();

    uint64_t *lz4_decompressed_hufdata;
    size_t lz4_decompressed_hufdata_size;
    lz4_decompress(handle, (void *)hufdata, hufdata_size,
                   lz4_decompressed_hufdata, lz4_decompressed_hufdata_size, 0);
    handle.sync_all();
    cudaFreeHelper(hufdata);
    hufdata = lz4_decompressed_hufdata;
    hufdata_size = lz4_decompressed_hufdata_size;
    // t2 = high_resolution_clock::now();
    // time_span = duration_cast<duration<double>>(t2 - t1);
    // printf("NVComp::LZ4 time: %.6f s \n", time_span.count());
    // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
    // (double)(total-free)/1e9, (double)total/1e9);
  }

  size_t quantized_count =
      handle.dofs[0][0] * handle.dofs[1][0] * handle.linearized_depth;
  int *dqv;

  // t1 = high_resolution_clock::now();
  huffman_decompress<T, D, int, uint32_t, uint64_t>(
      handle, (uint64_t *)hufmeta, hufmeta_size, hufdata, hufdata_size, dqv,
      outsize, 0);
  handle.sync_all();
  // t2 = high_resolution_clock::now();
  // time_span = duration_cast<duration<double>>(t2 - t1);
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

  T *dv;
  size_t dv_pitch;
  cudaMalloc3DHelper((void **)&dv, &dv_pitch, handle.dofs[0][0] * sizeof(T),
                     handle.dofs[1][0], handle.linearized_depth);
  int lddv1 = dv_pitch / sizeof(T);
  int lddv2 = handle.dofs[1][0];

  thrust::device_vector<int> ldvs(handle.D_padded);
  ldvs[0] = lddv1;
  for (int i = 1; i < handle.D_padded; i++) {
    ldvs[i] = handle.dofs[i][0];
  }

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

  // t1 = high_resolution_clock::now();
  levelwise_linear_dequantize<T, D>(handle, dshapes, handle.l_target, m, dqv,
                                    thrust::raw_pointer_cast(ldqvs.data()), dv,
                                    thrust::raw_pointer_cast(ldvs.data()),
                                    outlier_count, outlier_idx_d, outliers, 0);
  handle.sync_all();
  // t2 = high_resolution_clock::now();
  // time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("Dequantization time: %.6f s\n", time_span.count());

  cudaFreeHelper(dqv);
  cudaFreeHelper(outlier_idx_d);
  cudaFreeHelper(outliers);
  // cudaMemGetInfo(&free, &total);
  // printf("Mem: %f/%f\n", (double)(total-free)/1e9, (double)total/1e9);

  // printf("dv:\n");
  // print_matrix_cuda(1, quantized_count, dv, quantized_count);

  /**** refactoring ****/

  // cudaMemcpy3DAsyncHelper( handle,
  //   dv, lddv1 * sizeof(T), handle.dofs[0][0] * sizeof(T), handle.dofs[1][0],
  //     data, handle.dofs[0][0] * sizeof(T), handle.dofs[0][0] * sizeof(T),
  //     handle.dofs[1][0], handle.dofs[0][0] * sizeof(T), handle.dofs[1][0],
  //     handle.linearized_depth, H2D, 0);

  handle.allocate_workspace();

  // cudaMemGetInfo(&free, &total);
  // printf("Mem: %f/%f\n", (double)(total - free) / 1e9, (double)total / 1e9);

  // t1 = high_resolution_clock::now();
  recompose_reo<T, D>(handle, dv, ldvs, m.l_target);
  handle.sync_all();
  // t2 = high_resolution_clock::now();
  // time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("Recomposition time: %.6f s\n", time_span.count());

  handle.free_workspace();

  // handle.sync_all();
  // end = high_resolution_clock::now();
  // time_span = duration_cast<duration<double>>(end - start);
  // printf("Overall decompression time: %.6f s (%.6f GB/s)\n", time_span.count(),
  //        (double)(handle.dofs[0][0] * handle.dofs[1][0] *
  //                 handle.linearized_depth * sizeof(T)) /
  //            time_span.count() / 1e9);

  // cudaMemGetInfo(&free, &total);
  // printf("Mem: %f/%f\n", (double)(total - free) / 1e9, (double)total / 1e9);

  T *v = (T *)malloc(handle.dofs[0][0] * handle.dofs[1][0] *
                     handle.linearized_depth * sizeof(T));

  cudaMemcpy3DAsyncHelper(handle, v, handle.dofs[0][0] * sizeof(T),
                          handle.dofs[0][0] * sizeof(T), handle.dofs[1][0], dv,
                          lddv1 * sizeof(T), handle.dofs[0][0] * sizeof(T),
                          handle.dofs[1][0], handle.dofs[0][0] * sizeof(T),
                          handle.dofs[1][0], handle.linearized_depth, D2H, 0);

  cudaFreeHelper(dv);

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);
  return v;
}

#define KERNELS(T, D)                                                          \
  template unsigned char *refactor_qz_cuda<T, D>(                              \
      mgard_cuda_handle<T, D> & handle, T * u, size_t & outsize, T tol, T s);  \
  template T *recompose_udq_cuda<T, D>(mgard_cuda_handle<T, D> & handle,       \
                                       unsigned char *data, size_t data_len);

KERNELS(double, 1)
KERNELS(float, 1)
KERNELS(double, 2)
KERNELS(float, 2)
KERNELS(double, 3)
KERNELS(float, 3)
KERNELS(double, 4)
KERNELS(float, 4)
KERNELS(double, 5)
KERNELS(float, 5)
#undef KERNELS

} // namespace mgard_cuda
