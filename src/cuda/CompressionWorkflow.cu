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

#include "cuda/CommonInternal.h"

#include "cuda/CompressionWorkflow.h"

#include "cuda/MemoryManagement.h"

#include "cuda/DataRefactoring.h"
#include "cuda/LinearQuantization.h"
#include "cuda/LosslessCompression.h"

#define BLOCK_SIZE 64

using namespace std::chrono;

namespace mgard_cuda {

template <typename T>
struct linf_norm : public thrust::binary_function<T, T, T> {
  __host__ __device__ T operator()(T x, T y) { return max(abs(x), abs(y)); }
};

template <typename T> struct l2_norm : public thrust::unary_function<T, T> {
  __host__ __device__ T operator()(T x) { return x * x; }
};

template <DIM D, typename T>
Array<1, unsigned char> compress(Handle<D, T> &handle, Array<D, T> &in_array,
                                 enum error_bound_type type, T tol, T s) {

  cudaSetDeviceHelper(handle.dev_id);

  for (DIM i = 0; i < D; i++) {
    if (handle.shapes_h[0][i] != in_array.getShape()[i]) {
      std::cout << log::log_err
                << "The shape of input array does not match the shape "
                   "initilized in handle!\n";
      std::vector<SIZE> empty_shape;
      empty_shape.push_back(1);
      Array<1, unsigned char> empty(empty_shape);
      return empty;
    }
  }
  // handle.l_target = 3;
  high_resolution_clock::time_point t1, t2, start, end;
  duration<double> time_span;
  size_t free, total;

  // cudaMemGetInfo(&free, &total);
  // printf("Mem: %f/%f\n", (double)(total-free)/1e9, (double)total/1e9);

  if (handle.timing)
    start = high_resolution_clock::now();
  T norm = (T)1.0;

  if (type == REL) {
    // printf("Calculate norm\n");
    if (handle.timing)
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
    if (s == std::numeric_limits<T>::infinity()) {
      norm = thrust::reduce(v_vec.begin(), v_vec.end(), (T)0, linf_norm<T>());
    } else {
      thrust::transform(v_vec.begin(), v_vec.end(), v_vec.begin(),
                        l2_norm<T>());
      norm = thrust::reduce(v_vec.begin(), v_vec.end(), (T)0);
      norm = std::sqrt(norm);
    }
    if (handle.timing) {
      t2 = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(t2 - t1);
      std::cout << log::log_time << "Calculating norm using NVIDIA::Thrust: "
                << time_span.count() << " s\n";
    }
  }
  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);
  handle.allocate_workspace();
  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  // Decomposition
  if (handle.timing)
    t1 = high_resolution_clock::now();
  decompose<D, T>(handle, in_array.get_dv(), in_array.get_ldvs_h(),
                  in_array.get_ldvs_d(), handle.l_target, 0);
  handle.sync_all();
  if (handle.timing) {
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << log::log_time << "decomposition time: " << time_span.count()
              << " s\n";
  }

  // /////test
  // if (0){
  //   int block_size = BLOCK_SIZE;
  //   int queue_idx = 0;
  //   mgard_cuda::Handle<3, T> **** block_handle = new mgard_cuda::Handle<3,
  //   T>***[(int)std::ceil((float)handle.dofs[0][0]/block_size)]; for (int i =
  //   0; i < handle.dofs[0][0]; i += block_size) {
  //     block_handle[i/block_size] = new mgard_cuda::Handle<3,
  //     T>**[(int)std::ceil((float)handle.dofs[1][0]/block_size)]; for (int j =
  //     0; j < handle.dofs[1][0]; j += block_size) {
  //       block_handle[i/block_size][j/block_size] = new mgard_cuda::Handle<3,
  //       T>*[(int)std::ceil((float)handle.dofs[2][0]/block_size)]; for (int k
  //       = 0; k < handle.dofs[2][0]; k += block_size) {
  //         size_t b0 = std::min(block_size, handle.dofs[0][0] - i);
  //         size_t b1 = std::min(block_size, handle.dofs[1][0] - j);
  //         size_t b2 = std::min(block_size, handle.dofs[2][0] - k);
  //         std::vector<size_t> block_shape = {b2, b1, b0};
  //         block_handle[i/block_size][j/block_size][k/block_size] = new
  //         mgard_cuda::Handle<3, T>(block_shape);
  //         block_handle[i/block_size][j/block_size][k/block_size]->allocate_workspace();
  //       }
  //     }
  //   }

  //   t1 = high_resolution_clock::now();
  //   for (int i = 0; i < handle.dofs[0][0]; i += block_size) {
  //     for (int j = 0; j < handle.dofs[1][0]; j += block_size) {
  //       for (int k = 0; k < handle.dofs[2][0]; k += block_size) {
  //         size_t b0 = std::min(block_size, handle.dofs[0][0] - i);
  //         size_t b1 = std::min(block_size, handle.dofs[1][0] - j);
  //         size_t b2 = std::min(block_size, handle.dofs[2][0] - k);
  //         std::vector<size_t> block_shape = {b2, b1, b0};
  //         std::vector<int> idx = {(int)i, (int)j, (int)k};
  //         decompose<3,
  //         T>(*(block_handle[i/block_size][j/block_size][k/block_size]),
  //                         in_array.get_dv()+get_idx(in_array.get_ldvs_h(),
  //                         idx), in_array.get_ldvs_h(), in_array.get_ldvs_d(),
  //                 block_handle[i/block_size][j/block_size][k/block_size]->l_target,
  //                 0);
  //         block_handle[i/block_size][j/block_size][k/block_size]->sync_all();
  //       }
  //     }
  //   }

  // for (int i = 0; i < handle.dofs[0][0]; i += block_size) {
  //   for (int j = 0; j < handle.dofs[1][0]; j += block_size) {
  //     for (int k = 0; k < handle.dofs[2][0]; k += block_size) {
  //       block_handle[i/block_size][j/block_size][k/block_size]->sync_all();
  //     }
  //   }
  // }

  //   t2 = high_resolution_clock::now();
  //   time_span = duration_cast<duration<double>>(t2 - t1);
  //   printf("Blocked Decomposition time: %.6f s\n", time_span.count());

  //   for (int i = 0; i < handle.dofs[0][0]; i += block_size) {
  //     for (int j = 0; j < handle.dofs[1][0]; j += block_size) {
  //       for (int k = 0; k < handle.dofs[2][0]; k += block_size) {

  //         block_handle[i/block_size][j/block_size][k/block_size]->free_workspace();
  //       }
  //     }
  //   }
  // }

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);
  // printf("sync_all 2\n");
  // handle.sync_all();
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
  if (handle.timing)
    t1 = high_resolution_clock::now();
  bool prep_huffman = handle.gpu_lossless; // Disable preparation for huffman
                                           // when we do lossless on CPU
  SIZE dict_size = handle.huff_dict_size, block_size = handle.huff_block_size;
  LENGTH quantized_count =
      handle.dofs[0][0] * handle.dofs[1][0] * handle.linearized_depth;
  QUANTIZED_INT *dqv;
  cudaMallocHelper(
      handle, (void **)&dqv,
      (handle.dofs[0][0] * handle.dofs[1][0] * handle.linearized_depth) *
          sizeof(QUANTIZED_INT));

  thrust::device_vector<SIZE> ldqvs(handle.D_padded);
  ldqvs[0] = handle.dofs[0][0];
  for (int i = 1; i < handle.D_padded; i++) {
    ldqvs[i] = handle.dofs[i][0];
  }

  LENGTH estimate_outlier_count = (double)handle.dofs[0][0] *
                                  handle.dofs[1][0] * handle.linearized_depth *
                                  1;
  // printf("estimate_outlier_count: %llu\n", estimate_outlier_count);
  LENGTH *outlier_count_d;
  LENGTH *outlier_idx_d;
  QUANTIZED_INT *outliers;
  cudaMallocHelper(handle, (void **)&outliers,
                   estimate_outlier_count * sizeof(QUANTIZED_INT));
  cudaMallocHelper(handle, (void **)&outlier_count_d, sizeof(LENGTH));
  cudaMallocHelper(handle, (void **)&outlier_idx_d,
                   estimate_outlier_count * sizeof(LENGTH));
  LENGTH zero = 0, outlier_count, *outlier_idx_h;
  cudaMemcpyAsyncHelper(handle, outlier_count_d, &zero, sizeof(LENGTH), H2D, 0);

  Metadata m;
  m.total_dims = D;
  m.shape = new SIZE[D];
  for (int d = 0; d < D; d++)
    m.shape[d] = handle.dofs[D - 1 - d][0];
  m.dtype =
      std::is_same<T, double>::value ? data_type::Double : data_type::Float;
  m.norm = norm;
  m.s = s;
  m.tol = tol;
  m.dict_size = dict_size;
  m.enable_lz4 = handle.enable_lz4;
  m.l_target = handle.l_target;
  m.gpu_lossless = handle.gpu_lossless;

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  levelwise_linear_quantize<D, T>(
      handle, handle.ranges_d, handle.l_target, handle.volumes,
      handle.ldvolumes, m, in_array.get_dv(), in_array.get_ldvs_d(), dqv,
      thrust::raw_pointer_cast(ldqvs.data()), prep_huffman, handle.shapes_d[0],
      outlier_count_d, outlier_idx_d, outliers, 0);

  cudaMemcpyAsyncHelper(handle, &outlier_count, outlier_count_d, sizeof(LENGTH),
                        D2H, 0);

  // printf("outlier_count: %llu\n", outlier_count);

  // printf("dqv\n");
  // print_matrix_cuda(1, quantized_count, dqv, quantized_count);

  // printf("outlier_idx_d\n");
  // print_matrix_cuda(1, outlier_count, outlier_idx_d, quantized_count);

  // printf("outliers\n");
  // print_matrix_cuda(1, outlier_count, outliers, quantized_count);

  std::vector<size_t> outlier_idx;

  if (handle.timing) {
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << log::log_time << "Quantization time: " << time_span.count()
              << " s\n";
  }

  // cudaFreeHelper(dv);

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);
  if (handle.gpu_lossless) {
    // printf("gpu lossless\n");
    // Huffman compression
    if (handle.timing)
      t1 = high_resolution_clock::now();
    uint64_t *hufmeta;
    uint64_t *hufdata;
    size_t hufmeta_size;
    size_t hufdata_size;
    huffman_compress<D, T, int, DIM, uint64_t>(
        handle, dqv, quantized_count, outlier_idx, hufmeta, hufmeta_size,
        hufdata, hufdata_size, block_size, dict_size, 0);
    // printf("sync_all 3\n");
    handle.sync_all();
    cudaFreeHelper(dqv);
    if (handle.timing) {
      t2 = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(t2 - t1);
      std::cout << log::log_time
                << "GPU Huffman encoding time: " << time_span.count() << " s\n";
    }

    // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
    // (double)(total-free)/1e9, (double)total/1e9);

    // LZ4 compression
    void *lz4_hufmeta;
    size_t lz4_hufmeta_size;
    void *lz4_hufdata;
    size_t lz4_hufdata_size;

    if (handle.enable_lz4) {
      if (handle.timing)
        t1 = high_resolution_clock::now();
      lz4_compress(handle, hufdata, hufdata_size / sizeof(uint64_t),
                   lz4_hufdata, lz4_hufdata_size, handle.lz4_block_size, 0);
      // printf("sync_all 4\n");
      handle.sync_all();
      cudaFreeHelper(hufdata);
      hufdata = (uint64_t *)lz4_hufdata;
      hufdata_size = lz4_hufdata_size;
      if (handle.timing) {
        t2 = high_resolution_clock::now();
        time_span = duration_cast<duration<double>>(t2 - t1);
        std::cout << log::log_time
                  << "NVComp::LZ4 compression time: " << time_span.count()
                  << " s\n";
      }

      // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
      // (double)(total-free)/1e9, (double)total/1e9);
    }

    if (handle.timing) {
      end = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(end - start);
      std::cout << log::log_time
                << "Overall compression time: " << time_span.count() << " s ("
                << (double)(handle.dofs[0][0] * handle.dofs[1][0] *
                            handle.linearized_depth * sizeof(T)) /
                       time_span.count() / 1e9
                << " GB/s)\n";
    }

    // Output serilization
    if (handle.timing)
      t1 = high_resolution_clock::now();

    SIZE meta_size;
    SERIALIZED_TYPE *serizalied_meta = m.Serialize(meta_size);
    SIZE outsize = 0;
    outsize += meta_size;
    outsize += sizeof(LENGTH) + outlier_count * sizeof(LENGTH) +
               outlier_count * sizeof(QUANTIZED_INT);
    outsize += sizeof(size_t) + hufmeta_size;
    outsize += sizeof(size_t) + hufdata_size;

    std::vector<SIZE> out_shape(1);
    out_shape[0] = outsize;
    gpuErrchk(cudaDeviceSynchronize());
    Array<1, unsigned char> compressed_array(out_shape);
    SERIALIZED_TYPE *buffer = compressed_array.get_dv();
    void *buffer_p = (void *)buffer;

    cudaMemcpyAsyncHelper(handle, buffer_p, serizalied_meta, meta_size, AUTO,
                          0);
    buffer_p = buffer_p + meta_size;
    cudaMemcpyAsyncHelper(handle, buffer_p, outlier_count_d, sizeof(LENGTH),
                          AUTO, 0);
    buffer_p = buffer_p + sizeof(LENGTH);
    cudaMemcpyAsyncHelper(handle, buffer_p, outlier_idx_d,
                          outlier_count * sizeof(LENGTH), AUTO, 0);
    buffer_p = buffer_p + outlier_count * sizeof(LENGTH);
    cudaMemcpyAsyncHelper(handle, buffer_p, outliers,
                          outlier_count * sizeof(QUANTIZED_INT), AUTO, 0);
    buffer_p = buffer_p + outlier_count * sizeof(QUANTIZED_INT);

    // memcpy(buffer_p, &hufmeta_size, sizeof(size_t));
    cudaMemcpyAsyncHelper(handle, buffer_p, &hufmeta_size, sizeof(size_t), AUTO,
                          0);

    buffer_p = buffer_p + sizeof(size_t);
    cudaMemcpyAsyncHelper(handle, buffer_p, hufmeta, hufmeta_size, AUTO, 0);
    buffer_p = buffer_p + hufmeta_size;

    cudaMemcpyAsyncHelper(handle, buffer_p, &hufdata_size, sizeof(size_t), AUTO,
                          0);
    buffer_p = buffer_p + sizeof(size_t);

    cudaMemcpyAsyncHelper(handle, buffer_p, hufdata, hufdata_size, AUTO, 0);
    buffer_p = buffer_p + hufdata_size;
    // printf("sync_all 5\n");
    handle.sync_all();
    if (handle.timing) {
      t2 = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(t2 - t1);
      std::cout << log::log_time
                << "Compressed output seralization time: " << time_span.count()
                << " s\n";
    }

    delete serizalied_meta;
    cudaFreeHelper(outlier_count_d);
    cudaFreeHelper(outlier_idx_d);
    cudaFreeHelper(outliers);
    cudaFreeHelper(hufmeta);
    cudaFreeHelper(hufdata);

    // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
    // (double)(total-free)/1e9, (double)total/1e9);
    return compressed_array;
  } else { // cpu lossless
    // printf("cpu lossless\n");
    if (handle.timing)
      t1 = high_resolution_clock::now();
    unsigned char *cpu_lossless_data; // on GPU memory
    size_t cpu_lossless_size;
    cpu_lossless_compression(handle, dqv, quantized_count, cpu_lossless_data,
                             cpu_lossless_size);
    cudaFreeHelper(dqv);
    if (handle.timing) {
      t2 = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(t2 - t1);
      std::cout << log::log_time
                << "CPU lossless compression time: " << time_span.count()
                << " s\n";
    }

    if (handle.timing) {
      end = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(end - start);
      std::cout << log::log_time
                << "Overall compression time: " << time_span.count() << " s ("
                << (double)(handle.dofs[0][0] * handle.dofs[1][0] *
                            handle.linearized_depth * sizeof(T)) /
                       time_span.count() / 1e9
                << " GB/s)\n";
    }

    if (handle.timing)
      t1 = high_resolution_clock::now();

    SIZE meta_size;
    SERIALIZED_TYPE *serizalied_meta = m.Serialize(meta_size);

    SIZE outsize = 0;
    outsize += meta_size;
    outsize += sizeof(size_t) + cpu_lossless_size;
    // printf("cpu_lossless_size: %llu\n", cpu_lossless_size);
    std::vector<SIZE> out_shape(1);
    out_shape[0] = outsize;
    Array<1, unsigned char> compressed_array(out_shape);

    unsigned char *buffer = compressed_array.get_dv();
    // cudaMallocHostHelper((void**)&buffer, outsize);
    // else cudaMallocHelper((void**)&buffer, outsize);
    // unsigned char *buffer = (unsigned char *)malloc(outsize);

    void *buffer_p = (void *)buffer;
    cudaMemcpyAsyncHelper(handle, buffer_p, serizalied_meta, meta_size, AUTO,
                          0);
    buffer_p = buffer_p + meta_size;
    cudaMemcpyAsyncHelper(handle, buffer_p, &cpu_lossless_size, sizeof(size_t),
                          AUTO, 0);
    buffer_p = buffer_p + sizeof(size_t);
    cudaMemcpyAsyncHelper(handle, buffer_p, cpu_lossless_data,
                          cpu_lossless_size, AUTO, 0);
    buffer_p = buffer_p + cpu_lossless_size;

    delete[] serizalied_meta;
    cudaFreeHelper(cpu_lossless_data);
    if (handle.timing) {
      t2 = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(t2 - t1);
      std::cout << log::log_time
                << "Compressed data serialization time: " << time_span.count()
                << " s\n";
    }

    // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
    //(double)(total-free)/1e9, (double)total/1e9);
    return compressed_array;
  }
}

template <DIM D, typename T>
Array<D, T> decompress(Handle<D, T> &handle,
                       Array<1, unsigned char> &compressed_array) {

  cudaSetDeviceHelper(handle.dev_id);
  high_resolution_clock::time_point t1, t2, start, end;
  duration<double> time_span;

  size_t free, total;

  QUANTIZED_INT *dqv;
  LENGTH quantized_count =
      handle.dofs[0][0] * handle.dofs[1][0] * handle.linearized_depth;

  LENGTH outlier_count;
  LENGTH *outlier_idx_d;
  QUANTIZED_INT *outliers;

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  //   (double)(total-free)/1e9, (double)total/1e9);

  void *data_p = compressed_array.get_dv(); //(void *)data;

  Metadata m;
  SIZE meta_size;
  cudaMemcpyAsyncHelper(handle, &meta_size, data_p, sizeof(SIZE), AUTO, 0);
  SERIALIZED_TYPE *serizalied_meta = (SERIALIZED_TYPE *)std::malloc(meta_size);
  cudaMemcpyAsyncHelper(handle, serizalied_meta, data_p, meta_size, AUTO, 0);
  data_p = data_p + meta_size;
  m.Deserialize(serizalied_meta, meta_size);

  if (strcmp(m.signature, SIGNATURE) != 0) {
    std::cout << log::log_err
              << "This data was not compressed with MGARD-CUDA or corrupted!\n";
    exit(-1);
  }

  // printf("m.cpu_lossless: %d\n", m.cpu_lossless);
  if (m.gpu_lossless) {
    // printf("gpu lossless\n");
    if (handle.timing)
      t1 = high_resolution_clock::now();
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

    cudaMemcpyAsyncHelper(handle, &outlier_count, data_p, sizeof(LENGTH), AUTO,
                          0);
    data_p = data_p + sizeof(LENGTH);
    handle.sync(0);
    cudaMallocHelper(handle, (void **)&outlier_idx_d,
                     outlier_count * sizeof(LENGTH));
    cudaMemcpyAsyncHelper(handle, outlier_idx_d, data_p,
                          outlier_count * sizeof(LENGTH), AUTO, 0);
    // outlier_idx_d = (LENGTH *) data_p;
    data_p = data_p + outlier_count * sizeof(LENGTH);
    cudaMallocHelper(handle, (void **)&outliers,
                     outlier_count * sizeof(QUANTIZED_INT));
    cudaMemcpyAsyncHelper(handle, outliers, data_p,
                          outlier_count * sizeof(QUANTIZED_INT), AUTO, 0);
    // outliers = (QUANTIZED_INT *) data_p;
    data_p = data_p + outlier_count * sizeof(QUANTIZED_INT);
    cudaMemcpyAsyncHelper(handle, &hufmeta_size, data_p, sizeof(size_t), AUTO,
                          0);
    data_p = data_p + sizeof(size_t);
    handle.sync(0);

    cudaMallocHelper(handle, (void **)&hufmeta, hufmeta_size);
    cudaMemcpyAsyncHelper(handle, hufmeta, data_p, hufmeta_size, AUTO, 0);
    // hufmeta = (uint8_t *)data_p;
    data_p = data_p + hufmeta_size;
    cudaMemcpyAsyncHelper(handle, &hufdata_size, data_p, sizeof(size_t), AUTO,
                          0);
    data_p = data_p + sizeof(size_t);
    handle.sync(0);
    cudaMallocHelper(handle, (void **)&hufdata, hufdata_size);
    cudaMemcpyAsyncHelper(handle, hufdata, data_p, hufdata_size, H2D, 0);
    // hufdata = (uint64_t *)data_p;
    data_p = data_p + hufdata_size;
    handle.sync(0);

    // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
    // (double)(total-free)/1e9, (double)total/1e9);

    if (handle.timing) {
      t2 = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(t2 - t1);
      std::cout << log::log_time
                << "Compressed data deserialization time: " << time_span.count()
                << " s\n";
    }

    if (handle.timing)
      start = high_resolution_clock::now();

    if (m.enable_lz4) {
      if (!handle.enable_lz4)
        std::cout
            << log::log_warn
            << "Warning: This data was compressed with LZ4, but handler is "
               "configed to disable LZ4! Enabling LZ4 decompressor.\n";
      if (handle.timing)
        t1 = high_resolution_clock::now();

      uint64_t *lz4_decompressed_hufdata;
      size_t lz4_decompressed_hufdata_size;
      lz4_decompress(handle, (void *)hufdata, hufdata_size,
                     lz4_decompressed_hufdata, lz4_decompressed_hufdata_size,
                     0);
      // printf("sync_all 6\n");
      handle.sync_all();
      cudaFreeHelper(hufdata);
      hufdata = lz4_decompressed_hufdata;
      hufdata_size = lz4_decompressed_hufdata_size;
      t2 = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(t2 - t1);
      if (handle.timing) {
        std::cout << log::log_time
                  << "NVComp::LZ4 decompression time: " << time_span.count()
                  << " s\n";
      }
    }

    if (handle.timing)
      t1 = high_resolution_clock::now();
    huffman_decompress<D, T, int, DIM, uint64_t>(handle, (uint64_t *)hufmeta,
                                                 hufmeta_size, hufdata,
                                                 hufdata_size, dqv, outsize, 0);
    handle.sync_all();
    cudaFreeHelper(hufmeta);
    cudaFreeHelper(hufdata);
    if (handle.timing) {
      t2 = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(t2 - t1);
      std::cout << log::log_time
                << "GPU Huffman decoding time: " << time_span.count() << " s\n";
    }

    // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
    // (double)(total-free)/1e9, (double)total/1e9);

  } else { // cpu lossless
    // printf("cpu lossless\n");
    // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
    //(double)(total-free)/1e9, (double)total/1e9);
    if (handle.gpu_lossless)
      std::cout << log::log_warn
                << "This data was compressed with CPU lossless compressors, "
                   "but handler is "
                   "configed to use GPU lossless compressors! Switching to GPU "
                   "lossless decompressor.\n";
    if (handle.timing)
      t1 = high_resolution_clock::now();
    unsigned char *cpu_lossless_data; // on GPU memory
    size_t cpu_lossless_size;
    cudaMemcpyAsyncHelper(handle, &cpu_lossless_size, data_p, sizeof(size_t),
                          AUTO, 0);
    data_p = data_p + sizeof(size_t);
    handle.sync(0);
    // cudaMallocHelper(handle, (void **)&cpu_lossless_data,
    //                 cpu_lossless_size * sizeof(unsigned char));
    // cudaMemcpyAsyncHelper(handle, cpu_lossless_data, data_p,
    //                      cpu_lossless_size * sizeof(unsigned char), AUTO, 0);
    cpu_lossless_data = (unsigned char *)data_p;
    cpu_lossless_decompression(handle, cpu_lossless_data, cpu_lossless_size,
                               dqv, quantized_count);
    // cudaFreeHelper(cpu_lossless_data);
    if (handle.timing) {
      t2 = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(t2 - t1);
      std::cout << log::log_time
                << "CPU lossless decompression time: " << time_span.count()
                << " s\n";
    }
    // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
    //(double)(total-free)/1e9, (double)total/1e9);
  }

  if (handle.timing)
    t1 = high_resolution_clock::now();
  thrust::device_vector<SIZE> ldqvs(handle.D_padded);
  ldqvs[0] = handle.dofs[0][0];
  for (int i = 1; i < handle.D_padded; i++) {
    ldqvs[i] = handle.dofs[i][0];
  }

  std::vector<SIZE> decompressed_shape(D);
  for (int i = 0; i < D; i++)
    decompressed_shape[i] = handle.shapes_h[0][i];
  std::reverse(decompressed_shape.begin(), decompressed_shape.end());
  Array<D, T> decompressed_data(decompressed_shape);

  // printf("sync_all 7.5\n");
  handle.sync_all();

  // printf("dqv\n");
  // print_matrix_cuda(1, quantized_count, dqv, quantized_count);

  bool prep_huffman = m.gpu_lossless;
  levelwise_linear_dequantize<D, T>(
      handle, handle.ranges_d, handle.l_target, handle.volumes,
      handle.ldvolumes, m, dqv, thrust::raw_pointer_cast(ldqvs.data()),
      decompressed_data.get_dv(), decompressed_data.get_ldvs_d(), prep_huffman,
      outlier_count, outlier_idx_d, outliers, 0);
  // printf("sync_all 8\n");
  handle.sync_all();
  cudaFreeHelper(dqv);
  if (prep_huffman) {
    cudaFreeHelper(outlier_idx_d);
    cudaFreeHelper(outliers);
  }
  if (handle.timing) {
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << log::log_time << "Dequantization time: " << time_span.count()
              << " s\n";
  }

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

  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  if (handle.timing)
    t1 = high_resolution_clock::now();
  recompose<D, T>(handle, decompressed_data.get_dv(),
                  decompressed_data.get_ldvs_h(),
                  decompressed_data.get_ldvs_d(), m.l_target, 0);

  {
    // int block_size = BLOCK_SIZE;
    // for (int i = 0; i < handle.dofs[0][0]; i += block_size) {
    //   for (int j = 0; j < handle.dofs[1][0]; j += block_size) {
    //     for (int k = 0; k < handle.dofs[2][0]; k += block_size) {
    //       size_t b0 = std::min(block_size, handle.dofs[0][0] - i);
    //       size_t b1 = std::min(block_size, handle.dofs[1][0] - j);
    //       size_t b2 = std::min(block_size, handle.dofs[2][0] - k);
    //       std::vector<size_t> block_shape = {b2, b1, b0};
    //       // mgard_cuda::Array<3, T> block_array(block_shape);
    //       mgard_cuda::Handle<3, T> block_handle(block_shape);
    //       std::vector<int> idx = {(int)i, (int)j, (int)k};
    //       // printf("recompose: %llu, %llu, %llu\n", i, j, k);
    //       // printf("block_array: %llu, %llu, %llu ld %d %d %d\n", b0, b1,
    //       b2, block_array.get_ldvs_h()[0],
    //       //         block_array.get_ldvs_h()[1],
    //       block_array.get_ldvs_h()[2]);

    //       //
    //       block_array.loadData(in_array.get_dv()+get_idx(in_array.get_ldvs_h(),
    //       idx), in_array.get_ldvs_h()[0]); block_handle.allocate_workspace();
    //       recompose<3, T>(block_handle,
    //       decompressed_data.get_dv()+get_idx(decompressed_data.get_ldvs_h(),
    //       idx), decompressed_data.get_ldvs_h(),
    //               block_handle.l_target);
    //       block_handle.free_workspace();
    //     }
    //   }
    // }
  }

  // printf("sync_all 9\n");
  handle.sync_all();
  if (handle.timing) {
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << log::log_time << "Recomposition time: " << time_span.count()
              << " s\n";
  }

  handle.free_workspace();

  // printf("sync_all 10\n");
  handle.sync_all();
  if (handle.timing) {
    end = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(end - start);
    std::cout << log::log_time
              << "Overall decompression time: " << time_span.count() << " s ("
              << (double)(handle.dofs[0][0] * handle.dofs[1][0] *
                          handle.linearized_depth * sizeof(T)) /
                     time_span.count() / 1e9
              << " GB/s)\n";
  }

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
