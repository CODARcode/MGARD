/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#include "cuda/CommonInternal.h"

#include "cuda/MemoryManagement.h"
#include "cuda/PrecomputeKernels.h"

#include "cuda/Handle.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mgard_cuda {

template <DIM D, typename T>
void Handle<D, T>::coord_to_dist(SIZE dof, T *coord, T *dist) {
  if (dof <= 1)
    return;
  // printf("coord_to_dist\n");
  T *h_coord = new T[dof];
  T *h_dist = new T[dof];
  for (int i = 0; i < dof; i++)
    h_dist[i] = 0.0;
  cudaMemcpyAsyncHelper(*this, h_coord, coord, dof * sizeof(T), AUTO, 0);
  this->sync(0);
  for (int i = 0; i < dof - 1; i++) {
    h_dist[i] = h_coord[i + 1] - h_coord[i];
  }
  if (dof != 2 && dof % 2 == 0) {
    T last_dist = h_dist[dof - 2];
    h_dist[dof - 2] = last_dist / 2.0;
    h_dist[dof - 1] = last_dist / 2.0;
  }
  cudaMemcpyAsyncHelper(*this, dist, h_dist, dof * sizeof(T), AUTO, 0);
  this->sync(0);
  delete[] h_coord;
  delete[] h_dist;
}

template <DIM D, typename T>
void Handle<D, T>::dist_to_ratio(SIZE dof, T *dist, T *ratio) {
  if (dof <= 1)
    return;
  // printf("dist_to_ratio %llu\n", dof);
  T *h_dist = new T[dof];
  T *h_ratio = new T[dof];
  for (int i = 0; i < dof; i++)
    h_ratio[i] = 0.0;
  cudaMemcpyAsyncHelper(*this, h_dist, dist, dof * sizeof(T), AUTO, 0);
  this->sync(0);
  for (int i = 0; i < dof - 2; i++) {
    h_ratio[i] = h_dist[i] / (h_dist[i + 1] + h_dist[i]);
    // printf("dof: %llu ratio: %f\n", dof, h_ratio[i]);
  }
  if (dof % 2 == 0) {
    h_ratio[dof - 2] = h_dist[dof - 2] / (h_dist[dof - 1] + h_dist[dof - 2]);
    // printf("dof: %llu ratio: %f\n", dof, h_ratio[dof - 2]);
  }
  cudaMemcpyAsyncHelper(*this, ratio, h_ratio, dof * sizeof(T), AUTO, 0);
  this->sync(0);
  delete[] h_dist;
  delete[] h_ratio;
}

template <DIM D, typename T>
void Handle<D, T>::reduce_dist(SIZE dof, T *dist, T *dist2) {
  if (dof <= 1)
    return;
  // printf("reduce_dist\n");
  SIZE dof2 = dof / 2 + 1;
  T *h_dist = new T[dof];
  T *h_dist2 = new T[dof2];
  for (int i = 0; i < dof2; i++)
    h_dist2[i] = 0.0;
  cudaMemcpyAsyncHelper(*this, h_dist, dist, dof * sizeof(T), AUTO, 0);
  this->sync(0);
  for (int i = 0; i < dof2 - 1; i++) {
    h_dist2[i] = h_dist[i * 2] + h_dist[i * 2 + 1];
  }
  if (dof2 != 2 && dof2 % 2 == 0) {
    T last_dist = h_dist2[dof2 - 2];
    h_dist2[dof2 - 2] = last_dist / 2.0;
    h_dist2[dof2 - 1] = last_dist / 2.0;
  }
  cudaMemcpyAsyncHelper(*this, dist2, h_dist2, dof2 * sizeof(T), AUTO, 0);
  this->sync(0);
  delete[] h_dist;
  delete[] h_dist2;
}

template <DIM D, typename T>
void Handle<D, T>::calc_am_bm(SIZE dof, T *dist, T *am, T *bm) {
  T *h_dist = new T[dof];
  T *h_am = new T[dof + 1];
  T *h_bm = new T[dof + 1];
  for (int i = 0; i < dof + 1; i++) {
    h_am[i] = 0.0;
    h_bm[i] = 0.0;
  }
  cudaMemcpyAsyncHelper(*this, h_dist, dist, dof * sizeof(T), AUTO, 0);
  this->sync(0);
  h_bm[0] = 2 * h_dist[0] / 6;
  h_am[0] = 0.0;

  for (int i = 1; i < dof - 1; i++) {
    T a_j = h_dist[i - 1] / 6;
    T w = a_j / h_bm[i - 1];
    h_bm[i] = 2 * (h_dist[i - 1] + h_dist[i]) / 6 - w * a_j;
    h_am[i] = a_j;
  }
  T a_j = h_dist[dof - 2] / 6;
  T w = a_j / h_bm[dof - 2];
  h_bm[dof - 1] = 2 * h_dist[dof - 2] / 6 - w * a_j;
  h_am[dof - 1] = a_j;
#ifdef MGARD_CUDA_FMA
  for (int i = 0; i < dof + 1; i++) {
    h_am[i] = -1 * h_am[i];
    h_bm[i] = 1 / h_bm[i];
  }
#endif
  cudaMemcpyAsyncHelper(*this, am, h_am, dof * sizeof(T), AUTO, 0);
  cudaMemcpyAsyncHelper(*this, bm + 1, h_bm, dof * sizeof(T), AUTO,
                        0); // add offset
  T one = 1;
  cudaMemcpyAsyncHelper(*this, bm, &one, sizeof(T), AUTO, 0); // add offset
  T zero = 0;
  cudaMemcpyAsyncHelper(*this, am + dof, &zero, sizeof(T), AUTO, 0);

  this->sync(0);
  delete[] h_dist;
  delete[] h_am;
  delete[] h_bm;
}

template <DIM D, typename T>
void Handle<D, T>::calc_volume(SIZE dof, T *dist, T *volume) {
  T *h_dist = new T[dof];
  T *h_volume = new T[dof];
  for (int i = 0; i < dof; i++) {
    h_volume[i] = 0.0;
  }
  cudaMemcpyAsyncHelper(*this, h_dist, dist, dof * sizeof(T), AUTO, 0);
  this->sync(0);
  if (dof == 2) {
    h_volume[0] = h_dist[0] / 2;
    h_volume[1] = h_dist[0] / 2;
  } else {
    int node_coeff_div = dof / 2 + 1;
    h_volume[0] = h_dist[0] / 2;
    for (int i = 1; i < dof - 1; i++) {
      if (i % 2 == 0) { // node
        h_volume[i / 2] = (h_dist[i - 1] + h_dist[i]) / 2;
      } else { // coeff
        h_volume[node_coeff_div + i / 2] = (h_dist[i - 1] + h_dist[i]) / 2;
      }
    }
    if (dof % 2 != 0) {
      h_volume[node_coeff_div - 1] = h_dist[dof - 2] / 2;
    } else {
      h_volume[node_coeff_div - 1] = h_dist[dof - 1] / 2;
    }
  }

  for (int i = 0; i < dof; i++) {
    h_volume[i] = 1.0 / h_volume[i];
  }
  cudaMemcpyAsyncHelper(*this, volume, h_volume, dof * sizeof(T), AUTO, 0);
  this->sync(0);
  delete[] h_dist;
  delete[] h_volume;
}

template <DIM D, typename T>
void Handle<D, T>::init(std::vector<SIZE> shape, std::vector<T *> coords,
                        Config config) {

  this->shape = shape;
  // determine dof
  for (DIM i = 0; i < shape.size(); i++) {
    std::vector<SIZE> curr_dofs;
    int n = shape[i];
    // printf("shape[%d] = %d\n", i, shape[i]);
    while (n > 2) {
      curr_dofs.push_back(n);
      n = n / 2 + 1;
    }
    if (shape[i] > 1)
      curr_dofs.push_back(2);
    dofs.push_back(curr_dofs);
    // printf("dofs[%d].size() = %d\n", i, dofs[i].size());
  }

  // printf("isGPUPointer: %d\n", isGPUPointer(shape.data()));

  linearized_depth = 1;
  for (int i = 2; i < shape.size(); i++) {
    linearized_depth *= shape[i];
  }

  // workspace (assume 3d and above)
  padded_linearized_depth = 1;
  for (int i = 2; i < D; i++) {
    padded_linearized_depth *= (shape[i] + 2);
  }

  for (SIZE i = 1; i < shape.size(); i++) {
    if (shape[i] == 1) {
      for (SIZE l = 0; l < dofs[0].size(); l++) {
        dofs[i].push_back(1);
      }
    }
  }

  // for (int d = 0; d < std::max(3, (int)shape.size()); d++ ) {
  //   printf("shape[%d]: %d dofs[%d]: ", d, shape[d], d);
  //   for (int l = 0 ; l < dofs[d].size(); l++) {
  //     printf("%d ", dofs[d][l]);
  //   }
  //   printf("\n");
  // }

  // determine l target
  SIZE nlevel = dofs[0].size();
  for (int i = 1; i < shape.size(); i++) {
    nlevel = std::min(nlevel, (SIZE)dofs[i].size());
  }
  l_target = nlevel - 1;
  if (config.l_target != -1) {
    l_target = std::min(nlevel - 1, config.l_target);
  }
  // l_target = nlevel;
  // printf("nlevel - 1 %d, l_target: %d\n", nlevel - 1, config.l_target);

  // shapes
  for (int l = 0; l < l_target + 1; l++) {
    SIZE *curr_shape_h = new SIZE[D_padded];
    for (int d = 0; d < D_padded; d++) {
      curr_shape_h[d] = dofs[d][l];
    }
    shapes_h.push_back(curr_shape_h);
    SIZE *curr_shape_d;
    cudaMallocHelper(*this, (void **)&(curr_shape_d), D_padded * sizeof(SIZE));
    cudaMemcpyAsyncHelper(*this, curr_shape_d, curr_shape_h,
                          D_padded * sizeof(SIZE), mgard_cuda::H2D, 0);
    shapes_d.push_back(curr_shape_d);
  }

  // ranges
  ranges_h = new SIZE[D * (l_target + 2)];
  for (int d = 0; d < D; d++) {
    ranges_h[d * (l_target + 2)] = 0;
    for (int l = 1; l < l_target + 2; l++) {
      ranges_h[d * (l_target + 2) + l] = dofs[d][l_target + 1 - l];
    }
    // printf("hshapes[%d]: ", d);
    // for (int l = 0; l < handle.l_target+2; l++) { printf("%d ", hshapes[d *
    // (handle.l_target+2)+l]); } printf("\n");
  }
  cudaMallocHelper(*this, (void **)&ranges_d,
                   D * (l_target + 2) * sizeof(SIZE));
  cudaMemcpyAsyncHelper(*this, ranges_d, ranges_h,
                        D * (l_target + 2) * sizeof(SIZE), H2D, 0);

  processed_n = new DIM[D];
  processed_dims_h = new DIM *[D];
  processed_dims_d = new DIM *[D];

  {
    thrust::device_vector<DIM> tmp(0);
    for (int d = 0; d < D; d++) {
      processed_n[d] = tmp.size();
      processed_dims_h[d] = new DIM[processed_n[d]];
      cudaMemcpyAsyncHelper(*this, processed_dims_h[d],
                            thrust::raw_pointer_cast(tmp.data()),
                            processed_n[d] * sizeof(DIM), mgard_cuda::D2H, 0);
      cudaMallocHelper(*this, (void **)&processed_dims_d[d],
                       processed_n[d] * sizeof(DIM));
      cudaMemcpyAsyncHelper(*this, processed_dims_d[d],
                            thrust::raw_pointer_cast(tmp.data()),
                            processed_n[d] * sizeof(DIM), mgard_cuda::D2D, 0);
      tmp.push_back(d);
    }
  }
  {
    thrust::device_vector<DIM> tmp(0);
    for (int i = 3; i < D; i++) {
      tmp.push_back(i);
    }
    unprocessed_n = new DIM[tmp.size()];
    unprocessed_dims_h = new DIM *[tmp.size()];
    unprocessed_dims_d = new DIM *[tmp.size()];

    //+1 is used for storing empty status
    for (int d = 0; d < (int)D - 3 + 1; d++) {
      unprocessed_n[d] = tmp.size();
      unprocessed_dims_h[d] = new DIM[unprocessed_n[d]];
      cudaMemcpyAsyncHelper(*this, unprocessed_dims_h[d],
                            thrust::raw_pointer_cast(tmp.data()),
                            unprocessed_n[d] * sizeof(DIM), mgard_cuda::D2H, 0);
      cudaMallocHelper(*this, (void **)&unprocessed_dims_d[d],
                       unprocessed_n[d] * sizeof(DIM));
      cudaMemcpyAsyncHelper(*this, unprocessed_dims_d[d],
                            thrust::raw_pointer_cast(tmp.data()),
                            unprocessed_n[d] * sizeof(DIM), mgard_cuda::D2D, 0);
      tmp.pop_back();
    }
  }

  cudaMallocHelper(*this, (void **)&(quantizers), (l_target + 1) * sizeof(T));

  // handle coords
  this->coords_h = coords;
  for (int i = 0; i < shape.size(); i++) {
    T *curr_dcoords;
    cudaMallocHelper(*this, (void **)&(curr_dcoords), shape[i] * sizeof(T));
    cudaMemcpyAsyncHelper(*this, curr_dcoords, this->coords_h[i],
                          shape[i] * sizeof(T), AUTO, 0);
    this->coords_d.push_back(curr_dcoords);
  }

  // calculate dist and ratio
  for (int i = 0; i < shape.size(); i++) {
    std::vector<T *> curr_ddist_l, curr_dratio_l;
    // for level 0
    int last_dist = dofs[i][0] - 1;
    T *curr_ddist0, *curr_dratio0;
    cudaMallocHelper(*this, (void **)&curr_ddist0, dofs[i][0] * sizeof(T));
    cudaMallocHelper(*this, (void **)&curr_dratio0, dofs[i][0] * sizeof(T));
    curr_ddist_l.push_back(curr_ddist0);
    curr_dratio_l.push_back(curr_dratio0);
    coord_to_dist(dofs[i][0], this->coords_d[i], curr_ddist_l[0]);
    dist_to_ratio(dofs[i][0], curr_ddist_l[0], curr_dratio_l[0]);

    // for l = 1 ... l_target
    for (int l = 1; l < l_target + 1; l++) {
      T *curr_ddist, *curr_dratio;
      cudaMallocHelper(*this, (void **)&curr_ddist, dofs[i][l] * sizeof(T));
      cudaMallocHelper(*this, (void **)&curr_dratio, dofs[i][l] * sizeof(T));
      curr_ddist_l.push_back(curr_ddist);
      curr_dratio_l.push_back(curr_dratio);
      reduce_dist(dofs[i][l - 1], curr_ddist_l[l - 1], curr_ddist_l[l]);
      dist_to_ratio(dofs[i][l], curr_ddist_l[l], curr_dratio_l[l]);
    }
    dist.push_back(curr_ddist_l);
    ratio.push_back(curr_dratio_l);
  }

  // for (int l = 0; l < l_target+1; l++) {
  //   printf("l: %d\n", l);
  //   for (int d = 0; d < D; d++) {
  //     printf("dist: ");
  //     print_matrix_cuda(1, dofs[d][l], dist[d][l], dofs[d][l]);
  //     printf("ratio: ");
  //     print_matrix_cuda(1, dofs[d][l], ratio[d][l], dofs[d][l]);
  //   }
  // }

  // volume for quantization
  SIZE volumes_width = 0;
  for (int d = 0; d < D; d++) {
    volumes_width = std::max(volumes_width, dofs[d][0]);
  }
  size_t volumes_pitch;
  cudaMallocPitchHelper(*this, (void **)&volumes, &volumes_pitch,
                        volumes_width * sizeof(T), D * (l_target + 1));
  ldvolumes = (SIZE)volumes_pitch / sizeof(T);
  for (int d = 0; d < D; d++) {
    for (int l = 0; l < l_target + 1; l++) {
      calc_volume(dofs[d][l], dist[d][l],
                  volumes + ldvolumes * (d * (l_target + 1) + (l_target - l)));
    }
  }

  // printf("volumes:\n");
  //  print_matrix_cuda(D * (l_target+1), volumes_width, volumes, ldvolumes);

  for (DIM i = 0; i < D; i++) {
    std::vector<T *> curr_am_l, curr_bm_l;
    for (SIZE l = 0; l < l_target + 1; l++) {
      T *curr_am, *curr_bm;
      cudaMallocHelper(*this, (void **)&curr_am, (dofs[i][l] + 1) * sizeof(T));
      cudaMallocHelper(*this, (void **)&curr_bm, (dofs[i][l] + 1) * sizeof(T));
      cudaMemsetHelper((void **)&curr_am, (dofs[i][l] + 1) * sizeof(T), 0);
      cudaMemsetHelper((void **)&curr_bm, (dofs[i][l] + 1) * sizeof(T), 0);
      curr_am_l.push_back(curr_am);
      curr_bm_l.push_back(curr_bm);
      calc_am_bm(dofs[i][l], dist[i][l], curr_am_l[l], curr_bm_l[l]);
      // printf("d: %d, l: %d\n", i, l);
      // printf("am: ");
      // print_matrix_cuda(1, dofs[i][l]+1, curr_am_l[l], dofs[i][l]+1);
      // printf("bm: ");
      // print_matrix_cuda(1, dofs[i][l]+1, curr_bm_l[l], dofs[i][l]+1);
    }
    am.push_back(curr_am_l);
    bm.push_back(curr_bm_l);
  }

  lossless = config.lossless;
  huff_dict_size = config.huff_dict_size;
  huff_block_size = config.huff_block_size;
  lz4_block_size = config.lz4_block_size;
  reduce_memory_footprint = config.reduce_memory_footprint;
  profile_kernels = config.profile_kernels;
  sync_and_check_all_kernels = config.sync_and_check_all_kernels;
  timing = config.timing;

  initialized = true;
}

template <DIM D, typename T> void Handle<D, T>::destroy() {

  for (int i = 0; i < shapes_d.size(); i++) {
    cudaFreeHelper(shapes_d[i]);
  }

  delete[] ranges_h;
  cudaFreeHelper(ranges_d);

  for (int d = 0; d < D; d++) {
    delete[] processed_dims_h[d];
    cudaFreeHelper(processed_dims_d[d]);
  }
  delete[] processed_n;
  delete[] processed_dims_h;
  delete[] processed_dims_d;

  for (int d = 0; d < (int)D - 3; d++) {
    // printf("d=%d D-3=%d\n",d, D-3);
    delete[] unprocessed_dims_h[d];
    cudaFreeHelper(unprocessed_dims_d[d]);
  }
  delete[] unprocessed_n;
  delete[] unprocessed_dims_h;
  delete[] unprocessed_dims_d;

  cudaFreeHelper(quantizers);

  for (int i = 0; i < D_padded; i++) {
    cudaFreeHelper(coords_d[i]);
  }

  for (int i = 0; i < dist.size(); i++) {
    for (int l = 0; l < dist[i].size(); l++) {
      cudaFreeHelper(dist[i][l]);
      cudaFreeHelper(ratio[i][l]);
    }
  }

  for (int i = 0; i < am.size(); i++) {
    for (int l = 0; l < am[i].size(); l++) {
      cudaFreeHelper(am[i][l]);
      cudaFreeHelper(bm[i][l]);
    }
  }

  if (uniform_coords_created) {
    for (int d = 0; d < D; d++) {
      // delete [] this->coords_h[d];
    }
    uniform_coords_created = false;
  }
}

template <DIM D, typename T>
void Handle<D, T>::padding_dimensions(std::vector<SIZE> &shape,
                                      std::vector<T *> &coords) {
  D_padded = D;
  if (D < 3) {
    D_padded = 3;
  }
  if (D % 2 == 0) {
    D_padded = D + 1;
  }
  // padding dimensions
  for (int d = shape.size(); d < D_padded; d++) {
    shape.push_back(1);
    T *curr_coords = new T[shape[d]];
    for (int i = 0; i < shape[d]; i++) {
      curr_coords[i] = (T)i;
    }
    coords.push_back(curr_coords);
  }
  // printf("D: %d, D_padded: %d\n", D, D_padded);
}

template <DIM D, typename T> void Handle<D, T>::create_queues() {
  num_of_queues = 16;
  cudaStream_t *ptr = new cudaStream_t[num_of_queues];
  for (int i = 0; i < num_of_queues; i++) {
    gpuErrchk(cudaStreamCreate(ptr + i));
  }
  queues = (void *)ptr;
}

template <DIM D, typename T> void Handle<D, T>::destroy_queues() {
  cudaStream_t *ptr = (cudaStream_t *)queues;
  for (int i = 0; i < num_of_queues; i++) {
    gpuErrchk(cudaStreamDestroy(ptr[i]));
  }
}

template <DIM D, typename T>
std::vector<T *> Handle<D, T>::create_uniform_coords(std::vector<SIZE> shape,
                                                     int mode) {

  std::vector<T *> coords(D);
  for (int d = 0; d < D; d++) {
    T *curr_coords = new T[shape[d]];
    for (int i = 0; i < shape[d]; i++) {
      // 0...n-1
      if (mode == 0) {
        // printf("create_uniform_coords %d\n", mode);
        curr_coords[i] = (T)i;
      } else if (mode == 1) {
        // 0...1
        curr_coords[i] = (T)i / (shape[d] - 1);
      } else {
        std::cout << log::log_err << "wrong uniform coordinates mode!\n";
        exit(-1);
      }
    }
    coords[d] = curr_coords;
  }
  uniform_coords_created = true;
  return coords;
}

template <DIM D, typename T> void Handle<D, T>::init_auto_tuning_table() {

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev_id);

  arch = 1; // default optimized for Volta

  if (prop.major == 7 && prop.minor == 0) {
    arch = 1;
    // printf("Optimized: Volta\n");
  }

  if (prop.major == 7 && (prop.minor == 2 || prop.minor == 5)) {
    arch = 2;
    // printf("Optimized: Turing\n");
  }
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  if (sizeof(T) == sizeof(float)) {
    precision = 0;
  } else if (sizeof(T) == sizeof(double)) {
    precision = 1;
  }

  this->auto_tuning_cc = new int **[num_arch];
  this->auto_tuning_mr1 = new int **[num_arch];
  this->auto_tuning_mr2 = new int **[num_arch];
  this->auto_tuning_mr3 = new int **[num_arch];
  this->auto_tuning_ts1 = new int **[num_arch];
  this->auto_tuning_ts2 = new int **[num_arch];
  this->auto_tuning_ts3 = new int **[num_arch];
  for (int i = 0; i < num_arch; i++) {
    this->auto_tuning_cc[i] = new int *[num_precision];
    this->auto_tuning_mr1[i] = new int *[num_precision];
    this->auto_tuning_mr2[i] = new int *[num_precision];
    this->auto_tuning_mr3[i] = new int *[num_precision];
    this->auto_tuning_ts1[i] = new int *[num_precision];
    this->auto_tuning_ts2[i] = new int *[num_precision];
    this->auto_tuning_ts3[i] = new int *[num_precision];
    for (int j = 0; j < num_precision; j++) {
      this->auto_tuning_cc[i][j] = new int[num_range];
      this->auto_tuning_mr1[i][j] = new int[num_range];
      this->auto_tuning_mr2[i][j] = new int[num_range];
      this->auto_tuning_mr3[i][j] = new int[num_range];
      this->auto_tuning_ts1[i][j] = new int[num_range];
      this->auto_tuning_ts2[i][j] = new int[num_range];
      this->auto_tuning_ts3[i][j] = new int[num_range];
    }
  }

  // Default
  for (int i = 0; i < num_arch; i++) {
    for (int j = 0; j < num_precision; j++) {
      for (int k = 0; k < num_range; k++) {
        this->auto_tuning_cc[i][j][k] = 0;
        this->auto_tuning_mr1[i][j][k] = 0;
        this->auto_tuning_mr2[i][j][k] = 0;
        this->auto_tuning_mr3[i][j][k] = 0;
        this->auto_tuning_ts1[i][j][k] = 0;
        this->auto_tuning_ts2[i][j][k] = 0;
        this->auto_tuning_ts3[i][j][k] = 0;
      }
    }
  }

  // Volta-Single
  this->auto_tuning_cc[1][0][0] = 1;
  this->auto_tuning_cc[1][0][1] = 1;
  this->auto_tuning_cc[1][0][2] = 1;
  this->auto_tuning_cc[1][0][3] = 1;
  this->auto_tuning_cc[1][0][4] = 1;
  this->auto_tuning_cc[1][0][5] = 5;
  this->auto_tuning_cc[1][0][6] = 5;
  this->auto_tuning_cc[1][0][7] = 5;
  this->auto_tuning_cc[1][0][8] = 5;

  this->auto_tuning_mr1[1][0][0] = 1;
  this->auto_tuning_mr2[1][0][0] = 1;
  this->auto_tuning_mr3[1][0][0] = 1;
  this->auto_tuning_mr1[1][0][1] = 1;
  this->auto_tuning_mr2[1][0][1] = 1;
  this->auto_tuning_mr3[1][0][1] = 1;
  this->auto_tuning_mr1[1][0][2] = 1;
  this->auto_tuning_mr2[1][0][2] = 1;
  this->auto_tuning_mr3[1][0][2] = 1;
  this->auto_tuning_mr1[1][0][3] = 3;
  this->auto_tuning_mr2[1][0][3] = 3;
  this->auto_tuning_mr3[1][0][3] = 3;
  this->auto_tuning_mr1[1][0][4] = 4;
  this->auto_tuning_mr2[1][0][4] = 1;
  this->auto_tuning_mr3[1][0][4] = 3;
  this->auto_tuning_mr1[1][0][5] = 5;
  this->auto_tuning_mr2[1][0][5] = 3;
  this->auto_tuning_mr3[1][0][5] = 3;
  this->auto_tuning_mr1[1][0][6] = 5;
  this->auto_tuning_mr2[1][0][6] = 4;
  this->auto_tuning_mr3[1][0][6] = 4;
  this->auto_tuning_mr1[1][0][7] = 3;
  this->auto_tuning_mr2[1][0][7] = 4;
  this->auto_tuning_mr3[1][0][7] = 4;
  this->auto_tuning_mr1[1][0][8] = 3;
  this->auto_tuning_mr2[1][0][8] = 4;
  this->auto_tuning_mr3[1][0][8] = 4;

  this->auto_tuning_ts1[1][0][0] = 1;
  this->auto_tuning_ts2[1][0][0] = 1;
  this->auto_tuning_ts3[1][0][0] = 1;
  this->auto_tuning_ts1[1][0][1] = 1;
  this->auto_tuning_ts2[1][0][1] = 1;
  this->auto_tuning_ts3[1][0][1] = 1;
  this->auto_tuning_ts1[1][0][2] = 2;
  this->auto_tuning_ts2[1][0][2] = 2;
  this->auto_tuning_ts3[1][0][2] = 2;
  this->auto_tuning_ts1[1][0][3] = 3;
  this->auto_tuning_ts2[1][0][3] = 2;
  this->auto_tuning_ts3[1][0][3] = 2;
  this->auto_tuning_ts1[1][0][4] = 3;
  this->auto_tuning_ts2[1][0][4] = 2;
  this->auto_tuning_ts3[1][0][4] = 2;
  this->auto_tuning_ts1[1][0][5] = 3;
  this->auto_tuning_ts2[1][0][5] = 2;
  this->auto_tuning_ts3[1][0][5] = 2;
  this->auto_tuning_ts1[1][0][6] = 5;
  this->auto_tuning_ts2[1][0][6] = 3;
  this->auto_tuning_ts3[1][0][6] = 2;
  this->auto_tuning_ts1[1][0][7] = 5;
  this->auto_tuning_ts2[1][0][7] = 6;
  this->auto_tuning_ts3[1][0][7] = 5;
  this->auto_tuning_ts1[1][0][8] = 5;
  this->auto_tuning_ts2[1][0][8] = 6;
  this->auto_tuning_ts3[1][0][8] = 5;
  // Volta-Double

  this->auto_tuning_cc[1][1][0] = 1;
  this->auto_tuning_cc[1][1][1] = 1;
  this->auto_tuning_cc[1][1][2] = 1;
  this->auto_tuning_cc[1][1][3] = 1;
  this->auto_tuning_cc[1][1][4] = 4;
  this->auto_tuning_cc[1][1][5] = 5;
  this->auto_tuning_cc[1][1][6] = 6;
  this->auto_tuning_cc[1][1][7] = 6;
  this->auto_tuning_cc[1][1][8] = 5;

  this->auto_tuning_mr1[1][1][0] = 1;
  this->auto_tuning_mr2[1][1][0] = 1;
  this->auto_tuning_mr3[1][1][0] = 1;
  this->auto_tuning_mr1[1][1][1] = 1;
  this->auto_tuning_mr2[1][1][1] = 1;
  this->auto_tuning_mr3[1][1][1] = 1;
  this->auto_tuning_mr1[1][1][2] = 1;
  this->auto_tuning_mr2[1][1][2] = 1;
  this->auto_tuning_mr3[1][1][2] = 1;
  this->auto_tuning_mr1[1][1][3] = 1;
  this->auto_tuning_mr2[1][1][3] = 3;
  this->auto_tuning_mr3[1][1][3] = 1;
  this->auto_tuning_mr1[1][1][4] = 4;
  this->auto_tuning_mr2[1][1][4] = 3;
  this->auto_tuning_mr3[1][1][4] = 3;
  this->auto_tuning_mr1[1][1][5] = 5;
  this->auto_tuning_mr2[1][1][5] = 5;
  this->auto_tuning_mr3[1][1][5] = 5;
  this->auto_tuning_mr1[1][1][6] = 4;
  this->auto_tuning_mr2[1][1][6] = 6;
  this->auto_tuning_mr3[1][1][6] = 6;
  this->auto_tuning_mr1[1][1][7] = 6;
  this->auto_tuning_mr2[1][1][7] = 6;
  this->auto_tuning_mr3[1][1][7] = 5;
  this->auto_tuning_mr1[1][1][8] = 6;
  this->auto_tuning_mr2[1][1][8] = 6;
  this->auto_tuning_mr3[1][1][8] = 5;

  this->auto_tuning_ts1[1][1][0] = 1;
  this->auto_tuning_ts2[1][1][0] = 1;
  this->auto_tuning_ts3[1][1][0] = 1;
  this->auto_tuning_ts1[1][1][1] = 1;
  this->auto_tuning_ts2[1][1][1] = 1;
  this->auto_tuning_ts3[1][1][1] = 1;
  this->auto_tuning_ts1[1][1][2] = 2;
  this->auto_tuning_ts2[1][1][2] = 2;
  this->auto_tuning_ts3[1][1][2] = 2;
  this->auto_tuning_ts1[1][1][3] = 3;
  this->auto_tuning_ts2[1][1][3] = 2;
  this->auto_tuning_ts3[1][1][3] = 2;
  this->auto_tuning_ts1[1][1][4] = 3;
  this->auto_tuning_ts2[1][1][4] = 2;
  this->auto_tuning_ts3[1][1][4] = 2;
  this->auto_tuning_ts1[1][1][5] = 4;
  this->auto_tuning_ts2[1][1][5] = 2;
  this->auto_tuning_ts3[1][1][5] = 2;
  this->auto_tuning_ts1[1][1][6] = 5;
  this->auto_tuning_ts2[1][1][6] = 5;
  this->auto_tuning_ts3[1][1][6] = 2;
  this->auto_tuning_ts1[1][1][7] = 5;
  this->auto_tuning_ts2[1][1][7] = 6;
  this->auto_tuning_ts3[1][1][7] = 6;
  this->auto_tuning_ts1[1][1][8] = 5;
  this->auto_tuning_ts2[1][1][8] = 6;
  this->auto_tuning_ts3[1][1][8] = 6;

  // Turing-Single
  this->auto_tuning_cc[2][0][0] = 1;
  this->auto_tuning_cc[2][0][1] = 1;
  this->auto_tuning_cc[2][0][2] = 1;
  this->auto_tuning_cc[2][0][3] = 1;
  this->auto_tuning_cc[2][0][4] = 3;
  this->auto_tuning_cc[2][0][5] = 5;
  this->auto_tuning_cc[2][0][6] = 5;
  this->auto_tuning_cc[2][0][7] = 5;
  this->auto_tuning_cc[2][0][8] = 4;

  this->auto_tuning_mr1[2][0][0] = 1;
  this->auto_tuning_mr2[2][0][0] = 1;
  this->auto_tuning_mr3[2][0][0] = 1;
  this->auto_tuning_mr1[2][0][1] = 1;
  this->auto_tuning_mr2[2][0][1] = 1;
  this->auto_tuning_mr3[2][0][1] = 1;
  this->auto_tuning_mr1[2][0][2] = 1;
  this->auto_tuning_mr2[2][0][2] = 1;
  this->auto_tuning_mr3[2][0][2] = 1;
  this->auto_tuning_mr1[2][0][3] = 1;
  this->auto_tuning_mr2[2][0][3] = 1;
  this->auto_tuning_mr3[2][0][3] = 3;
  this->auto_tuning_mr1[2][0][4] = 4;
  this->auto_tuning_mr2[2][0][4] = 3;
  this->auto_tuning_mr3[2][0][4] = 4;
  this->auto_tuning_mr1[2][0][5] = 4;
  this->auto_tuning_mr2[2][0][5] = 3;
  this->auto_tuning_mr3[2][0][5] = 3;
  this->auto_tuning_mr1[2][0][6] = 6;
  this->auto_tuning_mr2[2][0][6] = 3;
  this->auto_tuning_mr3[2][0][6] = 3;
  this->auto_tuning_mr1[2][0][7] = 5;
  this->auto_tuning_mr2[2][0][7] = 4;
  this->auto_tuning_mr3[2][0][7] = 4;
  this->auto_tuning_mr1[2][0][8] = 5;
  this->auto_tuning_mr2[2][0][8] = 4;
  this->auto_tuning_mr3[2][0][8] = 4;

  this->auto_tuning_ts1[2][0][0] = 1;
  this->auto_tuning_ts2[2][0][0] = 1;
  this->auto_tuning_ts3[2][0][0] = 1;
  this->auto_tuning_ts1[2][0][1] = 1;
  this->auto_tuning_ts2[2][0][1] = 1;
  this->auto_tuning_ts3[2][0][1] = 1;
  this->auto_tuning_ts1[2][0][2] = 2;
  this->auto_tuning_ts2[2][0][2] = 2;
  this->auto_tuning_ts3[2][0][2] = 2;
  this->auto_tuning_ts1[2][0][3] = 3;
  this->auto_tuning_ts2[2][0][3] = 2;
  this->auto_tuning_ts3[2][0][3] = 2;
  this->auto_tuning_ts1[2][0][4] = 3;
  this->auto_tuning_ts2[2][0][4] = 2;
  this->auto_tuning_ts3[2][0][4] = 2;
  this->auto_tuning_ts1[2][0][5] = 3;
  this->auto_tuning_ts2[2][0][5] = 2;
  this->auto_tuning_ts3[2][0][5] = 2;
  this->auto_tuning_ts1[2][0][6] = 5;
  this->auto_tuning_ts2[2][0][6] = 5;
  this->auto_tuning_ts3[2][0][6] = 2;
  this->auto_tuning_ts1[2][0][7] = 5;
  this->auto_tuning_ts2[2][0][7] = 6;
  this->auto_tuning_ts3[2][0][7] = 6;
  this->auto_tuning_ts1[2][0][8] = 5;
  this->auto_tuning_ts2[2][0][8] = 6;
  this->auto_tuning_ts3[2][0][8] = 6;
  // Turing-Double

  this->auto_tuning_cc[2][1][0] = 0;
  this->auto_tuning_cc[2][1][1] = 0;
  this->auto_tuning_cc[2][1][2] = 2;
  this->auto_tuning_cc[2][1][3] = 2;
  this->auto_tuning_cc[2][1][4] = 3;
  this->auto_tuning_cc[2][1][5] = 4;
  this->auto_tuning_cc[2][1][6] = 4;
  this->auto_tuning_cc[2][1][7] = 6;
  this->auto_tuning_cc[2][1][8] = 3;

  this->auto_tuning_mr1[2][1][0] = 1;
  this->auto_tuning_mr2[2][1][0] = 1;
  this->auto_tuning_mr3[2][1][0] = 1;
  this->auto_tuning_mr1[2][1][1] = 1;
  this->auto_tuning_mr2[2][1][1] = 1;
  this->auto_tuning_mr3[2][1][1] = 1;
  this->auto_tuning_mr1[2][1][2] = 1;
  this->auto_tuning_mr2[2][1][2] = 1;
  this->auto_tuning_mr3[2][1][2] = 1;
  this->auto_tuning_mr1[2][1][3] = 1;
  this->auto_tuning_mr2[2][1][3] = 1;
  this->auto_tuning_mr3[2][1][3] = 1;
  this->auto_tuning_mr1[2][1][4] = 4;
  this->auto_tuning_mr2[2][1][4] = 4;
  this->auto_tuning_mr3[2][1][4] = 1;
  this->auto_tuning_mr1[2][1][5] = 1;
  this->auto_tuning_mr2[2][1][5] = 1;
  this->auto_tuning_mr3[2][1][5] = 1;
  this->auto_tuning_mr1[2][1][6] = 1;
  this->auto_tuning_mr2[2][1][6] = 1;
  this->auto_tuning_mr3[2][1][6] = 1;
  this->auto_tuning_mr1[2][1][7] = 1;
  this->auto_tuning_mr2[2][1][7] = 1;
  this->auto_tuning_mr3[2][1][7] = 1;
  this->auto_tuning_mr1[2][1][8] = 1;
  this->auto_tuning_mr2[2][1][8] = 1;
  this->auto_tuning_mr3[2][1][8] = 1;

  this->auto_tuning_ts1[2][1][0] = 1;
  this->auto_tuning_ts2[2][1][0] = 1;
  this->auto_tuning_ts3[2][1][0] = 1;
  this->auto_tuning_ts1[2][1][1] = 1;
  this->auto_tuning_ts2[2][1][1] = 1;
  this->auto_tuning_ts3[2][1][1] = 1;
  this->auto_tuning_ts1[2][1][2] = 2;
  this->auto_tuning_ts2[2][1][2] = 2;
  this->auto_tuning_ts3[2][1][2] = 2;
  this->auto_tuning_ts1[2][1][3] = 3;
  this->auto_tuning_ts2[2][1][3] = 2;
  this->auto_tuning_ts3[2][1][3] = 2;
  this->auto_tuning_ts1[2][1][4] = 2;
  this->auto_tuning_ts2[2][1][4] = 2;
  this->auto_tuning_ts3[2][1][4] = 2;
  this->auto_tuning_ts1[2][1][5] = 2;
  this->auto_tuning_ts2[2][1][5] = 2;
  this->auto_tuning_ts3[2][1][5] = 2;
  this->auto_tuning_ts1[2][1][6] = 3;
  this->auto_tuning_ts2[2][1][6] = 5;
  this->auto_tuning_ts3[2][1][6] = 3;
  this->auto_tuning_ts1[2][1][7] = 3;
  this->auto_tuning_ts2[2][1][7] = 6;
  this->auto_tuning_ts3[2][1][7] = 6;
  this->auto_tuning_ts1[2][1][8] = 3;
  this->auto_tuning_ts2[2][1][8] = 6;
  this->auto_tuning_ts3[2][1][8] = 6;
  auto_tuning_table_created = true;
}

template <DIM D, typename T> void Handle<D, T>::destroy_auto_tuning_table() {
  for (int i = 0; i < num_arch; i++) {
    for (int j = 0; j < num_precision; j++) {
      delete[] this->auto_tuning_cc[i][j];
      delete[] this->auto_tuning_mr1[i][j];
      delete[] this->auto_tuning_mr2[i][j];
      delete[] this->auto_tuning_mr3[i][j];
      delete[] this->auto_tuning_ts1[i][j];
      delete[] this->auto_tuning_ts2[i][j];
      delete[] this->auto_tuning_ts3[i][j];
    }
    delete[] this->auto_tuning_cc[i];
    delete[] this->auto_tuning_mr1[i];
    delete[] this->auto_tuning_mr2[i];
    delete[] this->auto_tuning_mr3[i];
    delete[] this->auto_tuning_ts1[i];
    delete[] this->auto_tuning_ts2[i];
    delete[] this->auto_tuning_ts3[i];
  }
  delete[] this->auto_tuning_cc;
  delete[] this->auto_tuning_mr1;
  delete[] this->auto_tuning_mr2;
  delete[] this->auto_tuning_mr3;
  delete[] this->auto_tuning_ts1;
  delete[] this->auto_tuning_ts2;
  delete[] this->auto_tuning_ts3;
}

template <DIM D, typename T> void Handle<D, T>::allocate_workspace() {

  // size_t free, total;
  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  // printf("allocate_workspace: %llu\n", (shapes_h[0][0] + 2) * sizeof(T) *
  // (shapes_h[0][1] + 2) * padded_linearized_depth);
  size_t dw_pitch;
  mgard_cuda::cudaMalloc3DHelper(*this, (void **)&(dw), &dw_pitch,
                                 (shapes_h[0][0] + 2) * sizeof(T),
                                 shapes_h[0][1] + 2, padded_linearized_depth);
  // printf("pitch %llu\n", dw_pitch);
  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  // ldws.push_back(dw_pitch / sizeof(T));
  // for (int i = 1; i < D_padded; i++) {
  //   ldws.push_back(shapes_h[0][i] + 2);
  // }
  lddw1 = (SIZE)dw_pitch / sizeof(T);
  lddw2 = shapes_h[0][1] + 2;

  // ldws_h = new int[D_padded];
  // ldws_h[0] = dw_pitch / sizeof(T);
  // for (int i = 1; i < D_padded; i++) {
  //   ldws_h[i] = shapes_h[0][i] + 2;
  // }

  ldws_h.push_back(dw_pitch / sizeof(T));
  for (int i = 1; i < D_padded; i++) {
    ldws_h.push_back(shapes_h[0][i] + 2);
  }

  mgard_cuda::cudaMallocHelper(*this, (void **)&ldws_d,
                               D_padded * sizeof(SIZE));
  mgard_cuda::cudaMemcpyAsyncHelper(*this, ldws_d, ldws_h.data(),
                                    D_padded * sizeof(SIZE), mgard_cuda::H2D,
                                    0);

  if (D > 3) {
    // printf("allocate_workspace: %llu\n", (shapes_h[0][0] + 2) * sizeof(T) *
    // (shapes_h[0][1] + 2) * padded_linearized_depth);
    size_t db_pitch;
    mgard_cuda::cudaMalloc3DHelper(*this, (void **)&(db), &db_pitch,
                                   (shapes_h[0][0] + 2) * sizeof(T),
                                   shapes_h[0][1] + 2, padded_linearized_depth);

    // ldbs.push_back(db_pitch / sizeof(T));
    // for (int i = 1; i < D_padded; i++) {
    //   ldbs.push_back(shapes_h[0][i] + 2);
    // }
    lddb1 = (SIZE)db_pitch / sizeof(T);
    lddb2 = shapes_h[0][1] + 2;

    // ldbs_h = new int[D_padded];
    // ldbs_h[0] = db_pitch / sizeof(T);
    // for (int i = 1; i < D_padded; i++) {
    //   ldbs_h[i] = shapes_h[0][i] + 2;
    // }

    ldbs_h.push_back(db_pitch / sizeof(T));
    for (int i = 1; i < D_padded; i++) {
      ldbs_h.push_back(shapes_h[0][i] + 2);
    }

    mgard_cuda::cudaMallocHelper(*this, (void **)&ldbs_d,
                                 D_padded * sizeof(SIZE));
    mgard_cuda::cudaMemcpyAsyncHelper(*this, ldbs_d, ldbs_h.data(),
                                      D_padded * sizeof(SIZE), mgard_cuda::H2D,
                                      0);
  }
  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);
}

template <DIM D, typename T> void Handle<D, T>::free_workspace() {
  mgard_cuda::cudaFreeHelper(dw);
  mgard_cuda::cudaFreeHelper(ldws_d);
  if (D > 3) {
    mgard_cuda::cudaFreeHelper(db);
    // delete [] ldbs_h;
    mgard_cuda::cudaFreeHelper(ldbs_d);
  }
}

// This constructor is for internal use only
template <DIM D, typename T> Handle<D, T>::Handle() {
  cudaSetDeviceHelper(dev_id);
  create_queues();
}

template <DIM D, typename T> Handle<D, T>::Handle(std::vector<SIZE> shape) {
  Config config;
  dev_id = config.dev_id;
  cudaSetDeviceHelper(dev_id);
  std::reverse(shape.begin(), shape.end());
  int ret = check_shape<D>(shape);
  if (ret == -1) {
    std::cerr << log::log_err
              << "Number of dimensions mismatch. mgard_cuda::Hanlde not "
                 "initialized!\n";
    return;
  }
  if (ret == -2) {
    std::cerr << log::log_err
              << "Size of any dimensions cannot be smaller than 3. "
                 "mgard_cuda::Hanlde not "
                 "initialized!\n";
  }
  dstype = data_structure_type::Cartesian_Grid_Uniform;
  std::vector<T *> coords = create_uniform_coords(shape, 0);
  padding_dimensions(shape, coords);
  create_queues();
  init_auto_tuning_table();
  init(shape, coords, config);
}

template <DIM D, typename T>
Handle<D, T>::Handle(std::vector<SIZE> shape, std::vector<T *> coords) {
  Config config;
  dev_id = config.dev_id;
  cudaSetDeviceHelper(dev_id);
  std::reverse(shape.begin(), shape.end());
  std::reverse(coords.begin(), coords.end());
  int ret = check_shape<D>(shape);
  if (ret == -1) {
    std::cerr << log::log_err
              << "Number of dimensions mismatch. mgard_cuda::Hanlde not "
                 "initialized!\n";
    return;
  }
  if (ret == -2) {
    std::cerr << log::log_err
              << "Size of any dimensions cannot be smaller than 3. "
                 "mgard_cuda::Hanlde not "
                 "initialized!\n";
  }

  dstype = data_structure_type::Cartesian_Grid_Non_Uniform;
  padding_dimensions(shape, coords);
  create_queues();
  init_auto_tuning_table();
  init(shape, coords, config);
}

template <DIM D, typename T>
Handle<D, T>::Handle(std::vector<SIZE> shape, Config config) {
  dev_id = config.dev_id;

  cudaSetDeviceHelper(dev_id);

  std::reverse(shape.begin(), shape.end());
  std::vector<T *> coords =
      create_uniform_coords(shape, config.uniform_coord_mode);
  int ret = check_shape<D>(shape);
  if (ret == -1) {
    std::cerr << log::log_err
              << "Number of dimensions mismatch. mgard_cuda::Hanlde not "
                 "initialized!\n";
    return;
  }
  if (ret == -2) {
    std::cerr << log::log_err
              << "Size of any dimensions cannot be smaller than 3. "
                 "mgard_cuda::Hanlde not "
                 "initialized!\n";
  }

  dstype = data_structure_type::Cartesian_Grid_Uniform;
  padding_dimensions(shape, coords);
  create_queues();
  init_auto_tuning_table();
  init(shape, coords, config);
}

template <DIM D, typename T>
Handle<D, T>::Handle(std::vector<SIZE> shape, std::vector<T *> coords,
                     Config config) {
  dev_id = config.dev_id;
  cudaSetDeviceHelper(dev_id);
  std::reverse(shape.begin(), shape.end());
  std::reverse(coords.begin(), coords.end());
  int ret = check_shape<D>(shape);
  if (ret == -1) {
    std::cerr << log::log_err
              << "Number of dimensions mismatch. mgard_cuda::Hanlde not "
                 "initialized!\n";
    return;
  }
  if (ret == -2) {
    std::cerr << log::log_err
              << "Size of any dimensions cannot be smaller than 3. "
                 "mgard_cuda::Hanlde not "
                 "initialized!\n";
  }

  dstype = data_structure_type::Cartesian_Grid_Non_Uniform;
  padding_dimensions(shape, coords);
  create_queues();
  init_auto_tuning_table();
  init(shape, coords, config);
}

template <DIM D, typename T> void *Handle<D, T>::get(int i) {
  cudaSetDeviceHelper(dev_id);
  cudaStream_t *ptr = (cudaStream_t *)(this->queues);
  return (void *)(ptr + i);
}

template <DIM D, typename T> void Handle<D, T>::sync(int i) {
  cudaSetDeviceHelper(dev_id);
  cudaStream_t *ptr = (cudaStream_t *)(this->queues);
  gpuErrchk(cudaStreamSynchronize(ptr[i]));
}

template <DIM D, typename T> void Handle<D, T>::sync_all() {
  cudaSetDeviceHelper(dev_id);
  cudaStream_t *ptr = (cudaStream_t *)(this->queues);
  for (int i = 0; i < this->num_of_queues; i++) {
    gpuErrchk(cudaStreamSynchronize(ptr[i]));
  }
}

template <DIM D, typename T> Handle<D, T>::~Handle() {
  cudaSetDeviceHelper(dev_id);
  if (initialized) {
    destroy();
  }
  if (auto_tuning_table_created) {
    destroy_auto_tuning_table();
  }
  destroy_queues();
}

template class Handle<1, double>;
template class Handle<1, float>;
template class Handle<2, double>;
template class Handle<2, float>;
template class Handle<3, double>;
template class Handle<3, float>;
template class Handle<4, double>;
template class Handle<4, float>;
template class Handle<5, double>;
template class Handle<5, float>;

} // namespace mgard_cuda