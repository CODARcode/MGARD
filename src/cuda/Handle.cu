/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/CommonInternal.h"
#include "cuda/Handle.h"
#include "cuda/MemoryManagement.h"
#include "cuda/PrecomputeKernels.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mgard_cuda {

template <typename T, uint32_t D>
void Handle<T, D>::init(std::vector<size_t> shape, std::vector<T *> coords,
                        mgard_cuda_config config) {

  // determine dof
  for (int i = 0; i < shape.size(); i++) {
    std::vector<int> curr_dofs;
    int n = shape[i];
    // printf("shape[%d] = %d\n", i, shape[i]);
    while (n > 2) {
      curr_dofs.push_back(n);
      n = n / 2 + 1;
    }
    dofs.push_back(curr_dofs);
    // printf("dofs[%d].size() = %d\n", i, dofs[i].size());
  }

  linearized_depth = 1;
  for (int i = 2; i < shape.size(); i++) {
    linearized_depth *= shape[i];
  }

  // workspace (assume 3d and above)
  padded_linearized_depth = 1;
  for (int i = 2; i < D; i++) {
    padded_linearized_depth *= (shape[i] + 2);
  }

  for (int i = 1; i < shape.size(); i++) {
    if (shape[i] == 1) {
      for (int l = 0; l < dofs[0].size(); l++) {
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
  int nlevel = dofs[0].size();
  for (int i = 1; i < shape.size(); i++) {
    nlevel = std::min(nlevel, (int)dofs[i].size());
  }
  l_target = nlevel - 1;
  if (config.l_target != -1) {
    l_target = std::min(nlevel - 1, config.l_target);
  }
  // l_target = nlevel;
  // printf("nlevel - 1 %d, l_target: %d\n", nlevel - 1, config.l_target);

  for (int l = 0; l < l_target + 1; l++) {
    int *curr_shape_h = new int[D_padded];
    for (int d = 0; d < D_padded; d++) {
      curr_shape_h[d] = dofs[d][l];
    }
    shapes_h.push_back(curr_shape_h);
    int *curr_shape_d;
    cudaMallocHelper((void **)&(curr_shape_d), D_padded * sizeof(int));
    cudaMemcpyAsyncHelper(*this, curr_shape_d, curr_shape_h,
                          D_padded * sizeof(int), mgard_cuda::H2D, 0);
    shapes_d.push_back(curr_shape_d);
  }

  processed_n = new int[D];
  processed_dims_h = new int *[D];
  processed_dims_d = new int *[D];

  thrust::device_vector<int> tmp(0);
  for (int d = 0; d < D; d++) {
    processed_n[d] = tmp.size();
    processed_dims_h[d] = new int[processed_n[d]];
    cudaMemcpyAsyncHelper(*this, processed_dims_h[d],
                          thrust::raw_pointer_cast(tmp.data()),
                          processed_n[d] * sizeof(int), mgard_cuda::D2H, 0);
    cudaMallocHelper((void **)&processed_dims_d[d],
                     processed_n[d] * sizeof(int));
    cudaMemcpyAsyncHelper(*this, processed_dims_d[d],
                          thrust::raw_pointer_cast(tmp.data()),
                          processed_n[d] * sizeof(int), mgard_cuda::D2D, 0);
    tmp.push_back(d);
  }

  cudaMallocHelper((void **)&(quantizers), (l_target + 1) * sizeof(T));

  // handle coords
  for (int i = 0; i < shape.size(); i++) {
    T *curr_dcoords;
    cudaMallocHelper((void **)&(curr_dcoords), shape[i] * sizeof(T));
    cudaMemcpyAsyncHelper(*this, curr_dcoords, coords[i], shape[i] * sizeof(T),
                          AUTO, 0);
    this->coords.push_back(curr_dcoords);
    // delete temporaly create uniform coords on host
    if (uniform_coords_created)
      delete[] coords[i];
  }

  // // calculate dist and ratio
  for (int i = 0; i < shape.size(); i++) {
    std::vector<T *> curr_ddist_l, curr_dratio_l;
    // for level 0
    int last_dist = dofs[i][0] - 1;
    T *curr_ddist0, *curr_dratio0;
    cudaMallocHelper((void **)&curr_ddist0, dofs[i][0] * sizeof(T));
    cudaMallocHelper((void **)&curr_dratio0, dofs[i][0] * sizeof(T));
    cudaMemsetHelper((void **)&curr_ddist0, dofs[i][0] * sizeof(T), 0);
    cudaMemsetHelper((void **)&curr_dratio0, dofs[i][0] * sizeof(T), 0);
    curr_ddist_l.push_back(curr_ddist0);
    curr_dratio_l.push_back(curr_dratio0);

    calc_cpt_dist(*this, dofs[i][0], this->coords[i], curr_ddist_l[0], 0);
    // mgard_cuda::calc_cpt_dist_ratio(*this, dofs[i][0], coords[i],
    // curr_dratio_l[0], 0);
    dist_to_ratio(*this, last_dist, curr_ddist_l[0], curr_dratio_l[0], 0);
    if (dofs[i][0] % 2 == 0) {
      // padding
      cudaMemcpyAsyncHelper(*this, curr_ddist_l[0] + dofs[i][0] - 1,
                            curr_ddist_l[0] + dofs[i][0] - 2, sizeof(T),
                            mgard_cuda::D2D, 0);
      last_dist += 1;

      T extra_ratio = 0.5;
      cudaMemcpyAsyncHelper(*this, curr_dratio_l[0] + dofs[i][0] - 2,
                            &extra_ratio, sizeof(T), mgard_cuda::H2D, 0);
    }

    // for l = 1 ... l_target
    for (int l = 1; l < l_target + 1; l++) {
      T *curr_ddist, *curr_dratio;
      cudaMallocHelper((void **)&curr_ddist, dofs[i][l] * sizeof(T));
      cudaMallocHelper((void **)&curr_dratio, dofs[i][l] * sizeof(T));
      cudaMemsetHelper((void **)&curr_ddist, dofs[i][0] * sizeof(T), 0);
      cudaMemsetHelper((void **)&curr_dratio, dofs[i][0] * sizeof(T), 0);
      curr_ddist_l.push_back(curr_ddist);
      curr_dratio_l.push_back(curr_dratio);
      reduce_two_dist(*this, last_dist, curr_ddist_l[l - 1], curr_ddist_l[l],
                      0);
      last_dist /= 2;
      dist_to_ratio(*this, last_dist, curr_ddist_l[l], curr_dratio_l[l], 0);

      if (last_dist % 2 != 0) {
        cudaMemcpyAsyncHelper(*this, curr_ddist_l[l] + last_dist,
                              curr_ddist_l[l] + last_dist - 1, sizeof(T),
                              mgard_cuda::D2D, 0);
        last_dist += 1;

        T extra_ratio = 0.5;
        cudaMemcpyAsyncHelper(*this, curr_dratio_l[l] + last_dist - 2,
                              &extra_ratio, sizeof(T), mgard_cuda::H2D, 0);
      }
    }
    dist.push_back(curr_ddist_l);
    ratio.push_back(curr_dratio_l);
  }

  for (int i = 0; i < shape.size(); i++) {
    std::vector<T *> curr_am_l, curr_bm_l;
    for (int l = 0; l < l_target + 1; l++) {
      T *curr_am, *curr_bm;
      cudaMallocHelper((void **)&curr_am, dofs[i][l] * sizeof(T));
      cudaMallocHelper((void **)&curr_bm, dofs[i][l] * sizeof(T));
      curr_am_l.push_back(curr_am);
      curr_bm_l.push_back(curr_bm);
      calc_am_bm(*this, dofs[i][l], dist[i][l], curr_am_l[l], curr_bm_l[l], 0);
    }
    am.push_back(curr_am_l);
    bm.push_back(curr_bm_l);
  }

  huff_dict_size = config.huff_dict_size;
  huff_block_size = config.huff_block_size;
  enable_lz4 = config.enable_lz4;
  lz4_block_size = config.lz4_block_size;

  initialized = true;
}

template <typename T, uint32_t D> void Handle<T, D>::destroy() {
  for (int i = 0; i < shapes_d.size(); i++) {
    cudaFreeHelper(shapes_d[i]);
  }

  for (int d = 0; d < D; d++) {
    delete[] processed_dims_h[d];
    cudaFreeHelper(processed_dims_d[d]);
  }
  delete[] processed_n;
  delete[] processed_dims_h;
  delete[] processed_dims_d;

  cudaFreeHelper(quantizers);

  for (int i = 0; i < D_padded; i++) {
    cudaFreeHelper(coords[i]);
  }

  for (int i = 0; i < D_padded; i++) {
    for (int l = 1; l < l_target + 1; l++) {
      cudaFreeHelper(dist[i][l]);
      cudaFreeHelper(ratio[i][l]);
      cudaFreeHelper(am[i][l]);
      cudaFreeHelper(bm[i][l]);
    }
  }
}

template <typename T, uint32_t D>
void Handle<T, D>::padding_dimensions(std::vector<size_t> &shape,
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

template <typename T, uint32_t D> void Handle<T, D>::create_queues() {
  num_of_queues = 16;
  cudaStream_t *ptr = new cudaStream_t[num_of_queues];
  for (int i = 0; i < num_of_queues; i++) {
    gpuErrchk(cudaStreamCreate(ptr + i));
  }
  queues = (void *)ptr;
}

template <typename T, uint32_t D> void Handle<T, D>::destroy_queues() {
  cudaStream_t *ptr = (cudaStream_t *)queues;
  for (int i = 0; i < num_of_queues; i++) {
    gpuErrchk(cudaStreamDestroy(ptr[i]));
  }
}

template <typename T, uint32_t D>
std::vector<T *>
Handle<T, D>::create_uniform_coords(std::vector<size_t> shape) {

  // to do make this GPU ptr
  std::vector<T *> coords(D);
  for (int d = 0; d < D; d++) {
    T *curr_coords = new T[shape[d]];
    for (int i = 0; i < shape[d]; i++) {
      curr_coords[i] = (T)i;
    }

    coords[d] = curr_coords;
  }
  uniform_coords_created = true;
  return coords;
}

template <typename T, uint32_t D> void Handle<T, D>::init_auto_tuning_table() {

  arch = 0; // default
#ifdef MGARD_CUDA_OPTIMIZE_VOLTA
  arch = 1;
  // printf("Optimized: Volta\n");
#endif
#ifdef MGARD_CUDA_OPTIMIZE_TURING
  arch = 2;
  // printf("Optimized: Turing\n");
#endif
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
  this->auto_tuning_cc[1][0][0] = 0;
  this->auto_tuning_cc[1][0][1] = 0;
  this->auto_tuning_cc[1][0][2] = 0;
  this->auto_tuning_cc[1][0][3] = 1;
  this->auto_tuning_cc[1][0][4] = 1;
  this->auto_tuning_cc[1][0][5] = 5;
  this->auto_tuning_cc[1][0][6] = 5;
  this->auto_tuning_cc[1][0][7] = 5;
  this->auto_tuning_cc[1][0][8] = 5;

  this->auto_tuning_mr1[1][0][0] = 0;
  this->auto_tuning_mr2[1][0][0] = 0;
  this->auto_tuning_mr3[1][0][0] = 0;
  this->auto_tuning_mr1[1][0][1] = 0;
  this->auto_tuning_mr2[1][0][1] = 1;
  this->auto_tuning_mr3[1][0][1] = 1;
  this->auto_tuning_mr1[1][0][2] = 1;
  this->auto_tuning_mr2[1][0][2] = 0;
  this->auto_tuning_mr3[1][0][2] = 0;
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

  this->auto_tuning_ts1[1][0][0] = 0;
  this->auto_tuning_ts2[1][0][0] = 0;
  this->auto_tuning_ts3[1][0][0] = 0;
  this->auto_tuning_ts1[1][0][1] = 0;
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

  this->auto_tuning_cc[1][1][0] = 0;
  this->auto_tuning_cc[1][1][1] = 0;
  this->auto_tuning_cc[1][1][2] = 0;
  this->auto_tuning_cc[1][1][3] = 0;
  this->auto_tuning_cc[1][1][4] = 4;
  this->auto_tuning_cc[1][1][5] = 5;
  this->auto_tuning_cc[1][1][6] = 6;
  this->auto_tuning_cc[1][1][7] = 6;
  this->auto_tuning_cc[1][1][8] = 5;

  this->auto_tuning_mr1[1][1][0] = 0;
  this->auto_tuning_mr2[1][1][0] = 0;
  this->auto_tuning_mr3[1][1][0] = 0;
  this->auto_tuning_mr1[1][1][1] = 0;
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

  this->auto_tuning_ts1[1][1][0] = 0;
  this->auto_tuning_ts2[1][1][0] = 0;
  this->auto_tuning_ts3[1][1][0] = 0;
  this->auto_tuning_ts1[1][1][1] = 0;
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
  this->auto_tuning_cc[2][0][0] = 0;
  this->auto_tuning_cc[2][0][1] = 0;
  this->auto_tuning_cc[2][0][2] = 0;
  this->auto_tuning_cc[2][0][3] = 0;
  this->auto_tuning_cc[2][0][4] = 3;
  this->auto_tuning_cc[2][0][5] = 5;
  this->auto_tuning_cc[2][0][6] = 5;
  this->auto_tuning_cc[2][0][7] = 5;
  this->auto_tuning_cc[2][0][8] = 4;

  this->auto_tuning_mr1[2][0][0] = 0;
  this->auto_tuning_mr2[2][0][0] = 0;
  this->auto_tuning_mr3[2][0][0] = 0;
  this->auto_tuning_mr1[2][0][1] = 0;
  this->auto_tuning_mr2[2][0][1] = 1;
  this->auto_tuning_mr3[2][0][1] = 1;
  this->auto_tuning_mr1[2][0][2] = 1;
  this->auto_tuning_mr2[2][0][2] = 0;
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

  this->auto_tuning_ts1[2][0][0] = 0;
  this->auto_tuning_ts2[2][0][0] = 0;
  this->auto_tuning_ts3[2][0][0] = 0;
  this->auto_tuning_ts1[2][0][1] = 0;
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

  this->auto_tuning_mr1[2][1][0] = 0;
  this->auto_tuning_mr2[2][1][0] = 0;
  this->auto_tuning_mr3[2][1][0] = 0;
  this->auto_tuning_mr1[2][1][1] = 0;
  this->auto_tuning_mr2[2][1][1] = 0;
  this->auto_tuning_mr3[2][1][1] = 1;
  this->auto_tuning_mr1[2][1][2] = 0;
  this->auto_tuning_mr2[2][1][2] = 0;
  this->auto_tuning_mr3[2][1][2] = 0;
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

  this->auto_tuning_ts1[2][1][0] = 0;
  this->auto_tuning_ts2[2][1][0] = 0;
  this->auto_tuning_ts3[2][1][0] = 0;
  this->auto_tuning_ts1[2][1][1] = 0;
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

template <typename T, uint32_t D>
void Handle<T, D>::destroy_auto_tuning_table() {
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

template <typename T, uint32_t D> void Handle<T, D>::allocate_workspace() {

  size_t dw_pitch;
  mgard_cuda::cudaMalloc3DHelper((void **)&(dw), &dw_pitch,
                                 (shapes_h[0][0] + 2) * sizeof(T),
                                 shapes_h[0][1] + 2, padded_linearized_depth);

  // ldws.push_back(dw_pitch / sizeof(T));
  // for (int i = 1; i < D_padded; i++) {
  //   ldws.push_back(shapes_h[0][i] + 2);
  // }
  lddw1 = dw_pitch / sizeof(T);
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

  mgard_cuda::cudaMallocHelper((void **)&ldws_d, D_padded * sizeof(int));
  mgard_cuda::cudaMemcpyAsyncHelper(*this, ldws_d, ldws_h.data(),
                                    D_padded * sizeof(int), mgard_cuda::H2D, 0);

  if (D > 3) {
    size_t db_pitch;
    mgard_cuda::cudaMalloc3DHelper((void **)&(db), &db_pitch,
                                   (shapes_h[0][0] + 2) * sizeof(T),
                                   shapes_h[0][1] + 2, padded_linearized_depth);

    // ldbs.push_back(db_pitch / sizeof(T));
    // for (int i = 1; i < D_padded; i++) {
    //   ldbs.push_back(shapes_h[0][i] + 2);
    // }
    lddb1 = db_pitch / sizeof(T);
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

    mgard_cuda::cudaMallocHelper((void **)&ldbs_d, D_padded * sizeof(int));
    mgard_cuda::cudaMemcpyAsyncHelper(*this, ldbs_d, ldbs_h.data(),
                                      D_padded * sizeof(int), mgard_cuda::H2D,
                                      0);
  }
}

template <typename T, uint32_t D> void Handle<T, D>::free_workspace() {
  mgard_cuda::cudaFreeHelper(dw);
  mgard_cuda::cudaFreeHelper(ldws_d);
  if (D > 3) {
    mgard_cuda::cudaFreeHelper(db);
    // delete [] ldbs_h;
    mgard_cuda::cudaFreeHelper(ldbs_d);
  }
}

template <typename T, uint32_t D>
mgard_cuda_config Handle<T, D>::default_config() {
  mgard_cuda_config config;
  config.l_target = -1; // no limit
  config.huff_dict_size = 8192;
  config.huff_block_size = 1024 * 30;
  config.enable_lz4 = true;
  config.lz4_block_size = 1 << 15;
  return config;
}

template <typename T, uint32_t D> Handle<T, D>::Handle() { create_queues(); }

template <typename T, uint32_t D>
Handle<T, D>::Handle(std::vector<size_t> shape) {
  std::reverse(shape.begin(), shape.end());
  std::vector<T *> coords = create_uniform_coords(shape);
  padding_dimensions(shape, coords);
  create_queues();
  init_auto_tuning_table();
  init(shape, coords, default_config());
}

template <typename T, uint32_t D>
Handle<T, D>::Handle(std::vector<size_t> shape, std::vector<T *> coords) {
  std::reverse(shape.begin(), shape.end());
  std::reverse(coords.begin(), coords.end());
  padding_dimensions(shape, coords);
  create_queues();
  init_auto_tuning_table();
  init(shape, coords, default_config());
}

template <typename T, uint32_t D>
Handle<T, D>::Handle(std::vector<size_t> shape, mgard_cuda_config config) {
  std::reverse(shape.begin(), shape.end());
  std::vector<T *> coords = create_uniform_coords(shape);
  padding_dimensions(shape, coords);
  create_queues();
  init_auto_tuning_table();
  init(shape, coords, config);
}

template <typename T, uint32_t D>
Handle<T, D>::Handle(std::vector<size_t> shape, std::vector<T *> coords,
                     mgard_cuda_config config) {
  std::reverse(shape.begin(), shape.end());
  std::reverse(coords.begin(), coords.end());
  padding_dimensions(shape, coords);
  create_queues();
  init_auto_tuning_table();
  init(shape, coords, config);
}

template <typename T, uint32_t D> void *Handle<T, D>::get(int i) {
  cudaStream_t *ptr = (cudaStream_t *)(this->queues);
  return (void *)(ptr + i);
}

template <typename T, uint32_t D> void Handle<T, D>::sync(int i) {
  cudaStream_t *ptr = (cudaStream_t *)(this->queues);
  gpuErrchk(cudaStreamSynchronize(ptr[i]));
}

template <typename T, uint32_t D> void Handle<T, D>::sync_all() {
  cudaStream_t *ptr = (cudaStream_t *)(this->queues);
  for (int i = 0; i < this->num_of_queues; i++) {
    gpuErrchk(cudaStreamSynchronize(ptr[i]));
  }
}

template <typename T, uint32_t D> Handle<T, D>::~Handle() {
  destroy_queues();
  if (initialized)
    destroy();
  if (auto_tuning_table_created)
    destroy_auto_tuning_table();
}

template class Handle<double, 1>;
template class Handle<float, 1>;
template class Handle<double, 2>;
template class Handle<float, 2>;
template class Handle<double, 3>;
template class Handle<float, 3>;
template class Handle<double, 4>;
template class Handle<float, 4>;
template class Handle<double, 5>;
template class Handle<float, 5>;

} // namespace mgard_cuda