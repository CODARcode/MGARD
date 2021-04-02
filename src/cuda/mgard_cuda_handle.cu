/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cascaded.hpp"
#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda_handle.h"
#include "cuda/mgard_cuda_helper.h"
#include "cuda/mgard_cuda_precompute_kernels.h"
#include "nvcomp.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace mgard {

template <typename T, int D>
void mgard_cuda_handle<T, D>::init(std::vector<size_t> shape,
                                   std::vector<T *> coords,
                                   mgard_cuda_config config) {

  // printf("Initializing handler\n");
  // for (int i = 0; i < shape[0]; i++)
  //   printf("%f ", coords[0][i]);
  // printf("\n");
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

  init_auto_tuning_table();

  // this->B = B;
  this->num_of_queues = 16;

  this->num_gpus = 1;
  this->dev_ids = new int[this->num_gpus];
  for (int i = 0; i < this->num_gpus; i++)
    this->dev_ids[i] = 0;

  // for (int i = 0; i < this->num_gpus; i++) {
  //  for (int j = 0; j < this->num_gpus; j++) {
  //    int r;
  //    cudaDeviceCanAccessPeer(&r, dev_ids[i], dev_ids[j]);
  //    if (r == 1)
  //      printf("%d - %d: Y\n", i, j);
  //    else
  //      printf("%d - %d: N\n", i, j);
  //  }
  // }

  // cudaStream_t *ptr = new cudaStream_t[this->num_of_queues];
  // for (int d = 0; d < this->num_gpus; d++) {
  //   // for (int i = 0; i < this->num_gpus*this->num_gpus; i++) {
  //     gpuErrchk(cudaStreamCreate(ptr + d));
  //   // }
  // }

  // for (int i = 0; i < this->num_of_queues; i++) {
  //   gpuErrchk(cudaStreamCreate(ptr + i));
  //   // std::cout << "created a stream: " << *(ptr+i) <<"\n";
  // }

  dev_assign = new int **[num_gpus];

  for (int dr = 0; dr < num_gpus; dr++) {
    dev_assign[dr] = new int *[num_gpus];
    for (int dc = 0; dc < num_gpus; dc++) {
      dev_assign[dr][dc] = new int[num_gpus];
      for (int df = 0; df < num_gpus; df++) {
        // round robin shift index
        int d = (df + dc + dr) % num_gpus;
        dev_assign[dr][dc][df] = dev_ids[d];
      }
    }
  }

  this->queue_per_block = num_of_queues;
  cudaStream_t *ptr =
      new cudaStream_t[num_gpus * num_gpus * num_gpus * queue_per_block];
  for (int dr = 0; dr < num_gpus; dr++) {
    for (int dc = 0; dc < num_gpus; dc++) {
      for (int df = 0; df < num_gpus; df++) {
        int d = dev_assign[dr][dc][df];
        mgard_cuda::cudaSetDeviceHelper(dev_ids[d]);
        for (int i = 0; i < queue_per_block; i++) {
          gpuErrchk(
              cudaStreamCreate(ptr +
                               (dr * num_gpus * num_gpus + dc * num_gpus + df) *
                                   queue_per_block +
                               i));
          // printf("create:%d\n",
          // (dr*num_gpus*num_gpus+dc*num_gpus+df)*queue_per_block+i);
        }
      }
    }
  }

  mgard_cuda::cudaSetDeviceHelper(dev_ids[0]);
  this->queues = (void *)ptr;
  this->opt = opt;

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
  padded_linearized_depth = shape[2] + 2;
  for (int i = 3; i < shape.size(); i++) {
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
    mgard_cuda::cudaMallocHelper((void **)&(curr_shape_d),
                                 D_padded * sizeof(int));
    mgard_cuda::cudaMemcpyAsyncHelper(*this, curr_shape_d, curr_shape_h,
                                      D_padded * sizeof(int), mgard_cuda::H2D,
                                      0);
    shapes_d.push_back(curr_shape_d);
  }

  processed_n = new int[D];
  processed_dims_h = new int *[D];
  processed_dims_d = new int *[D];

  thrust::device_vector<int> tmp(0);
  for (int d = 0; d < D; d++) {
    processed_n[d] = tmp.size();
    processed_dims_h[d] = new int[processed_n[d]];
    mgard_cuda::cudaMemcpyAsyncHelper(
        *this, processed_dims_h[d], thrust::raw_pointer_cast(tmp.data()),
        processed_n[d] * sizeof(int), mgard_cuda::D2H, 0);
    mgard_cuda::cudaMallocHelper((void **)&processed_dims_d[d],
                                 processed_n[d] * sizeof(int));
    mgard_cuda::cudaMemcpyAsyncHelper(
        *this, processed_dims_d[d], thrust::raw_pointer_cast(tmp.data()),
        processed_n[d] * sizeof(int), mgard_cuda::D2D, 0);
    tmp.push_back(d);
  }

  mgard_cuda::cudaMallocHelper((void **)&(quantizers),
                               (l_target + 1) * sizeof(T));

  // // handle coords
  for (int i = 0; i < shape.size(); i++) {
    if (!mgard_cuda::isGPUPointer(coords[i])) {
      T *curr_dcoords;
      mgard_cuda::cudaMallocHelper((void **)&(curr_dcoords),
                                   shape[i] * sizeof(T));
      mgard_cuda::cudaMemcpyAsyncHelper(*this, curr_dcoords, coords[i],
                                        shape[i] * sizeof(T), mgard_cuda::H2D,
                                        0);
      coords[i] =
          curr_dcoords; // coords will always point to GPU coords in the end
    }
  }

  // // calculate dist and ratio
  for (int i = 0; i < shape.size(); i++) {
    std::vector<T *> curr_ddist_l, curr_dratio_l;
    // for level 0
    int last_dist = dofs[i][0] - 1;
    T *curr_ddist0, *curr_dratio0;
    mgard_cuda::cudaMallocHelper((void **)&curr_ddist0, dofs[i][0] * sizeof(T));
    mgard_cuda::cudaMallocHelper((void **)&curr_dratio0,
                                 dofs[i][0] * sizeof(T));
    mgard_cuda::cudaMemsetHelper((void **)&curr_ddist0, dofs[i][0] * sizeof(T),
                                 0);
    mgard_cuda::cudaMemsetHelper((void **)&curr_dratio0, dofs[i][0] * sizeof(T),
                                 0);
    curr_ddist_l.push_back(curr_ddist0);
    curr_dratio_l.push_back(curr_dratio0);

    mgard_cuda::calc_cpt_dist(*this, dofs[i][0], coords[i], curr_ddist_l[0], 0);
    // mgard_cuda::calc_cpt_dist_ratio(*this, dofs[i][0], coords[i],
    // curr_dratio_l[0], 0);
    mgard_cuda::dist_to_ratio(*this, last_dist, curr_ddist_l[0],
                              curr_dratio_l[0], 0);
    if (dofs[i][0] % 2 == 0) {
      // padding
      mgard_cuda::cudaMemcpyAsyncHelper(*this, curr_ddist_l[0] + dofs[i][0] - 1,
                                        curr_ddist_l[0] + dofs[i][0] - 2,
                                        sizeof(T), mgard_cuda::D2D, 0);
      last_dist += 1;

      T extra_ratio = 0.5;
      mgard_cuda::cudaMemcpyAsyncHelper(
          *this, curr_dratio_l[0] + dofs[i][0] - 2, &extra_ratio, sizeof(T),
          mgard_cuda::H2D, 0);
    }

    // for l = 1 ... l_target
    for (int l = 1; l < l_target + 1; l++) {
      T *curr_ddist, *curr_dratio;
      mgard_cuda::cudaMallocHelper((void **)&curr_ddist,
                                   dofs[i][l] * sizeof(T));
      mgard_cuda::cudaMallocHelper((void **)&curr_dratio,
                                   dofs[i][l] * sizeof(T));
      mgard_cuda::cudaMemsetHelper((void **)&curr_ddist, dofs[i][0] * sizeof(T),
                                   0);
      mgard_cuda::cudaMemsetHelper((void **)&curr_dratio,
                                   dofs[i][0] * sizeof(T), 0);
      curr_ddist_l.push_back(curr_ddist);
      curr_dratio_l.push_back(curr_dratio);
      mgard_cuda::reduce_two_dist(*this, last_dist, curr_ddist_l[l - 1],
                                  curr_ddist_l[l], 0);
      last_dist /= 2;
      mgard_cuda::dist_to_ratio(*this, last_dist, curr_ddist_l[l],
                                curr_dratio_l[l], 0);

      if (last_dist % 2 != 0) {
        mgard_cuda::cudaMemcpyAsyncHelper(*this, curr_ddist_l[l] + last_dist,
                                          curr_ddist_l[l] + last_dist - 1,
                                          sizeof(T), mgard_cuda::D2D, 0);
        last_dist += 1;

        T extra_ratio = 0.5;
        mgard_cuda::cudaMemcpyAsyncHelper(
            *this, curr_dratio_l[l] + last_dist - 2, &extra_ratio, sizeof(T),
            mgard_cuda::H2D, 0);
      }
    }
    dist.push_back(curr_ddist_l);
    ratio.push_back(curr_dratio_l);
  }

  for (int i = 0; i < shape.size(); i++) {
    std::vector<T *> curr_am_l, curr_bm_l;
    for (int l = 0; l < l_target + 1; l++) {
      T *curr_am, *curr_bm;
      mgard_cuda::cudaMallocHelper((void **)&curr_am, dofs[i][l] * sizeof(T));
      mgard_cuda::cudaMallocHelper((void **)&curr_bm, dofs[i][l] * sizeof(T));
      curr_am_l.push_back(curr_am);
      curr_bm_l.push_back(curr_bm);
      mgard_cuda::calc_am_bm(*this, dofs[i][l], dist[i][l], curr_am_l[l],
                             curr_bm_l[l], 0);
    }
    am.push_back(curr_am_l);
    bm.push_back(curr_bm_l);
  }

  huff_dict_size = config.huff_dict_size;
  huff_block_size = config.huff_block_size;
  enable_lz4 = config.enable_lz4;
  lz4_block_size = config.lz4_block_size;

  // printf("Done pre-compute am and bm 2\n");

  // printf("sizeof huffman_flags: %d*%d*%d+%d\n",
  // shape[0],shape[1],linearized_depth,sizeof(mgard_cuda::quant_meta<T>)/sizeof(int));
  // mgard_cuda::cudaMallocHelper((void **)&(huffman_flags),
  // (shape[0]*shape[1]*linearized_depth+sizeof(mgard_cuda::quant_meta<T>)/sizeof(int))
  // * sizeof(bool)); for (int i = 0; i < D_padded; i++) {
  //   ldhuffman_flags.push_back(shape[i]);
  // }

  // size_t dw_pitch;
  // mgard_cuda::cudaMalloc3DHelper((void **)&(dw), &dw_pitch,
  //                                (shapes_h[0][0]+2) * sizeof(T),
  //                                shapes_h[0][1]+2, padded_linearized_depth);

  // ldws.push_back(dw_pitch / sizeof(T));
  // for (int i = 1; i < D_padded; i++) {
  //   ldws.push_back(shapes_h[0][i] + 2);
  // }
  // lddw1 = dw_pitch / sizeof(T);
  // lddw2 = shapes_h[0][1] +2;

  // ldws_h = new int[D_padded];
  // ldws_h[0] = dw_pitch / sizeof(T);
  // for (int i = 1; i < D_padded; i++) {
  //   ldws_h[i] = shapes_h[0][i] + 2;
  // }
  // mgard_cuda::cudaMallocHelper((void **)&ldws_d, D_padded * sizeof(int));
  // mgard_cuda::cudaMemcpyAsyncHelper(*this, ldws_d, ldws_h,
  //   D_padded * sizeof(int), mgard_cuda::H2D, 0);

  // if (D > 3) {
  //   size_t db_pitch;
  //   mgard_cuda::cudaMalloc3DHelper((void **)&(db), &db_pitch,
  //                                  (shapes_h[0][0]+2) * sizeof(T),
  //                                  shapes_h[0][1]+2,
  //                                  padded_linearized_depth);

  //   ldbs.push_back(db_pitch / sizeof(T));
  //   for (int i = 1; i < D_padded; i++) {
  //     ldbs.push_back(shapes_h[0][i] + 2);
  //   }
  //   lddb1 = db_pitch / sizeof(T);
  //   lddb2 = shapes_h[0][1] +2;

  //   ldbs_h = new int[D_padded];
  //   ldbs_h[0] = db_pitch / sizeof(T);
  //   for (int i = 1; i < D_padded; i++) {
  //     ldbs_h[i] = shapes_h[0][i] + 2;
  //   }
  //   mgard_cuda::cudaMallocHelper((void **)&ldbs_d, D_padded * sizeof(int));
  //   mgard_cuda::cudaMemcpyAsyncHelper(*this, ldbs_d, ldbs_h,
  //     D_padded * sizeof(int), mgard_cuda::H2D, 0);
  // }
}

template <typename T, int D>
void mgard_cuda_handle<T, D>::padding_dimensions(std::vector<size_t> &shape,
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

template <typename T, int D>
void mgard_cuda_handle<T, D>::init_auto_tuning_table() {
  int num_arch = 3;
  int num_precision = 2;
  int num_range = 9;
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
}

template <typename T, int D>
void mgard_cuda_handle<T, D>::allocate_workspace() {
  size_t dw_pitch;
  mgard_cuda::cudaMalloc3DHelper((void **)&(dw), &dw_pitch,
                                 (shapes_h[0][0] + 2) * sizeof(T),
                                 shapes_h[0][1] + 2, padded_linearized_depth);

  ldws.push_back(dw_pitch / sizeof(T));
  for (int i = 1; i < D_padded; i++) {
    ldws.push_back(shapes_h[0][i] + 2);
  }
  lddw1 = dw_pitch / sizeof(T);
  lddw2 = shapes_h[0][1] + 2;

  ldws_h = new int[D_padded];
  ldws_h[0] = dw_pitch / sizeof(T);
  for (int i = 1; i < D_padded; i++) {
    ldws_h[i] = shapes_h[0][i] + 2;
  }
  mgard_cuda::cudaMallocHelper((void **)&ldws_d, D_padded * sizeof(int));
  mgard_cuda::cudaMemcpyAsyncHelper(*this, ldws_d, ldws_h,
                                    D_padded * sizeof(int), mgard_cuda::H2D, 0);

  if (D > 3) {
    size_t db_pitch;
    mgard_cuda::cudaMalloc3DHelper((void **)&(db), &db_pitch,
                                   (shapes_h[0][0] + 2) * sizeof(T),
                                   shapes_h[0][1] + 2, padded_linearized_depth);

    ldbs.push_back(db_pitch / sizeof(T));
    for (int i = 1; i < D_padded; i++) {
      ldbs.push_back(shapes_h[0][i] + 2);
    }
    lddb1 = db_pitch / sizeof(T);
    lddb2 = shapes_h[0][1] + 2;

    ldbs_h = new int[D_padded];
    ldbs_h[0] = db_pitch / sizeof(T);
    for (int i = 1; i < D_padded; i++) {
      ldbs_h[i] = shapes_h[0][i] + 2;
    }
    mgard_cuda::cudaMallocHelper((void **)&ldbs_d, D_padded * sizeof(int));
    mgard_cuda::cudaMemcpyAsyncHelper(
        *this, ldbs_d, ldbs_h, D_padded * sizeof(int), mgard_cuda::H2D, 0);
  }
}

template <typename T, int D> void mgard_cuda_handle<T, D>::free_workspace() {
  mgard_cuda::cudaFreeHelper(dw);
  delete[] ldws_h;
  mgard_cuda::cudaFreeHelper(ldws_d);
  if (D > 3) {
    mgard_cuda::cudaFreeHelper(db);
    delete[] ldbs_h;
    mgard_cuda::cudaFreeHelper(ldbs_d);
  }
}

template <typename T, int D>
mgard_cuda_config mgard_cuda_handle<T, D>::default_config() {
  mgard_cuda_config config;
  config.l_target = -1; // no limit
  config.huff_dict_size = 8192;
  config.huff_block_size = 1024 * 30;
  config.enable_lz4 = true;
  config.lz4_block_size = 1 << 15;
  return config;
}

template <typename T, int D> mgard_cuda_handle<T, D>::mgard_cuda_handle() {
  std::vector<T> coords_r(5), coords_c(5), coords_f(5);
  std::vector<size_t> shape(3);
  shape[2] = 5;
  shape[1] = 5;
  shape[0] = 5;
  std::vector<T *> coords(3);
  coords[2] = coords_r.data();
  coords[1] = coords_c.data();
  coords[0] = coords_f.data();
  for (int d = 0; d < 3; d++) {
    T *curr_coords = new T[shape[d]];
    for (int i = 0; i < shape[d]; i++) {
      curr_coords[i] = (T)i;
    }
    coords[d] = curr_coords;
  }
  padding_dimensions(shape, coords);
  init(shape, coords, default_config());
}

template <typename T, int D>
mgard_cuda_handle<T, D>::mgard_cuda_handle(std::vector<size_t> shape) {
  std::vector<T *> coords(D);
  for (int d = 0; d < D; d++) {
    T *curr_coords = new T[shape[d]];
    for (int i = 0; i < shape[d]; i++) {
      curr_coords[i] = (T)i;
    }
    coords[d] = curr_coords;
  }

  padding_dimensions(shape, coords);

  init(shape, coords, default_config());
}

template <typename T, int D>
mgard_cuda_handle<T, D>::mgard_cuda_handle(std::vector<size_t> shape,
                                           std::vector<T *> coords) {
  padding_dimensions(shape, coords);
  init(shape, coords, default_config());
}

template <typename T, int D>
mgard_cuda_handle<T, D>::mgard_cuda_handle(std::vector<size_t> shape,
                                           mgard_cuda_config config) {
  std::vector<T *> coords(D);
  for (int d = 0; d < D; d++) {
    T *curr_coords = new T[shape[d]];
    for (int i = 0; i < shape[d]; i++) {
      curr_coords[i] = (T)i;
    }
    coords[d] = curr_coords;
  }

  padding_dimensions(shape, coords);

  init(shape, coords, config);
}

template <typename T, int D>
mgard_cuda_handle<T, D>::mgard_cuda_handle(std::vector<size_t> shape,
                                           std::vector<T *> coords,
                                           mgard_cuda_config config) {
  padding_dimensions(shape, coords);
  init(shape, coords, config);
}

template <typename T, int D> void *mgard_cuda_handle<T, D>::get(int i) {
  cudaStream_t *ptr = (cudaStream_t *)(this->queues);
  return (void *)(ptr + i);
}

template <typename T, int D>
int mgard_cuda_handle<T, D>::get_queues(int dr, int dc, int df) {
  return (dr * num_gpus * num_gpus + dc * num_gpus + df) * queue_per_block;
}

template <typename T, int D> void mgard_cuda_handle<T, D>::sync(int i) {
  cudaStream_t *ptr = (cudaStream_t *)(this->queues);
  gpuErrchk(cudaStreamSynchronize(ptr[i]));
}

template <typename T, int D> void mgard_cuda_handle<T, D>::sync_all() {
  cudaStream_t *ptr = (cudaStream_t *)(this->queues);
  for (int i = 0; i < this->num_gpus * this->num_gpus * this->num_gpus *
                          this->queue_per_block;
       i++) {
    gpuErrchk(cudaStreamSynchronize(ptr[i]));
  }
}

template <typename T, int D> mgard_cuda_handle<T, D>::~mgard_cuda_handle() {
  destory();
}

template <typename T, int D> void mgard_cuda_handle<T, D>::destory() {
  cudaStream_t *ptr = (cudaStream_t *)(this->queues);
  for (int i = 0; i < this->num_gpus * this->num_gpus * this->num_gpus *
                          this->queue_per_block;
       i++) {
    gpuErrchk(cudaStreamDestroy(ptr[i]));
  }
}

template class mgard_cuda_handle<double, 1>;
template class mgard_cuda_handle<float, 1>;
template class mgard_cuda_handle<double, 2>;
template class mgard_cuda_handle<float, 2>;
template class mgard_cuda_handle<double, 3>;
template class mgard_cuda_handle<float, 3>;
template class mgard_cuda_handle<double, 4>;
template class mgard_cuda_handle<float, 4>;
template class mgard_cuda_handle<double, 5>;
template class mgard_cuda_handle<float, 5>;

} // namespace mgard