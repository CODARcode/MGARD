/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cuda/CommonInternal.h"

#include "cuda/GridProcessingKernel.h"
#include "cuda/GridProcessingKernel3D.h"
#include "cuda/IterativeProcessingKernel.h"
#include "cuda/IterativeProcessingKernel3D.h"
#include "cuda/LevelwiseProcessingKernel.h"
#include "cuda/LinearProcessingKernel.h"
#include "cuda/LinearProcessingKernel3D.h"

#include "cuda/DataRefactoring.h"

#include <iostream>

#include <chrono>
namespace mgard_cuda {

bool store = false;
bool verify = false;

template <typename T, uint32_t D>
void refactor_reo(Handle<T, D> &handle, T *dv, std::vector<int> ldvs,
                  int l_target) {

  int *ldvs_h = new int[handle.D_padded];
  for (int d = 0; d < handle.D_padded; d++) {
    ldvs_h[d] = ldvs[d];
  }
  int *ldvs_d;
  cudaMallocHelper((void **)&ldvs_d, handle.D_padded * sizeof(int));
  cudaMemcpyAsyncHelper(handle, ldvs_d, ldvs_h, handle.D_padded * sizeof(int),
                        H2D, 0);

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(handle.shapes_h[0][d]) + "_";
  // std::cout << prefix << std::endl;

  if (D <= 3) {
    thrust::device_vector<int> empty_vector(0);
    int unprocessed_n = 0;
    int *unprocessed_dims = thrust::raw_pointer_cast(empty_vector.data());

    for (int l = 0; l < l_target; ++l) {
      // printf("[gpu] l = %d\n", l);
      int stride = std::pow(2, l);
      int Cstride = stride * 2;
      int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
      int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);
      // printf("range_l: %d, range_lp1: %d\n", range_l, range_lp1);

      thrust::device_vector<int> shape(handle.D_padded);
      thrust::device_vector<int> shape_c(handle.D_padded);
      for (int d = 0; d < handle.D_padded; d++) {
        shape[d] = handle.dofs[d][l];
        shape_c[d] = handle.dofs[d][l + 1];
      }

      // printf("input v\n");
      // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      // handle.dofs[0][l],
      //                     dv, ldvs[0], ldvs[1], ldvs[0]);

      lwpk<T, D, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l], dv,
                       ldvs_d, handle.dw, handle.ldws_d, 0);

      // printf("before pi_Ql_reo\n");
      // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      // handle.dofs[0][l],
      //                   handle.dw, handle.ldws_h[0], handle.ldws_h[1]
      //                   ,handle.ldws_h[0]);

      // printf("before pi_Ql_reo\n");
      // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      // handle.dofs[0][l],
      //                     dv, ldvs[0], ldvs[1], ldvs[0]);

      // thrust::device_vector<int> unprocessed_dims(0);
      int lddv1 = 1, lddv2 = 1, lddw1 = 1, lddw2 = 1;
      for (int s = 0; s < 1; s++) {
        lddv1 *= ldvs[s];
      }
      for (int s = 1; s < 2; s++) {
        lddv2 *= ldvs[s];
      }
      for (int s = 0; s < 1; s++) {
        lddw1 *= handle.ldws_h[s];
      }
      for (int s = 1; s < 2; s++) {
        lddw2 *= handle.ldws_h[s];
      }

      T *null = NULL;
      // printf("gpk_reo\n");
      // gpk_reo<T, D, D, true, true, 1>(handle,
      //           handle.shapes_h[l], handle.shapes_d[l], handle.shapes_d[l+1],
      //           handle.ldws_d, ldvs_d, unprocessed_n, unprocessed_dims, 2, 1,
      //           0, handle.ratio[2][l], handle.ratio[1][l],
      //           handle.ratio[0][l], handle.dw, handle.ldws_h[0],
      //           handle.ldws_h[1], dv, ldvs_h[0], ldvs_h[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs_h[0], ldvs_h[1], 0, 0, handle.dofs[0][l+1]),
      //           ldvs_h[0], ldvs_h[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs_h[0], ldvs_h[1], 0, handle.dofs[1][l+1], 0),
      //           ldvs_h[0], ldvs_h[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs_h[0], ldvs_h[1], handle.dofs[2][l+1], 0, 0),
      //           ldvs_h[0], ldvs_h[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs_h[0], ldvs_h[1], 0, handle.dofs[1][l+1],
      //           handle.dofs[0][l+1]), ldvs_h[0], ldvs_h[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs_h[0], ldvs_h[1], handle.dofs[2][l+1], 0,
      //           handle.dofs[0][l+1]), ldvs_h[0], ldvs_h[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs_h[0], ldvs_h[1], handle.dofs[2][l+1],
      //           handle.dofs[1][l+1], 0), ldvs_h[0], ldvs_h[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs_h[0], ldvs_h[1], handle.dofs[2][l+1],
      //           handle.dofs[1][l+1], handle.dofs[0][l+1]), ldvs_h[0],
      //           ldvs_h[1],
      //           //null, ldvs[0], ldvs[1],
      //           0,
      //           handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

      gpk_reo_3d(
          handle, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
          handle.ratio[2][l], handle.ratio[1][l], handle.ratio[0][l], handle.dw,
          handle.ldws_h[0], handle.ldws_h[1], dv, ldvs_h[0], ldvs_h[1],
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs_h[0], ldvs_h[1], 0, 0, handle.dofs[0][l + 1]),
          ldvs_h[0], ldvs_h[1],
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs_h[0], ldvs_h[1], 0, handle.dofs[1][l + 1], 0),
          ldvs_h[0], ldvs_h[1],
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs_h[0], ldvs_h[1], handle.dofs[2][l + 1], 0, 0),
          ldvs_h[0], ldvs_h[1],
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs_h[0], ldvs_h[1], 0, handle.dofs[1][l + 1],
                       handle.dofs[0][l + 1]),
          ldvs_h[0], ldvs_h[1],
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs_h[0], ldvs_h[1], handle.dofs[2][l + 1], 0,
                       handle.dofs[0][l + 1]),
          ldvs_h[0], ldvs_h[1],
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs_h[0], ldvs_h[1], handle.dofs[2][l + 1],
                       handle.dofs[1][l + 1], 0),
          ldvs_h[0], ldvs_h[1],
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs_h[0], ldvs_h[1], handle.dofs[2][l + 1],
                       handle.dofs[1][l + 1], handle.dofs[0][l + 1]),
          ldvs_h[0], ldvs_h[1],
          // null, ldvs[0], ldvs[1],
          0, handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
      // printf("gpk_reo\n");
      // //handle.sync(0);
      verify_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
                         handle.dofs[0][l], dv, ldvs_h[0], ldvs_h[1], ldvs_h[0],
                         prefix + "gpk_reo_3d" + "_level_" + std::to_string(l),
                         store, verify);

      // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      // handle.dofs[0][l],
      //                     dv, ldvs[0], ldvs[1], ldvs[0]);

      // gpk_reo<T, D, D, true, false, 1>(handle,
      //           shape, shape_c, handle.ldws_h, ldvs, unprocessed_dims,
      //           2, 1, 0,
      //           handle.ratio[2][l], handle.ratio[1][l], handle.ratio[0][l],
      //           handle.dw, handle.ldws_h[0], handle.ldws_h[1],
      //           dv, ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs[0], ldvs[1], 0, 0, handle.dofs[0][l+1]),
      //           ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs[0], ldvs[1], 0, handle.dofs[1][l+1], 0),
      //           ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs[0], ldvs[1], handle.dofs[2][l+1], 0, 0),
      //           ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs[0], ldvs[1], 0, handle.dofs[1][l+1],
      //           handle.dofs[0][l+1]), ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs[0], ldvs[1], handle.dofs[2][l+1], 0,
      //           handle.dofs[0][l+1]), ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs[0], ldvs[1], handle.dofs[2][l+1],
      //           handle.dofs[1][l+1], 0), ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs[0], ldvs[1], handle.dofs[2][l+1],
      //           handle.dofs[1][l+1], handle.dofs[0][l+1]), ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           0,
      //           handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

      // printf("after interpolate\n");
      // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      // handle.dofs[0][l],
      //                     dv, ldvs[0], ldvs[1], ldvs[0]);

      // gpk_reo<T, D, D, false, true, 1>(handle,
      //           shape, shape_c, handle.ldws_h, ldvs, unprocessed_dims,
      //           2, 1, 0,
      //           handle.ratio[2][l], handle.ratio[1][l], handle.ratio[0][l],
      //           handle.dw, handle.ldws_h[0], handle.ldws_h[1],
      //           dv, ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs[0], ldvs[1], 0, 0, handle.dofs[0][l+1]),
      //           ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs[0], ldvs[1], 0, handle.dofs[1][l+1], 0),
      //           ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs[0], ldvs[1], handle.dofs[2][l+1], 0, 0),
      //           ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs[0], ldvs[1], 0, handle.dofs[1][l+1],
      //           handle.dofs[0][l+1]), ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs[0], ldvs[1], handle.dofs[2][l+1], 0,
      //           handle.dofs[0][l+1]), ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs[0], ldvs[1], handle.dofs[2][l+1],
      //           handle.dofs[1][l+1], 0), ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           dv+get_idx(ldvs[0], ldvs[1], handle.dofs[2][l+1],
      //           handle.dofs[1][l+1], handle.dofs[0][l+1]), ldvs[0], ldvs[1],
      //           //null, ldvs[0], ldvs[1],
      //           0,
      //           handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

      // printf("after pi_Ql_reo\n");
      // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      // handle.dofs[0][l],
      //                     dv, ldvs[0], ldvs[1], ldvs[0]);

      thrust::device_vector<int> processed_dims(0);

      if (D >= 1) {
        // lpk_reo_1<T, D>(handle,
        //                 handle.shapes_h[l], handle.shapes_h[l+1],
        //                 handle.shapes_d[l], handle.shapes_d[l+1],
        //                 ldvs_d, handle.ldws_d,
        //                 handle.processed_n[0], handle.processed_dims_h[0],
        //                 handle.processed_dims_d[0], 2, 1, 0,
        //                 handle.dist[0][l], handle.ratio[0][l],
        //                 dv, ldvs_h[0], ldvs_h[1],
        //                 dv+get_idx(ldvs_h[0], ldvs_h[1], 0, 0,
        //                 handle.dofs[0][l+1]), ldvs_h[0], ldvs_h[1],
        //                 handle.dw, handle.ldws_h[0], handle.ldws_h[1],
        //                 0,
        //                 handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

        lpk_reo_1_3d(
            handle, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
            handle.dofs[0][l + 1], handle.dofs[2][l + 1], handle.dofs[1][l + 1],
            handle.dofs[0][l + 1], handle.dist[0][l], handle.ratio[0][l], dv,
            ldvs_h[0], ldvs_h[1],
            dv + get_idx(ldvs_h[0], ldvs_h[1], 0, 0, handle.dofs[0][l + 1]),
            ldvs_h[0], ldvs_h[1], handle.dw, handle.ldws_h[0], handle.ldws_h[1],
            0,
            handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

        // //handle.sync(0);
        verify_matrix_cuda(
            handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l + 1],
            handle.dw, handle.ldws_h[0], handle.ldws_h[1], handle.ldws_h[0],
            prefix + "lpk_reo_1_3d" + "_level_" + std::to_string(l), store,
            verify);

        processed_dims.push_back(0);

        // printf("after mass_trans_multiply_1_cpt\n");
        // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
        // handle.dofs[0][l+1],
        //                 handle.dw, handle.ldws_h[0], handle.ldws_h[1]
        //                 ,handle.ldws_h[0]);

        // ipk_1<T, D>(handle,
        //             handle.shapes_h[l], handle.shapes_h[l+1],
        //             handle.shapes_d[l], handle.shapes_d[l+1],
        //             handle.ldws_d, handle.ldws_d,
        //             handle.processed_n[0], handle.processed_dims_h[0],
        //             handle.processed_dims_d[0], 2, 1, 0, handle.am[0][l+1],
        //             handle.bm[0][l+1], handle.dist[0][l+1], handle.dw,
        //             handle.ldws_h[0], handle.ldws_h[1], 0,
        //             handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);

        ipk_1_3d(
            handle, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l + 1],
            handle.am[0][l + 1], handle.bm[0][l + 1], handle.dist[0][l + 1],
            handle.dw, handle.ldws_h[0], handle.ldws_h[1], 0,
            handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);

        // //handle.sync(0);
        verify_matrix_cuda(
            handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l + 1],
            handle.dw, handle.ldws_h[0], handle.ldws_h[1], handle.ldws_h[0],
            prefix + "ipk_1_3d" + "_level_" + std::to_string(l), store, verify);

        // printf("after solve_tridiag_1_cpt\n");
        // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
        // handle.dofs[0][l+1],
        //                 handle.dw, handle.ldws_h[0], handle.ldws_h[1]
        //                 ,handle.ldws_h[0]);

        if (D == 1) {
          lwpk<T, D, ADD>(handle, handle.shapes_h[l + 1],
                          handle.shapes_d[l + 1], handle.dw, handle.ldws_d, dv,
                          ldvs_d, 0);
          // printf("after add\n");
          // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
          // handle.dofs[0][l],
          //                 dv, ldvs[0], ldvs[1], ldvs[0]);
        }
      }

      if (D >= 2) {
        // lpk_reo_2<T, D>(handle,
        //                 handle.shapes_h[l], handle.shapes_h[l+1],
        //                 handle.shapes_d[l], handle.shapes_d[l+1],
        //                 handle.ldws_d, handle.ldws_d,
        //                 handle.processed_n[1], handle.processed_dims_h[1],
        //                 handle.processed_dims_d[1], 2, 1, 0,
        //                 handle.dist[1][l], handle.ratio[1][l],
        //                 handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1],
        //                 0, 0, 0), handle.ldws_h[0], handle.ldws_h[1],
        //                 handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1],
        //                 0, handle.dofs[1][l+1], 0), handle.ldws_h[0],
        //                 handle.ldws_h[1], handle.dw+get_idx(handle.ldws_h[0],
        //                 handle.ldws_h[1], 0, 0, handle.dofs[0][l+1]),
        //                 handle.ldws_h[0], handle.ldws_h[1], 0,
        //                 handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

        lpk_reo_2_3d(
            handle, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l + 1],
            handle.dofs[1][l + 1], handle.dist[1][l], handle.ratio[1][l],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0, 0, 0),
            handle.ldws_h[0], handle.ldws_h[1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
                                handle.dofs[1][l + 1], 0),
            handle.ldws_h[0], handle.ldws_h[1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0, 0,
                                handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], 0,
            handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

        // //handle.sync(0);
        verify_matrix_cuda(
            handle.dofs[2][l], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0, 0,
                                handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], handle.ldws_h[0],
            prefix + "lpk_reo_2_3d" + "_level_" + std::to_string(l), store,
            verify);

        // printf("after mass_trans_multiply_2_cpt\n");
        // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l+1],
        // handle.dofs[0][l+1],
        //                 handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1],
        //                 0, 0, handle.dofs[0][l+1]), handle.ldws_h[0],
        //                 handle.ldws_h[1] ,handle.ldws_h[0]);

        // ipk_2<T, D>(handle,
        //             handle.shapes_h[l], handle.shapes_h[l+1],
        //             handle.shapes_d[l], handle.shapes_d[l+1],
        //             handle.ldws_d, handle.ldws_d,
        //             handle.processed_n[1], handle.processed_dims_h[1],
        //             handle.processed_dims_d[1], 2, 1, 0, handle.am[1][l+1],
        //             handle.bm[1][l+1], handle.dist[1][l+1],
        //             handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
        //             0, handle.dofs[0][l+1]), handle.ldws_h[0],
        //             handle.ldws_h[1], 0,
        //             handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);

        ipk_2_3d(
            handle, handle.dofs[2][l], handle.dofs[1][l + 1],
            handle.dofs[0][l + 1], handle.am[1][l + 1], handle.bm[1][l + 1],
            handle.dist[1][l + 1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0, 0,
                                handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], 0,
            handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);

        // handle.sync(0);
        verify_matrix_cuda(
            handle.dofs[2][l], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0, 0,
                                handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], handle.ldws_h[0],
            prefix + "ipk_2_3d" + "_level_" + std::to_string(l), store, verify);

        // printf("after solve_tridiag_2_cpt\n");
        // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l+1],
        // handle.dofs[0][l+1],
        //                 handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1],
        //                 0, 0, handle.dofs[0][l+1]), handle.ldws_h[0],
        //                 handle.ldws_h[1] ,handle.ldws_h[0]);

        // printf("before add\n");
        // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
        // handle.dofs[0][l],
        //                   dv, ldvs[0], ldvs[1], ldvs[0]);

        if (D == 2) {
          lwpk<T, D, ADD>(
              handle, handle.shapes_h[l + 1], handle.shapes_d[l + 1],
              handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0, 0,
                                  handle.dofs[0][l + 1]),
              handle.ldws_d, dv, ldvs_d, 0);
          // printf("after add\n");
          // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
          // handle.dofs[0][l],
          //                   dv, ldvs[0], ldvs[1], ldvs[0]);
        }
      }

      if (D == 3) {
        processed_dims.push_back(1);
        lpk_reo_3<T, D>(
            handle, handle.shapes_h[l], handle.shapes_h[l + 1],
            handle.shapes_d[l], handle.shapes_d[l + 1], handle.ldws_d,
            handle.ldws_d, handle.processed_n[2], handle.processed_dims_h[2],
            handle.processed_dims_d[2], 2, 1, 0, handle.dist[2][l],
            handle.ratio[2][l],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0, 0,
                                handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1],
                                handle.dofs[2][l + 1], 0,
                                handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
                                handle.dofs[1][l + 1], handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], 0,
            handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

        // lpk_reo_3_3d(handle,
        //              handle.dofs[2][l], handle.dofs[1][l+1],
        //              handle.dofs[0][l+1], handle.dofs[2][l+1],
        //              handle.dist[2][l], handle.ratio[2][l],
        //              handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
        //              0, handle.dofs[0][l+1]), handle.ldws_h[0],
        //              handle.ldws_h[1], handle.dw+get_idx(handle.ldws_h[0],
        //              handle.ldws_h[1], handle.dofs[2][l+1], 0,
        //              handle.dofs[0][l+1]), handle.ldws_h[0],
        //              handle.ldws_h[1], handle.dw+get_idx(handle.ldws_h[0],
        //              handle.ldws_h[1], 0, handle.dofs[1][l+1],
        //              handle.dofs[0][l+1]), handle.ldws_h[0],
        //              handle.ldws_h[1], 0,
        //              handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

        // handle.sync(0);
        verify_matrix_cuda(
            handle.dofs[2][l + 1], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
                                handle.dofs[1][l + 1], handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], handle.ldws_h[0],
            prefix + "lpk_reo_3_3d" + "_level_" + std::to_string(l), store,
            verify);

        // printf("after mass_trans_multiply_3_cpt\n");
        // print_matrix_cuda(handle.dofs[2][l+1], handle.dofs[1][l+1],
        // handle.dofs[0][l+1],
        //                 handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1],
        //                 0, handle.dofs[1][l+1], handle.dofs[0][l+1]),
        //                 handle.ldws_h[0], handle.ldws_h[1]
        //                 ,handle.ldws_h[0]);

        // ipk_3<T, D>(handle,
        //             handle.shapes_h[l], handle.shapes_h[l+1],
        //             handle.shapes_d[l], handle.shapes_d[l+1],
        //             handle.ldws_d, handle.ldws_d,
        //             handle.processed_n[2], handle.processed_dims_h[2],
        //             handle.processed_dims_d[2], 2, 1, 0, handle.am[2][l+1],
        //             handle.bm[2][l+1], handle.dist[2][l+1],
        //             handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
        //             handle.dofs[1][l+1], handle.dofs[0][l+1]),
        //             handle.ldws_h[0], handle.ldws_h[1], 0,
        //             handle.auto_tuning_ts3[handle.arch][handle.precision][range_lp1]);

        ipk_3_3d(
            handle, handle.dofs[2][l + 1], handle.dofs[1][l + 1],
            handle.dofs[0][l + 1], handle.am[2][l + 1], handle.bm[2][l + 1],
            handle.dist[2][l + 1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
                                handle.dofs[1][l + 1], handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], 0,
            handle.auto_tuning_ts3[handle.arch][handle.precision][range_lp1]);

        // handle.sync(0);
        verify_matrix_cuda(
            handle.dofs[2][l + 1], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
                                handle.dofs[1][l + 1], handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], handle.ldws_h[0],
            prefix + "ipk_3_3d" + "_level_" + std::to_string(l), store, verify);

        // printf("after solve_tridiag_3_cpt\n");
        // print_matrix_cuda(handle.dofs[2][l+1], handle.dofs[1][l+1],
        // handle.dofs[0][l+1],
        //                 handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1],
        //                 0, handle.dofs[1][l+1], handle.dofs[0][l+1]),
        //                 handle.ldws_h[0], handle.ldws_h[1]
        //                 ,handle.ldws_h[0]);

        if (D == 3) {
          lwpk<T, D, ADD>(
              handle, handle.shapes_h[l + 1], handle.shapes_d[l + 1],
              handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
                                  handle.dofs[1][l + 1], handle.dofs[0][l + 1]),
              handle.ldws_d, dv, ldvs_d, 0);

          // handle.sync(0);
          verify_matrix_cuda(
              handle.dofs[2][l + 1], handle.dofs[1][l + 1],
              handle.dofs[0][l + 1], dv, ldvs_h[0], ldvs_h[1], ldvs_h[0],
              prefix + "lwpk" + "_level_" + std::to_string(l), store, verify);
        }
      }
    } // end of loop
    // printf("output of decomposition\n");
    // print_matrix_cuda(handle.dofs[2][0], handle.dofs[1][0],
    // handle.dofs[0][0],
    //                     dv, ldvs[0], ldvs[1], ldvs[0]);
  }

  if (D >= 4) {

    for (int l = 0; l < l_target; ++l) {
      // printf("[gpu] l = %d\n", l);
      int stride = std::pow(2, l);
      int Cstride = stride * 2;
      int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
      int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);
      bool f_padding = handle.dofs[0][l] % 2 == 0;
      bool c_padding = handle.dofs[1][l] % 2 == 0;
      bool r_padding = handle.dofs[2][l] % 2 == 0;

      int curr_dim_r, curr_dim_c, curr_dim_f;
      int lddv1, lddv2;
      int lddw1, lddw2;
      int lddb1, lddb2;

      // printf("D_padded: %d\n", handle.D_padded);
      thrust::device_vector<int> shape(handle.D_padded);
      thrust::device_vector<int> shape_c(handle.D_padded);
      for (int d = 0; d < handle.D_padded; d++) {
        shape[d] = handle.dofs[d][l];
        shape_c[d] = handle.dofs[d][l + 1];
        // printf("%d %d\n", shape[d], shape_c[d]);
      }

      thrust::device_vector<int> unprocessed_dims;
      for (int i = 3; i < D; i++)
        unprocessed_dims.push_back(i);

      // printf("input: \n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[curr_dim_r][l],
      //   handle.dofs[curr_dim_c][l], handle.dofs[curr_dim_f][l],
      //                     dv+i*ldvs[0]*ldvs[1]*ldvs[2], ldvs[0], ldvs[1],
      //                     ldvs[0]);
      // }

      lwpk<T, D, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l], dv,
                       ldvs_d, handle.dw, handle.ldws_d, 0);
      lwpk<T, D, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l], dv,
                       ldvs_d, handle.db, handle.ldbs_d, 0);

      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   // print_matrix_cuda(handle.dofs[curr_dim_r][l],
      //   handle.dofs[curr_dim_c][l], handle.dofs[curr_dim_f][l],
      //   // handle.dw+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //   handle.ldws_h[0], handle.ldws_h[1], handle.ldws_h[0]);
      //   compare_matrix_cuda(handle.dofs[curr_dim_r][l],
      //   handle.dofs[curr_dim_c][l], handle.dofs[curr_dim_f][l],
      //                   dv+i*ldvs[0]*ldvs[1]*ldvs[2], ldvs[0], ldvs[1],
      //                   ldvs[0],
      //                   handle.dw+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                   handle.ldws_h[0], handle.ldws_h[1],
      //                   handle.ldws_h[0],false);
      // }

      // printf("ldvs: ");
      // for (int i = 0; i < D; i++) { std::cout << ldvs[i] << " ";}
      // printf("\n");

      // printf("ldws_h: ");
      // for (int i = 0; i < D; i++) { std::cout << handle.ldws_h[i] << " ";}
      // printf("\n");

      // printf("lddv: %d %d lddw: %d %d\n", lddv1, lddv2, lddw1, lddw2);

      // cudaMemset3DHelper(dv, ldvs[0]*sizeof(T), ldvs[0]*sizeof(T),
      //                   ldvs[1], 0, handle.dofs[0][l]*sizeof(T),
      //                   handle.dofs[1][l],
      //                  handle.dofs[2][l]*handle.dofs[3][l]);

      // printf("interpolate 1-3D\n");
      curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
      lddv1 = 1, lddv2 = 1, lddw1 = 1, lddw2 = 1;
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddv1 *= ldvs[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddv2 *= ldvs[s];
      }
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddw1 *= handle.ldws_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddw2 *= handle.ldws_h[s];
      }
      gpk_reo<T, D, 3, true, false, 1>(
          handle, handle.shapes_h[l], handle.shapes_d[l],
          handle.shapes_d[l + 1], handle.ldws_d, ldvs_d,
          unprocessed_dims.size(),
          thrust::raw_pointer_cast(unprocessed_dims.data()), curr_dim_r,
          curr_dim_c, curr_dim_f, handle.ratio[curr_dim_r][l],
          handle.ratio[curr_dim_c][l], handle.ratio[curr_dim_f][l], handle.dw,
          lddw1, lddw2, dv, lddv1, lddv2,
          // null, lddv1, lddv2,
          dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                     curr_dim_f, 0, 0,
                                     handle.dofs[curr_dim_f][l + 1])),
          lddv1, lddv2,
          // null, lddv1, lddv2,
          dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                     curr_dim_f, 0,
                                     handle.dofs[curr_dim_c][l + 1], 0)),
          lddv1, lddv2,
          // null, lddv1, lddv2,
          dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                     curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                     0, 0)),
          lddv1, lddv2,
          // null, lddv1, lddv2,
          dv + get_idx(ldvs,
                       gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                               curr_dim_f, 0, handle.dofs[curr_dim_c][l + 1],
                               handle.dofs[curr_dim_f][l + 1])),
          lddv1, lddv2,
          // null, lddv1, lddv2,
          dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                     curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                     0, handle.dofs[curr_dim_f][l + 1])),
          lddv1, lddv2,
          // null, lddv1, lddv2,
          dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                     curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                     handle.dofs[curr_dim_c][l + 1], 0)),
          lddv1, lddv2,
          // null, lddv1, lddv2,
          dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                     curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                     handle.dofs[curr_dim_c][l + 1],
                                     handle.dofs[curr_dim_f][l + 1])),
          lddv1, lddv2,
          // null, lddv1, lddv2,
          0, handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[curr_dim_r][l],
      //   handle.dofs[curr_dim_c][l], handle.dofs[curr_dim_f][l],
      //                     dv+i*ldvs[0]*ldvs[1]*ldvs[2], ldvs[0], ldvs[1],
      //                     ldvs[0]);
      // }

      lwpk<T, D, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l], dv,
                       ldvs_d, handle.dw, handle.ldws_d, 0);

      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[curr_dim_r][l],
      //   handle.dofs[curr_dim_c][l], handle.dofs[curr_dim_f][l],
      //                     handle.dw+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                     handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);
      // }

      // printf("interpolate 4-5D\n");
      curr_dim_f = 0, curr_dim_c = 3, curr_dim_r = 4;
      lddv1 = 1, lddv2 = 1, lddw1 = 1, lddw2 = 1;
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddv1 *= ldvs[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddv2 *= ldvs[s];
      }
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddw1 *= handle.ldws_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddw2 *= handle.ldws_h[s];
      }
      // printf("lddv1(%d), lddv2(%d), lddw1(%d), lddw2(%d)\n", lddv1, lddv2,
      // lddw1, lddw2);
      if (D % 2 == 0) {
        unprocessed_dims.pop_back();
        gpk_reo<T, D, 2, true, false, 2>(
            handle, handle.shapes_h[l], handle.shapes_d[l],
            handle.shapes_d[l + 1], handle.ldws_d, ldvs_d,
            unprocessed_dims.size(),
            thrust::raw_pointer_cast(unprocessed_dims.data()), curr_dim_r,
            curr_dim_c, curr_dim_f, handle.ratio[curr_dim_r][l],
            handle.ratio[curr_dim_c][l], handle.ratio[curr_dim_f][l], handle.dw,
            lddw1, lddw2, dv, lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f, 0, 0,
                                       handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f, 0,
                                       handle.dofs[curr_dim_c][l + 1], 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f,
                                       handle.dofs[curr_dim_r][l + 1], 0, 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, 0, handle.dofs[curr_dim_c][l + 1],
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1], 0,
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                 handle.dofs[curr_dim_c][l + 1], 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                 handle.dofs[curr_dim_c][l + 1],
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            0, handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
      } else {
        unprocessed_dims.pop_back();
        unprocessed_dims.pop_back();
        gpk_reo<T, D, 3, true, false, 2>(
            handle, handle.shapes_h[l], handle.shapes_d[l],
            handle.shapes_d[l + 1], handle.ldws_d, ldvs_d,
            unprocessed_dims.size(),
            thrust::raw_pointer_cast(unprocessed_dims.data()), curr_dim_r,
            curr_dim_c, curr_dim_f, handle.ratio[curr_dim_r][l],
            handle.ratio[curr_dim_c][l], handle.ratio[curr_dim_f][l], handle.dw,
            lddw1, lddw2, dv, lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f, 0, 0,
                                       handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f, 0,
                                       handle.dofs[curr_dim_c][l + 1], 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f,
                                       handle.dofs[curr_dim_r][l + 1], 0, 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, 0, handle.dofs[curr_dim_c][l + 1],
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1], 0,
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                 handle.dofs[curr_dim_c][l + 1], 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                 handle.dofs[curr_dim_c][l + 1],
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            0, handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
      }

      // printf("after interpolate 4D:\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[curr_dim_r][l],
      //   handle.dofs[curr_dim_c][l], handle.dofs[curr_dim_f][l],
      //                     dv+i*ldvs[0]*ldvs[1]*ldvs[2], ldvs[0], ldvs[1],
      //                     ldvs[0]);
      // }

      // printf("reorder 1-3D\n");
      curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
      lddw1 = 1, lddw2 = 1, lddb1 = 1, lddb2 = 1;
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddw1 *= handle.ldws_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddw2 *= handle.ldws_h[s];
      }
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddb1 *= handle.ldbs_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddb2 *= handle.ldbs_h[s];
      }
      for (int i = 3; i < D; i++)
        unprocessed_dims.push_back(i);

      gpk_reo<T, D, 3, false, false, 1>(
          handle, handle.shapes_h[l], handle.shapes_d[l],
          handle.shapes_d[l + 1], handle.ldbs_d, handle.ldws_d,
          unprocessed_dims.size(),
          thrust::raw_pointer_cast(unprocessed_dims.data()), curr_dim_r,
          curr_dim_c, curr_dim_f, handle.ratio[curr_dim_r][l],
          handle.ratio[curr_dim_c][l], handle.ratio[curr_dim_f][l], handle.db,
          lddb1, lddb2, handle.dw, lddw1, lddw2,
          // null, lddv1, lddv2,
          handle.dw +
              get_idx(handle.ldws_h, gen_idx(handle.D_padded, curr_dim_r,
                                             curr_dim_c, curr_dim_f, 0, 0,
                                             handle.dofs[curr_dim_f][l + 1])),
          lddw1, lddw2,
          // null, lddv1, lddv2,
          handle.dw + get_idx(handle.ldws_h,
                              gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                      curr_dim_f, 0,
                                      handle.dofs[curr_dim_c][l + 1], 0)),
          lddw1, lddw2,
          // null, lddv1, lddv2,
          handle.dw + get_idx(handle.ldws_h,
                              gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                      curr_dim_f,
                                      handle.dofs[curr_dim_r][l + 1], 0, 0)),
          lddw1, lddw2,
          // null, lddv1, lddv2,
          handle.dw +
              get_idx(handle.ldws_h,
                      gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                              curr_dim_f, 0, handle.dofs[curr_dim_c][l + 1],
                              handle.dofs[curr_dim_f][l + 1])),
          lddw1, lddw2,
          // null, lddv1, lddv2,
          handle.dw +
              get_idx(handle.ldws_h,
                      gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                              curr_dim_f, handle.dofs[curr_dim_r][l + 1], 0,
                              handle.dofs[curr_dim_f][l + 1])),
          lddw1, lddw2,
          // null, lddv1, lddv2,
          handle.dw +
              get_idx(handle.ldws_h,
                      gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                              curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                              handle.dofs[curr_dim_c][l + 1], 0)),
          lddw1, lddw2,
          // null, lddv1, lddv2,
          handle.dw +
              get_idx(handle.ldws_h,
                      gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                              curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                              handle.dofs[curr_dim_c][l + 1],
                              handle.dofs[curr_dim_f][l + 1])),
          lddw1, lddw2,
          // null, lddv1, lddv2,
          0, handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

      // printf("dv before calc\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[curr_dim_r][l],
      //   handle.dofs[curr_dim_c][l], handle.dofs[curr_dim_f][l],
      //                     dv+i*ldvs[0]*ldvs[1]*ldvs[2], ldvs[0], ldvs[1],
      //                     ldvs[0]);
      // }

      lwpk<T, D, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l],
                       handle.dw, handle.ldws_d, handle.db, handle.ldbs_d, 0);

      // printf("db before calc\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[curr_dim_r][l],
      //   handle.dofs[curr_dim_c][l], handle.dofs[curr_dim_f][l],
      //                     handle.db+i*handle.ldbs_h[0]*handle.ldbs_h[1]*handle.ldbs_h[2],
      //                     handle.ldbs_h[0], handle.ldbs_h[1],
      //                     handle.ldbs_h[0]);
      // }

      // printf("calc coeff 1-5D\n");
      curr_dim_f = 0, curr_dim_c = 3, curr_dim_r = 4;
      lddv1 = 1, lddv2 = 1, lddb1 = 1, lddb2 = 1;
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddv1 *= ldvs[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddv2 *= ldvs[s];
      }
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddb1 *= handle.ldbs_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddb2 *= handle.ldbs_h[s];
      }
      if (D % 2 == 0) {
        unprocessed_dims.pop_back();
        gpk_reo<T, D, 2, false, true, 2>(
            handle, handle.shapes_h[l], handle.shapes_d[l],
            handle.shapes_d[l + 1], handle.ldbs_d, ldvs_d,
            unprocessed_dims.size(),
            thrust::raw_pointer_cast(unprocessed_dims.data()), curr_dim_r,
            curr_dim_c, curr_dim_f, handle.ratio[curr_dim_r][l],
            handle.ratio[curr_dim_c][l], handle.ratio[curr_dim_f][l], handle.db,
            lddb1, lddb2, dv, lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f, 0, 0,
                                       handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f, 0,
                                       handle.dofs[curr_dim_c][l + 1], 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f,
                                       handle.dofs[curr_dim_r][l + 1], 0, 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, 0, handle.dofs[curr_dim_c][l + 1],
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1], 0,
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                 handle.dofs[curr_dim_c][l + 1], 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                 handle.dofs[curr_dim_c][l + 1],
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            0, handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

      } else {
        unprocessed_dims.pop_back();
        unprocessed_dims.pop_back();
        gpk_reo<T, D, 3, false, true, 2>(
            handle, handle.shapes_h[l], handle.shapes_d[l],
            handle.shapes_d[l + 1], handle.ldbs_d, ldvs_d,
            unprocessed_dims.size(),
            thrust::raw_pointer_cast(unprocessed_dims.data()), curr_dim_r,
            curr_dim_c, curr_dim_f, handle.ratio[curr_dim_r][l],
            handle.ratio[curr_dim_c][l], handle.ratio[curr_dim_f][l], handle.db,
            lddb1, lddb2, dv, lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f, 0, 0,
                                       handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f, 0,
                                       handle.dofs[curr_dim_c][l + 1], 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f,
                                       handle.dofs[curr_dim_r][l + 1], 0, 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, 0, handle.dofs[curr_dim_c][l + 1],
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1], 0,
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                 handle.dofs[curr_dim_c][l + 1], 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                 handle.dofs[curr_dim_c][l + 1],
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            0, handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
      }
      // printf("after calc coeff 4D\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      // // for (int i = 0; i < 1; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[curr_dim_r][l],
      //   handle.dofs[curr_dim_c][l], handle.dofs[curr_dim_f][l],
      //                     dv+i*ldvs[0]*ldvs[1]*ldvs[2], ldvs[0], ldvs[1],
      //                     ldvs[0]);
      // }

      // start correction calculation
      int prev_dim_r, prev_dim_c, prev_dim_f;
      curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
      T *dw_out = handle.dw;
      T *dw_in1 = dv;
      T *dw_in2 =
          dv + get_idx(ldvs, gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f, 0,
                                     0, handle.dofs[curr_dim_f][l + 1]));
      // printf("mass trans 1D\n");
      curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
      prev_dim_f = curr_dim_f;
      prev_dim_c = curr_dim_c;
      prev_dim_r = curr_dim_r;
      lddv1 = 1, lddv2 = 1, lddw1 = 1, lddw2 = 1;
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddv1 *= ldvs[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddv2 *= ldvs[s];
      }
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddw1 *= handle.ldws_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddw2 *= handle.ldws_h[s];
      }
      thrust::device_vector<int> processed_dims;
      lpk_reo_1<T, D>(
          handle, handle.shapes_h[l], handle.shapes_h[l + 1],
          handle.shapes_d[l], handle.shapes_d[l + 1], ldvs_d, handle.ldws_d,
          handle.processed_n[0], handle.processed_dims_h[0],
          handle.processed_dims_d[0], curr_dim_r, curr_dim_c, curr_dim_f,
          handle.dist[curr_dim_f][l], handle.ratio[curr_dim_f][l], dw_in1,
          lddv1, lddv2,
          // dv+get_idx(ldvs, gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f, 0,
          // 0, handle.dofs[0][l+1])),
          dw_in2, lddv1, lddv2, dw_out, lddw1, lddw2, 0,
          handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

      // printf("after mass_trans_1\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      //   handle.dofs[0][l+1],
      //                     dw_out+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                     handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);
      // }

      // printf("solve tridiag 1D\n");
      // ipk_1<T, D>(handle, shape, shape_c, handle.ldws_h, handle.ldws_h,
      // processed_dims, curr_dim_r, curr_dim_c, curr_dim_f,
      // handle.am[curr_dim_f][l+1], handle.bm[curr_dim_f][l+1],
      // handle.dist[curr_dim_f][l+1], dw_out, lddw1, lddw2, 0,
      // handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);
      ipk_1<T, D>(
          handle, handle.shapes_h[l], handle.shapes_h[l + 1],
          handle.shapes_d[l], handle.shapes_d[l + 1], handle.ldws_d,
          handle.ldws_d, handle.processed_n[0], handle.processed_dims_h[0],
          handle.processed_dims_d[0], curr_dim_r, curr_dim_c, curr_dim_f,
          handle.am[curr_dim_f][l + 1], handle.bm[curr_dim_f][l + 1],
          handle.dist[curr_dim_f][l + 1], dw_out, lddw1, lddw2, 0,
          handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);
      processed_dims.push_back(curr_dim_f);
      // printf("after solve_tridiag_1\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      //   handle.dofs[0][l+1],
      //                     dw_out+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                     handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);
      // }

      // mass trans 2D
      curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
      dw_in1 = dw_out;
      dw_in2 = dw_out + get_idx(handle.ldws_h,
                                gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f,
                                        0, handle.dofs[curr_dim_c][l + 1], 0));
      dw_out +=
          get_idx(handle.ldws_h, gen_idx(D, prev_dim_r, prev_dim_c, prev_dim_f,
                                         0, 0, handle.dofs[prev_dim_f][l + 1]));
      prev_dim_f = curr_dim_f;
      prev_dim_c = curr_dim_c;
      prev_dim_r = curr_dim_r;
      lddv1 = 1, lddv2 = 1, lddw1 = 1, lddw2 = 1;
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddw1 *= handle.ldws_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddw2 *= handle.ldws_h[s];
      }
      // printf("mass trans 2D\n");
      lpk_reo_2<T, D>(
          handle, handle.shapes_h[l], handle.shapes_h[l + 1],
          handle.shapes_d[l], handle.shapes_d[l + 1], handle.ldws_d,
          handle.ldws_d, handle.processed_n[1], handle.processed_dims_h[1],
          handle.processed_dims_d[1], curr_dim_r, curr_dim_c, curr_dim_f,
          handle.dist[curr_dim_c][l], handle.ratio[curr_dim_c][l],
          // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r, curr_dim_c,
          // curr_dim_f, 0, 0, 0)),
          dw_in1, lddw1, lddw2,
          // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r, curr_dim_c,
          // curr_dim_f, 0, handle.dofs[1][l+1], 0)),
          dw_in2, lddw1, lddw2,
          // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r, curr_dim_c,
          // curr_dim_f, 0, 0, handle.dofs[0][l+1])),
          dw_out, lddw1, lddw2, 0,
          handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

      // printf("after mass_trans_2\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l+1],
      //   handle.dofs[0][l+1],
      //                     dw_out+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                     handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);
      // }
      // printf("solve tridiag 2D\n");
      ipk_2<T, D>(
          handle, handle.shapes_h[l], handle.shapes_h[l + 1],
          handle.shapes_d[l], handle.shapes_d[l + 1], handle.ldws_d,
          handle.ldws_d, handle.processed_n[1], handle.processed_dims_h[1],
          handle.processed_dims_d[1], curr_dim_r, curr_dim_c, curr_dim_f,
          handle.am[curr_dim_c][l + 1], handle.bm[curr_dim_c][l + 1],
          handle.dist[curr_dim_c][l + 1], dw_out, lddw1, lddw2, 0,
          handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);
      processed_dims.push_back(curr_dim_c);
      // printf("after solve_tridiag_2\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l+1],
      //   handle.dofs[0][l+1],
      //                     dw_out+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                     handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);
      // }

      // mass trans 3D

      curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
      dw_in1 = dw_out;
      dw_in2 = dw_out + get_idx(handle.ldws_h,
                                gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f,
                                        handle.dofs[curr_dim_r][l + 1], 0, 0));
      dw_out +=
          get_idx(handle.ldws_h, gen_idx(D, prev_dim_r, prev_dim_c, prev_dim_f,
                                         0, handle.dofs[prev_dim_c][l + 1], 0));
      prev_dim_f = curr_dim_f;
      prev_dim_c = curr_dim_c;
      prev_dim_r = curr_dim_r;
      lddv1 = 1, lddv2 = 1, lddw1 = 1, lddw2 = 1;
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddw1 *= handle.ldws_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddw2 *= handle.ldws_h[s];
      }
      // printf("mass trans 3D\n");
      lpk_reo_3<T, D>(
          handle, handle.shapes_h[l], handle.shapes_h[l + 1],
          handle.shapes_d[l], handle.shapes_d[l + 1], handle.ldws_d,
          handle.ldws_d, handle.processed_n[2], handle.processed_dims_h[2],
          handle.processed_dims_d[2], curr_dim_r, curr_dim_c, curr_dim_f,
          handle.dist[curr_dim_r][l], handle.ratio[curr_dim_r][l],
          // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r, curr_dim_c,
          // curr_dim_f, 0, 0, handle.dofs[0][l+1])),
          dw_in1, lddw1, lddw2,
          // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r, curr_dim_c,
          // curr_dim_f, handle.dofs[2][l+1], 0, handle.dofs[0][l+1])),
          dw_in2, lddw1, lddw2,
          // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r, curr_dim_c,
          // curr_dim_f, 0, handle.dofs[1][l+1], handle.dofs[0][l+1])),
          dw_out, lddw1, lddw2, 0,
          handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

      // printf("after mass_trans_3\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[2][l+1], handle.dofs[1][l+1],
      //   handle.dofs[0][l+1],
      //                     dw_out+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                     handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);
      // }
      // printf("solve tridiag 3D\n");
      ipk_3<T, D>(
          handle, shape, shape_c, handle.ldws_h, handle.ldws_h, processed_dims,
          curr_dim_r, curr_dim_c, curr_dim_f, handle.am[curr_dim_r][l + 1],
          handle.bm[curr_dim_r][l + 1], handle.dist[curr_dim_r][l + 1], dw_out,
          lddw1, lddw2, 0,
          handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);
      processed_dims.push_back(curr_dim_r);

      // printf("after solve_tridiag_3\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[2][l+1], handle.dofs[1][l+1],
      //   handle.dofs[0][l+1],
      //                     dw_out+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                     handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);
      // }

      // mass trans 4D+
      for (int i = 3; i < D; i++) {
        curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = i;
        dw_in1 = dw_out;
        dw_in2 =
            dw_out +
            get_idx(handle.ldws_h,
                    gen_idx(handle.D_padded, curr_dim_r, curr_dim_c, curr_dim_f,
                            handle.dofs[curr_dim_r][l + 1], 0, 0));
        dw_out +=
            get_idx(handle.ldws_h,
                    gen_idx(handle.D_padded, prev_dim_r, prev_dim_c, prev_dim_f,
                            handle.dofs[prev_dim_r][l + 1], 0, 0));
        prev_dim_f = curr_dim_f;
        prev_dim_c = curr_dim_c;
        prev_dim_r = curr_dim_r;
        lddv1 = 1, lddv2 = 1, lddw1 = 1, lddw2 = 1;
        for (int s = curr_dim_f; s < curr_dim_c; s++) {
          lddw1 *= handle.ldws_h[s];
        }
        for (int s = curr_dim_c; s < curr_dim_r; s++) {
          lddw2 *= handle.ldws_h[s];
        }
        // printf("mass trans %dD\n", i+1);
        lpk_reo_3<T, D>(
            handle, handle.shapes_h[l], handle.shapes_h[l + 1],
            handle.shapes_d[l], handle.shapes_d[l + 1], handle.ldws_d,
            handle.ldws_d, handle.processed_n[i], handle.processed_dims_h[i],
            handle.processed_dims_d[i], curr_dim_r, curr_dim_c, curr_dim_f,
            handle.dist[curr_dim_r][l], handle.ratio[curr_dim_r][l],
            // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r,
            // curr_dim_c, curr_dim_f, 0, 0, handle.dofs[0][l+1])),
            dw_in1, lddw1, lddw2,
            // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r,
            // curr_dim_c, curr_dim_f, handle.dofs[2][l+1], 0,
            // handle.dofs[0][l+1])),
            dw_in2, lddw1, lddw2, dw_out, lddw1, lddw2, 0,
            handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

        // printf("after mass_trans_4\n");
        // for (int i = 0; i < handle.dofs[3][l+1]; i++) {
        //   printf("i = %d\n", i);
        //   print_matrix_cuda(handle.dofs[2][l+1], handle.dofs[1][l+1],
        //   handle.dofs[0][l+1],
        //                     dw_out+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
        //                     handle.ldws_h[0], handle.ldws_h[1],
        //                     handle.ldws_h[0]);
        // }
        // printf("solve tridiag %dD\n", i+1);
        ipk_3<T, D>(
            handle, handle.shapes_h[l], handle.shapes_h[l + 1],
            handle.shapes_d[l], handle.shapes_d[l + 1], handle.ldws_d,
            handle.ldws_d, handle.processed_n[i], handle.processed_dims_h[i],
            handle.processed_dims_d[i], curr_dim_r, curr_dim_c, curr_dim_f,
            handle.am[curr_dim_r][l + 1], handle.bm[curr_dim_r][l + 1],
            handle.dist[curr_dim_r][l + 1], dw_out, lddw1, lddw2, 0,
            handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);
        processed_dims.push_back(i);
      }

      // printf("after solve_tridiag_4\n");
      // for (int i = 0; i < handle.dofs[3][l+1]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[2][l+1], handle.dofs[1][l+1],
      //   handle.dofs[0][l+1],
      //                     dw_out+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                     handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);
      // }

      // apply correction
      lwpk<T, D, ADD>(handle, handle.shapes_h[l + 1], handle.shapes_d[l + 1],
                      dw_out, handle.ldws_d, dv, ldvs_d, 0);

      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[curr_dim_r][l],
      //   handle.dofs[curr_dim_c][l], handle.dofs[curr_dim_f][l],
      //                     dv+i*ldvs[0]*ldvs[1]*ldvs[2], ldvs[0], ldvs[1],
      //                     ldvs[0]);
      // }
    }
  }
}

template <typename T, uint32_t D>
void recompose_reo(Handle<T, D> &handle, T *dv, std::vector<int> ldvs,
                   int l_target) {

  // l_end=handle.l_target-4;

  int *ldvs_h = new int[handle.D_padded];
  for (int d = 0; d < handle.D_padded; d++) {
    ldvs_h[d] = ldvs[d];
  }
  int *ldvs_d;
  cudaMallocHelper((void **)&ldvs_d, handle.D_padded * sizeof(int));
  cudaMemcpyAsyncHelper(handle, ldvs_d, ldvs_h, handle.D_padded * sizeof(int),
                        H2D, 0);

  if (D <= 3) {

    // printf("intput of recomposition\n");
    // print_matrix_cuda(handle.dofs[2][0], handle.dofs[1][0],
    // handle.dofs[0][0],
    //                     dv, ldvs[0], ldvs[1], ldvs[0]);

    std::string prefix = "recomp_";
    if (sizeof(T) == sizeof(double))
      prefix += "d_";
    if (sizeof(T) == sizeof(float))
      prefix += "f_";
    for (int d = 0; d < D; d++)
      prefix += std::to_string(handle.shapes_h[0][d]) + "_";
    // std::cout << prefix << std::endl;

    for (int l = l_target - 1; l >= 0; l--) {
      // printf("[gpu] l = %d\n", l);
      int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
      int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);

      bool f_padding = handle.dofs[0][l] % 2 == 0;
      bool c_padding = handle.dofs[1][l] % 2 == 0;
      bool r_padding = handle.dofs[0][l] % 2 == 0;

      // printf("input v\n");
      // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      // handle.dofs[0][l],
      //                   dv, ldvs[0], ldvs[1], ldvs[0]);

      int curr_dim_r = 2;
      int curr_dim_c = 1;
      int curr_dim_f = 0;
      int lddv1, lddv2;
      int lddw1, lddw2;
      int lddb1, lddb2;

      thrust::device_vector<int> shape(handle.D_padded);
      thrust::device_vector<int> shape_c(handle.D_padded);
      for (int d = 0; d < handle.D_padded; d++) {
        shape[d] = handle.dofs[d][l];
        shape_c[d] = handle.dofs[d][l + 1];
      }
      thrust::device_vector<int> unprocessed_dims(1);
      unprocessed_dims[0] = 3;

      thrust::device_vector<int> processed_dims(0);
      if (D >= 1) {
        // lpk_reo_1<T, D>(handle,
        //                 handle.shapes_h[l], handle.shapes_h[l+1],
        //                 handle.shapes_d[l], handle.shapes_d[l+1],
        //                 ldvs_d, handle.ldws_d,
        //                 handle.processed_n[0], handle.processed_dims_h[0],
        //                 handle.processed_dims_d[0], 2, 1, 0,
        //                 handle.dist[0][l], handle.ratio[0][l],
        //                 dv, ldvs_h[0], ldvs_h[1],
        //                 dv+get_idx(ldvs_h[0], ldvs_h[1], 0, 0,
        //                 handle.dofs[0][l+1]), ldvs_h[0], ldvs_h[1],
        //                 handle.dw, handle.ldws_h[0], handle.ldws_h[1],
        //                 0,
        //                 handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);
        lpk_reo_1_3d(
            handle, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
            handle.dofs[0][l + 1], handle.dofs[2][l + 1], handle.dofs[1][l + 1],
            handle.dofs[0][l + 1], handle.dist[0][l], handle.ratio[0][l], dv,
            ldvs_h[0], ldvs_h[1],
            dv + get_idx(ldvs_h[0], ldvs_h[1], 0, 0, handle.dofs[0][l + 1]),
            ldvs_h[0], ldvs_h[1], handle.dw, handle.ldws_h[0], handle.ldws_h[1],
            0,
            handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

        // handle.sync(0);
        verify_matrix_cuda(
            handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l + 1],
            handle.dw, handle.ldws_h[0], handle.ldws_h[1], handle.ldws_h[0],
            prefix + "lpk_reo_1_3d" + "_level_" + std::to_string(l), store,
            verify);

        processed_dims.push_back(0);

        // printf("after mass_trans_multiply_1_cpt\n");
        // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
        // handle.dofs[0][l+1],
        //                 handle.dw, handle.ldws_h[0], handle.ldws_h[1]
        //                 ,handle.ldws_h[0]);

        // ipk_1<T, D>(handle,
        //             handle.shapes_h[l], handle.shapes_h[l+1],
        //             handle.shapes_d[l], handle.shapes_d[l+1],
        //             handle.ldws_d, handle.ldws_d,
        //             handle.processed_n[0], handle.processed_dims_h[0],
        //             handle.processed_dims_d[0], 2, 1, 0, handle.am[0][l+1],
        //             handle.bm[0][l+1], handle.dist[0][l+1], handle.dw,
        //             handle.ldws_h[0], handle.ldws_h[1], 0,
        //             handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);
        ipk_1_3d(
            handle, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l + 1],
            handle.am[0][l + 1], handle.bm[0][l + 1], handle.dist[0][l + 1],
            handle.dw, handle.ldws_h[0], handle.ldws_h[1], 0,
            handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);

        // handle.sync(0);
        verify_matrix_cuda(
            handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l + 1],
            handle.dw, handle.ldws_h[0], handle.ldws_h[1], handle.ldws_h[0],
            prefix + "ipk_1_3d" + "_level_" + std::to_string(l), store, verify);

        // printf("after solve_tridiag_1_cpt\n");
        // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
        // handle.dofs[0][l+1],
        //                 handle.dw, handle.ldws_h[0], handle.ldws_h[1]
        //                 ,handle.ldws_h[0]);
        if (D == 1) {
          lwpk<T, D, SUBTRACT>(handle, handle.shapes_h[l + 1],
                               handle.shapes_d[l + 1], handle.dw, handle.ldws_d,
                               dv, ldvs_d, 0);
        }
      }

      if (D >= 2) {
        // lpk_reo_2<T, D>(handle,
        //                 handle.shapes_h[l], handle.shapes_h[l+1],
        //                 handle.shapes_d[l], handle.shapes_d[l+1],
        //                 handle.ldws_d, handle.ldws_d,
        //                 handle.processed_n[1], handle.processed_dims_h[1],
        //                 handle.processed_dims_d[1], 2, 1, 0,
        //                 handle.dist[1][l], handle.ratio[1][l],
        //                 handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1],
        //                 0, 0, 0), handle.ldws_h[0], handle.ldws_h[1],
        //                 handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1],
        //                 0, handle.dofs[1][l+1], 0), handle.ldws_h[0],
        //                 handle.ldws_h[1], handle.dw+get_idx(handle.ldws_h[0],
        //                 handle.ldws_h[1], 0, 0, handle.dofs[0][l+1]),
        //                 handle.ldws_h[0], handle.ldws_h[1], 0,
        //                 handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);
        lpk_reo_2_3d(
            handle, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l + 1],
            handle.dofs[1][l + 1], handle.dist[1][l], handle.ratio[1][l],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0, 0, 0),
            handle.ldws_h[0], handle.ldws_h[1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
                                handle.dofs[1][l + 1], 0),
            handle.ldws_h[0], handle.ldws_h[1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0, 0,
                                handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], 0,
            handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

        // handle.sync(0);
        verify_matrix_cuda(
            handle.dofs[2][l], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0, 0,
                                handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], handle.ldws_h[0],
            prefix + "lpk_reo_2_3d" + "_level_" + std::to_string(l), store,
            verify);

        // printf("after mass_trans_multiply_2_cpt\n");
        // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l+1],
        // handle.dofs[0][l+1],
        //                 handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1],
        //                 0, 0, handle.dofs[0][l+1]), handle.ldws_h[0],
        //                 handle.ldws_h[1] ,handle.ldws_h[0]);

        // ipk_2<T, D>(handle,
        //             handle.shapes_h[l], handle.shapes_h[l+1],
        //             handle.shapes_d[l], handle.shapes_d[l+1],
        //             handle.ldws_d, handle.ldws_d,
        //             handle.processed_n[1], handle.processed_dims_h[1],
        //             handle.processed_dims_d[1], 2, 1, 0, handle.am[1][l+1],
        //             handle.bm[1][l+1], handle.dist[1][l+1],
        //             handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
        //             0, handle.dofs[0][l+1]), handle.ldws_h[0],
        //             handle.ldws_h[1], 0,
        //             handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);
        ipk_2_3d(
            handle, handle.dofs[2][l], handle.dofs[1][l + 1],
            handle.dofs[0][l + 1], handle.am[1][l + 1], handle.bm[1][l + 1],
            handle.dist[1][l + 1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0, 0,
                                handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], 0,
            handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);

        // handle.sync(0);
        verify_matrix_cuda(
            handle.dofs[2][l], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0, 0,
                                handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], handle.ldws_h[0],
            prefix + "ipk_2_3d" + "_level_" + std::to_string(l), store, verify);

        // printf("after solve_tridiag_2_cpt\n");
        // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l+1],
        // handle.dofs[0][l+1],
        //                 handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1],
        //                 0, 0, handle.dofs[0][l+1]), handle.ldws_h[0],
        //                 handle.ldws_h[1] ,handle.ldws_h[0]);

        // printf("before sub\n");
        // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
        // handle.dofs[0][l],
        //                   dv, ldvs[0], ldvs[1], ldvs[0]);

        if (D == 2) {
          lwpk<T, D, SUBTRACT>(
              handle, handle.shapes_h[l + 1], handle.shapes_d[l + 1],
              handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0, 0,
                                  handle.dofs[0][l + 1]),
              handle.ldws_d, dv, ldvs_d, 0);

          // printf("after sub\n");
          // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
          // handle.dofs[0][l],
          //                   dv, ldvs[0], ldvs[1], ldvs[0]);
        }
      }

      if (D == 3) {
        processed_dims.push_back(1);
        // lpk_reo_3<T, D>(handle,
        //                 handle.shapes_h[l], handle.shapes_h[l+1],
        //                 handle.shapes_d[l], handle.shapes_d[l+1],
        //                 handle.ldws_d, handle.ldws_d,
        //                 handle.processed_n[2], handle.processed_dims_h[2],
        //                 handle.processed_dims_d[2], 2, 1, 0,
        //                 handle.dist[2][l], handle.ratio[2][l],
        //                 handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1],
        //                 0, 0, handle.dofs[0][l+1]), handle.ldws_h[0],
        //                 handle.ldws_h[1], handle.dw+get_idx(handle.ldws_h[0],
        //                 handle.ldws_h[1], handle.dofs[2][l+1], 0,
        //                 handle.dofs[0][l+1]), handle.ldws_h[0],
        //                 handle.ldws_h[1], handle.dw+get_idx(handle.ldws_h[0],
        //                 handle.ldws_h[1], 0, handle.dofs[1][l+1],
        //                 handle.dofs[0][l+1]), handle.ldws_h[0],
        //                 handle.ldws_h[1], 0,
        //                 handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);
        lpk_reo_3_3d(
            handle, handle.dofs[2][l], handle.dofs[1][l + 1],
            handle.dofs[0][l + 1], handle.dofs[2][l + 1], handle.dist[2][l],
            handle.ratio[2][l],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0, 0,
                                handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1],
                                handle.dofs[2][l + 1], 0,
                                handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
                                handle.dofs[1][l + 1], handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], 0,
            handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

        // handle.sync(0);
        verify_matrix_cuda(
            handle.dofs[2][l + 1], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
                                handle.dofs[1][l + 1], handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], handle.ldws_h[0],
            prefix + "lpk_reo_3_3d" + "_level_" + std::to_string(l), store,
            verify);

        // printf("after mass_trans_multiply_3_cpt\n");
        // print_matrix_cuda(handle.dofs[2][l+1], handle.dofs[1][l+1],
        // handle.dofs[0][l+1],
        //                 handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1],
        //                 0, handle.dofs[1][l+1], handle.dofs[0][l+1]),
        //                 handle.ldws_h[0], handle.ldws_h[1],handle.ldws_h[0]);

        // ipk_3<T, D>(handle,
        //             handle.shapes_h[l], handle.shapes_h[l+1],
        //             handle.shapes_d[l], handle.shapes_d[l+1],
        //             handle.ldws_d, handle.ldws_d,
        //             handle.processed_n[2], handle.processed_dims_h[2],
        //             handle.processed_dims_d[2], 2, 1, 0, handle.am[2][l+1],
        //             handle.bm[2][l+1], handle.dist[2][l+1],
        //             handle.dw+get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
        //             handle.dofs[1][l+1], handle.dofs[0][l+1]),
        //             handle.ldws_h[0], handle.ldws_h[1], 0,
        //             handle.auto_tuning_ts3[handle.arch][handle.precision][range_lp1]);
        ipk_3_3d(
            handle, handle.dofs[2][l + 1], handle.dofs[1][l + 1],
            handle.dofs[0][l + 1], handle.am[2][l + 1], handle.bm[2][l + 1],
            handle.dist[2][l + 1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
                                handle.dofs[1][l + 1], handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], 0,
            handle.auto_tuning_ts3[handle.arch][handle.precision][range_lp1]);

        // handle.sync(0);
        verify_matrix_cuda(
            handle.dofs[2][l + 1], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
            handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
                                handle.dofs[1][l + 1], handle.dofs[0][l + 1]),
            handle.ldws_h[0], handle.ldws_h[1], handle.ldws_h[0],
            prefix + "ipk_3_3d" + "_level_" + std::to_string(l), store, verify);

        // printf("after solve_tridiag_3_cpt\n");
        // print_matrix_cuda(handle.dofs[2][l+1], handle.dofs[1][l+1],
        // handle.dofs[0][l+1],
        //                   handle.dw+get_idx(handle.ldws_h[0],
        //                   handle.ldws_h[1], 0, handle.dofs[1][l+1],
        //                   handle.dofs[0][l+1]), handle.ldws_h[0],
        //                   handle.ldws_h[1],handle.ldws_h[0]);

        if (D == 3) {
          lwpk<T, D, SUBTRACT>(
              handle, handle.shapes_h[l + 1], handle.shapes_d[l + 1],
              handle.dw + get_idx(handle.ldws_h[0], handle.ldws_h[1], 0,
                                  handle.dofs[1][l + 1], handle.dofs[0][l + 1]),
              handle.ldws_d, dv, ldvs_d, 0);

          // handle.sync(0);
          verify_matrix_cuda(
              handle.dofs[2][l + 1], handle.dofs[1][l + 1],
              handle.dofs[0][l + 1], dv, ldvs_h[0], ldvs_h[1], ldvs_h[0],
              prefix + "lwpk" + "_level_" + std::to_string(l), store, verify);
        }
      }

      // printf("before prolongate_reo\n");
      // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      // handle.dofs[0][l],
      //                     dv, ldvs[0], ldvs[1], ldvs[0]);
      T *null = NULL;

      // gpk_rev<T, D, D, true, true, 1>(handle,
      //             handle.shapes_h[l], handle.shapes_d[l],
      //             handle.shapes_d[l+1], handle.ldws_d, ldvs_d,
      //             unprocessed_dims.size(),
      //             thrust::raw_pointer_cast(unprocessed_dims.data()), 2, 1, 0,
      //             handle.ratio[2][l], handle.ratio[1][l], handle.ratio[0][l],
      //             handle.dw, handle.ldws_h[0], handle.ldws_h[1],
      //             dv, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  0, 0, handle.dofs[0][l+1]),
      //             ldvs[0], ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  0, handle.dofs[1][l+1], 0),
      //             ldvs[0], ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  handle.dofs[2][l+1], 0, 0),
      //             ldvs[0], ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  0, handle.dofs[1][l+1],
      //             handle.dofs[0][l+1]), ldvs[0], ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  handle.dofs[2][l+1], 0,
      //             handle.dofs[0][l+1]), ldvs[0], ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  handle.dofs[2][l+1],
      //             handle.dofs[1][l+1], 0), ldvs[0], ldvs[1],
      //             // null,ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  handle.dofs[2][l+1],
      //             handle.dofs[1][l+1], handle.dofs[0][l+1]), ldvs[0],
      //             ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             0, 0, 0, handle.dofs[2][l], handle.dofs[1][l],
      //             handle.dofs[0][l], 0,
      //             handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

      gpk_rev_3d(
          handle, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
          handle.ratio[2][l], handle.ratio[1][l], handle.ratio[0][l], handle.dw,
          handle.ldws_h[0], handle.ldws_h[1], dv, ldvs[0], ldvs[1],
          dv + get_idx(ldvs[0], ldvs[1], 0, 0, handle.dofs[0][l + 1]), ldvs[0],
          ldvs[1],
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs[0], ldvs[1], 0, handle.dofs[1][l + 1], 0), ldvs[0],
          ldvs[1],
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs[0], ldvs[1], handle.dofs[2][l + 1], 0, 0), ldvs[0],
          ldvs[1],
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs[0], ldvs[1], 0, handle.dofs[1][l + 1],
                       handle.dofs[0][l + 1]),
          ldvs[0], ldvs[1],
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs[0], ldvs[1], handle.dofs[2][l + 1], 0,
                       handle.dofs[0][l + 1]),
          ldvs[0], ldvs[1],
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs[0], ldvs[1], handle.dofs[2][l + 1],
                       handle.dofs[1][l + 1], 0),
          ldvs[0], ldvs[1],
          // null,ldvs[0], ldvs[1],
          dv + get_idx(ldvs[0], ldvs[1], handle.dofs[2][l + 1],
                       handle.dofs[1][l + 1], handle.dofs[0][l + 1]),
          ldvs[0], ldvs[1],
          // null, ldvs[0], ldvs[1],
          0, 0, 0, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l], 0,
          handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

      // handle.sync(0);
      verify_matrix_cuda(
          handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l], handle.dw,
          handle.ldws_h[0], handle.ldws_h[1], handle.ldws_h[0],
          prefix + "gpk_rev_3d" + "_level_" + std::to_string(l), store, verify);

      // gpk_rev<T, D, D, true, false, 1>(handle,
      //             shape, shape_c, handle.ldws_h, ldvs, unprocessed_dims,
      //             2, 1, 0,
      //             handle.ratio[2][l], handle.ratio[1][l], handle.ratio[0][l],
      //             handle.dw, handle.ldws_h[0], handle.ldws_h[1],
      //             dv, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  0, 0, handle.dofs[0][l+1]),
      //             ldvs[0], ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  0, handle.dofs[1][l+1], 0),
      //             ldvs[0], ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  handle.dofs[2][l+1], 0, 0),
      //             ldvs[0], ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  0, handle.dofs[1][l+1],
      //             handle.dofs[0][l+1]), ldvs[0], ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  handle.dofs[2][l+1], 0,
      //             handle.dofs[0][l+1]), ldvs[0], ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  handle.dofs[2][l+1],
      //             handle.dofs[1][l+1], 0), ldvs[0], ldvs[1],
      //             // null,ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  handle.dofs[2][l+1],
      //             handle.dofs[1][l+1], handle.dofs[0][l+1]), ldvs[0],
      //             ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             0, 0, 0, handle.dofs[2][l], handle.dofs[1][l],
      //             handle.dofs[0][l], 0,
      //             handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

      // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      // handle.dofs[0][l],
      //                     handle.dw, handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);

      // gpk_rev<T, D, D, false, true, 1>(handle,
      //             shape, shape_c, handle.ldws_h, ldvs, unprocessed_dims,
      //             2, 1, 0,
      //             handle.ratio[2][l], handle.ratio[1][l], handle.ratio[0][l],
      //             handle.dw, handle.ldws_h[0], handle.ldws_h[1],
      //             dv, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  0, 0, handle.dofs[0][l+1]),
      //             ldvs[0], ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  0, handle.dofs[1][l+1], 0),
      //             ldvs[0], ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  handle.dofs[2][l+1], 0, 0),
      //             ldvs[0], ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  0, handle.dofs[1][l+1],
      //             handle.dofs[0][l+1]), ldvs[0], ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  handle.dofs[2][l+1], 0,
      //             handle.dofs[0][l+1]), ldvs[0], ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  handle.dofs[2][l+1],
      //             handle.dofs[1][l+1], 0), ldvs[0], ldvs[1],
      //             // null,ldvs[0], ldvs[1],
      //             dv+get_idx(ldvs[0], ldvs[1],  handle.dofs[2][l+1],
      //             handle.dofs[1][l+1], handle.dofs[0][l+1]), ldvs[0],
      //             ldvs[1],
      //             // null, ldvs[0], ldvs[1],
      //             0, 0, 0, handle.dofs[2][l], handle.dofs[1][l],
      //             handle.dofs[0][l], 0,
      //             handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

      // printf("after prolongate_reo\n");
      // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      // handle.dofs[0][l],
      //                     handle.dw, handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);

      lwpk<T, D, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l],
                       handle.dw, handle.ldws_d, dv, ldvs_d, 0);

      // printf("output\n");
      // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      // handle.dofs[0][l],
      //                     dv, ldvs[0], ldvs[1], ldvs[0]);
    }
  }
  if (D >= 4) {
    for (int l = l_target - 1; l >= 0; l--) {
      // printf("[gpu] l = %d\n", l);
      int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
      int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);
      bool f_padding = handle.dofs[0][l] % 2 == 0;
      bool c_padding = handle.dofs[1][l] % 2 == 0;
      bool r_padding = handle.dofs[0][l] % 2 == 0;

      // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      // handle.dofs[0][l],
      //                   dv, lddv1, lddv2, lddv1);

      int curr_dim_r, curr_dim_c, curr_dim_f;
      int lddv1, lddv2;
      int lddw1, lddw2;
      int lddb1, lddb2;

      thrust::device_vector<int> shape(handle.D_padded);
      thrust::device_vector<int> shape_c(handle.D_padded);
      for (int d = 0; d < handle.D_padded; d++) {
        shape[d] = handle.dofs[d][l];
        shape_c[d] = handle.dofs[d][l + 1];
      }

      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[curr_dim_r][l],
      //   handle.dofs[curr_dim_c][l], handle.dofs[curr_dim_f][l],
      //                     dv+i*ldvs[0]*ldvs[1]*ldvs[2], ldvs[0], ldvs[1],
      //                     ldvs[0]);
      // }

      // start correction calculation
      curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
      int prev_dim_r, prev_dim_c, prev_dim_f;
      T *dw_out = handle.dw;
      T *dw_in1 = dv;
      T *dw_in2 =
          dv + get_idx(ldvs, gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f, 0,
                                     0, handle.dofs[curr_dim_f][l + 1]));
      // mass trans 1D
      curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
      prev_dim_f = curr_dim_f;
      prev_dim_c = curr_dim_c;
      prev_dim_r = curr_dim_r;
      lddv1 = 1, lddv2 = 1, lddw1 = 1, lddw2 = 1;
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddv1 *= ldvs[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddv2 *= ldvs[s];
      }
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddw1 *= handle.ldws_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddw2 *= handle.ldws_h[s];
      }
      thrust::device_vector<int> processed_dims;
      // printf("mass trans 1D\n");
      lpk_reo_1<T, D>(
          handle, handle.shapes_h[l], handle.shapes_h[l + 1],
          handle.shapes_d[l], handle.shapes_d[l + 1], ldvs_d, handle.ldws_d,
          handle.processed_n[0], handle.processed_dims_h[0],
          handle.processed_dims_d[0], curr_dim_r, curr_dim_c, curr_dim_f,
          handle.dist[curr_dim_f][l], handle.ratio[curr_dim_f][l], dw_in1,
          lddv1, lddv2,
          // dv+get_idx(ldvs, gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f, 0,
          // 0, handle.dofs[0][l+1])),
          dw_in2, lddv1, lddv2, dw_out, lddw1, lddw2, 0,
          handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

      // printf("after mass_trans_1\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      //   handle.dofs[0][l],
      //                     handle.dw+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                     handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);
      // }
      // printf("solve tridiag 1D\n");
      ipk_1<T, D>(
          handle, handle.shapes_h[l], handle.shapes_h[l + 1],
          handle.shapes_d[l], handle.shapes_d[l + 1], handle.ldws_d,
          handle.ldws_d, handle.processed_n[0], handle.processed_dims_h[0],
          handle.processed_dims_d[0], curr_dim_r, curr_dim_c, curr_dim_f,
          handle.am[curr_dim_f][l + 1], handle.bm[curr_dim_f][l + 1],
          handle.dist[curr_dim_f][l + 1], dw_out, lddw1, lddw2, 0,
          handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);
      // processed_dims.push_back(curr_dim_f);

      // printf("after solve_tridiag_1\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      //   handle.dofs[0][l+1],
      //                     dw_out+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                     handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);
      // }

      // mass trans 2D
      curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
      dw_in1 = dw_out;
      dw_in2 = dw_out + get_idx(handle.ldws_h,
                                gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f,
                                        0, handle.dofs[curr_dim_c][l + 1], 0));
      dw_out +=
          get_idx(handle.ldws_h, gen_idx(D, prev_dim_r, prev_dim_c, prev_dim_f,
                                         0, 0, handle.dofs[prev_dim_f][l + 1]));
      prev_dim_f = curr_dim_f;
      prev_dim_c = curr_dim_c;
      prev_dim_r = curr_dim_r;
      lddv1 = 1, lddv2 = 1, lddw1 = 1, lddw2 = 1;
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddw1 *= handle.ldws_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddw2 *= handle.ldws_h[s];
      }
      // printf("mass trans 2D\n");
      lpk_reo_2<T, D>(
          handle, handle.shapes_h[l], handle.shapes_h[l + 1],
          handle.shapes_d[l], handle.shapes_d[l + 1], handle.ldws_d,
          handle.ldws_d, handle.processed_n[1], handle.processed_dims_h[1],
          handle.processed_dims_d[1], curr_dim_r, curr_dim_c, curr_dim_f,
          handle.dist[curr_dim_c][l], handle.ratio[curr_dim_c][l],
          // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r, curr_dim_c,
          // curr_dim_f, 0, 0, 0)),
          dw_in1, lddw1, lddw2,
          // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r, curr_dim_c,
          // curr_dim_f, 0, handle.dofs[1][l+1], 0)),
          dw_in2, lddw1, lddw2,
          // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r, curr_dim_c,
          // curr_dim_f, 0, 0, handle.dofs[0][l+1])),
          dw_out, lddw1, lddw2, 0,
          handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

      // printf("after mass_trans_2\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l+1],
      //   handle.dofs[0][l+1],
      //                     dw_out+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                     handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);
      // }
      // printf("solve tridiag 2D\n");
      ipk_2<T, D>(
          handle, handle.shapes_h[l], handle.shapes_h[l + 1],
          handle.shapes_d[l], handle.shapes_d[l + 1], handle.ldws_d,
          handle.ldws_d, handle.processed_n[1], handle.processed_dims_h[1],
          handle.processed_dims_d[1], curr_dim_r, curr_dim_c, curr_dim_f,
          handle.am[curr_dim_c][l + 1], handle.bm[curr_dim_c][l + 1],
          handle.dist[curr_dim_c][l + 1], dw_out, lddw1, lddw2, 0,
          handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);
      // processed_dims.push_back(curr_dim_c);
      // printf("after solve_tridiag_2\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l+1],
      //   handle.dofs[0][l+1],
      //                     dw_out+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                     handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);
      // }

      // mass trans 3D

      curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
      dw_in1 = dw_out;
      dw_in2 = dw_out + get_idx(handle.ldws_h,
                                gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f,
                                        handle.dofs[curr_dim_r][l + 1], 0, 0));
      dw_out +=
          get_idx(handle.ldws_h, gen_idx(D, prev_dim_r, prev_dim_c, prev_dim_f,
                                         0, handle.dofs[prev_dim_c][l + 1], 0));
      prev_dim_f = curr_dim_f;
      prev_dim_c = curr_dim_c;
      prev_dim_r = curr_dim_r;
      lddv1 = 1, lddv2 = 1, lddw1 = 1, lddw2 = 1;
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddw1 *= handle.ldws_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddw2 *= handle.ldws_h[s];
      }
      // printf("mass trans 3D\n");
      lpk_reo_3<T, D>(
          handle, handle.shapes_h[l], handle.shapes_h[l + 1],
          handle.shapes_d[l], handle.shapes_d[l + 1], handle.ldws_d,
          handle.ldws_d, handle.processed_n[2], handle.processed_dims_h[2],
          handle.processed_dims_d[2], curr_dim_r, curr_dim_c, curr_dim_f,
          handle.dist[curr_dim_r][l], handle.ratio[curr_dim_r][l],
          // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r, curr_dim_c,
          // curr_dim_f, 0, 0, handle.dofs[0][l+1])),
          dw_in1, lddw1, lddw2,
          // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r, curr_dim_c,
          // curr_dim_f, handle.dofs[2][l+1], 0, handle.dofs[0][l+1])),
          dw_in2, lddw1, lddw2,
          // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r, curr_dim_c,
          // curr_dim_f, 0, handle.dofs[1][l+1], handle.dofs[0][l+1])),
          dw_out, lddw1, lddw2, 0,
          handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

      // printf("after mass_trans_3\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[2][l+1], handle.dofs[1][l+1],
      //   handle.dofs[0][l+1],
      //                     dw_out+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                     handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);
      // }
      // printf("solve tridiag 3D\n");
      ipk_3<T, D>(
          handle, handle.shapes_h[l], handle.shapes_h[l + 1],
          handle.shapes_d[l], handle.shapes_d[l + 1], handle.ldws_d,
          handle.ldws_d, handle.processed_n[2], handle.processed_dims_h[2],
          handle.processed_dims_d[2], curr_dim_r, curr_dim_c, curr_dim_f,
          handle.am[curr_dim_r][l + 1], handle.bm[curr_dim_r][l + 1],
          handle.dist[curr_dim_r][l + 1], dw_out, lddw1, lddw2, 0,
          handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);
      // processed_dims.push_back(curr_dim_r);
      // printf("after solve_tridiag_3\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[2][l+1], handle.dofs[1][l+1],
      //   handle.dofs[0][l+1],
      //                     dw_out+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                     handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);
      // }

      // mass trans 4D
      for (int i = 3; i < D; i++) {
        curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = i;
        dw_in1 = dw_out;
        dw_in2 =
            dw_out +
            get_idx(handle.ldws_h,
                    gen_idx(handle.D_padded, curr_dim_r, curr_dim_c, curr_dim_f,
                            handle.dofs[curr_dim_r][l + 1], 0, 0));
        dw_out +=
            get_idx(handle.ldws_h,
                    gen_idx(handle.D_padded, prev_dim_r, prev_dim_c, prev_dim_f,
                            handle.dofs[prev_dim_r][l + 1], 0, 0));
        prev_dim_f = curr_dim_f;
        prev_dim_c = curr_dim_c;
        prev_dim_r = curr_dim_r;
        lddv1 = 1, lddv2 = 1, lddw1 = 1, lddw2 = 1;
        for (int s = curr_dim_f; s < curr_dim_c; s++) {
          lddw1 *= handle.ldws_h[s];
        }
        for (int s = curr_dim_c; s < curr_dim_r; s++) {
          lddw2 *= handle.ldws_h[s];
        }
        // printf("mass trans %dD\n", i+1);
        lpk_reo_3<T, D>(
            handle, handle.shapes_h[l], handle.shapes_h[l + 1],
            handle.shapes_d[l], handle.shapes_d[l + 1], handle.ldws_d,
            handle.ldws_d, handle.processed_n[i], handle.processed_dims_h[i],
            handle.processed_dims_d[i], curr_dim_r, curr_dim_c, curr_dim_f,
            handle.dist[curr_dim_r][l], handle.ratio[curr_dim_r][l],
            // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r,
            // curr_dim_c, curr_dim_f, 0, 0, handle.dofs[0][l+1])),
            dw_in1, lddw1, lddw2,
            // handle.dw+get_idx(handle.ldws_h, gen_idx(D, curr_dim_r,
            // curr_dim_c, curr_dim_f, handle.dofs[2][l+1], 0,
            // handle.dofs[0][l+1])),
            dw_in2, lddw1, lddw2, dw_out, lddw1, lddw2, 0,
            handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

        // printf("after mass_trans_4\n");
        // for (int i = 0; i < handle.dofs[3][l+1]; i++) {
        //   printf("i = %d\n", i);
        //   print_matrix_cuda(handle.dofs[2][l+1], handle.dofs[1][l+1],
        //   handle.dofs[0][l+1],
        //                     dw_out+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
        //                     handle.ldws_h[0], handle.ldws_h[1],
        //                     handle.ldws_h[0]);
        // }
        // printf("solve tridiag %dD\n", i+1);
        ipk_3<T, D>(
            handle, handle.shapes_h[l], handle.shapes_h[l + 1],
            handle.shapes_d[l], handle.shapes_d[l + 1], handle.ldws_d,
            handle.ldws_d, handle.processed_n[i], handle.processed_dims_h[i],
            handle.processed_dims_d[i], curr_dim_r, curr_dim_c, curr_dim_f,
            handle.am[curr_dim_r][l + 1], handle.bm[curr_dim_r][l + 1],
            handle.dist[curr_dim_r][l + 1], dw_out, lddw1, lddw2, 0,
            handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);
        // processed_dims.push_back(i);
      }

      // printf("after solve_tridiag_4\n");
      // for (int i = 0; i < handle.dofs[3][l+1]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[2][l+1], handle.dofs[1][l+1],
      //   handle.dofs[0][l+1],
      //                     dw_out+i*handle.ldws_h[0]*handle.ldws_h[1]*handle.ldws_h[2],
      //                     handle.ldws_h[0], handle.ldws_h[1],
      //                     handle.ldws_h[0]);
      // }

      // un-apply correction
      lwpk<T, D, SUBTRACT>(handle, handle.shapes_h[l + 1],
                           handle.shapes_d[l + 1], dw_out, handle.ldws_d, dv,
                           ldvs_d, 0);

      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[curr_dim_r][l],
      //   handle.dofs[curr_dim_c][l], handle.dofs[curr_dim_f][l],
      //                     dv+i*ldvs[0]*ldvs[1]*ldvs[2], ldvs[0], ldvs[1],
      //                     ldvs[0]);
      // }

      lwpk<T, D, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l], dv,
                       ldvs_d, handle.db, handle.ldbs_d, 0);

      // printf("interpolate 1-3D rev\n");
      thrust::device_vector<int> unprocessed_dims;
      for (int i = 3; i < D; i++)
        unprocessed_dims.push_back(i);
      curr_dim_r = 2, curr_dim_c = 1, curr_dim_f = 0;
      lddv1 = 1, lddv2 = 1, lddw1 = 1, lddw2 = 1;
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddv1 *= ldvs[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddv2 *= ldvs[s];
      }
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddw1 *= handle.ldws_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddw2 *= handle.ldws_h[s];
      }
      gpk_rev<T, D, 3, true, false, 1>(
          handle, handle.shapes_h[l], handle.shapes_d[l],
          handle.shapes_d[l + 1], handle.ldws_d, ldvs_d,
          unprocessed_dims.size(),
          thrust::raw_pointer_cast(unprocessed_dims.data()), curr_dim_r,
          curr_dim_c, curr_dim_f, handle.ratio[curr_dim_r][l],
          handle.ratio[curr_dim_c][l], handle.ratio[curr_dim_f][l], handle.dw,
          handle.ldws_h[0], handle.ldws_h[1], dv, lddv1, lddv2,
          dv + get_idx(ldvs, gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f, 0,
                                     0, handle.dofs[curr_dim_f][l + 1])),
          lddv1, lddv2,
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs, gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f, 0,
                                     handle.dofs[curr_dim_c][l + 1], 0)),
          lddv1, lddv2,
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs, gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f,
                                     handle.dofs[curr_dim_r][l + 1], 0, 0)),
          ldvs[0], ldvs[1],
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs, gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f, 0,
                                     handle.dofs[curr_dim_c][l + 1],
                                     handle.dofs[curr_dim_f][l + 1])),
          lddv1, lddv2,
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs, gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f,
                                     handle.dofs[curr_dim_r][l + 1], 0,
                                     handle.dofs[curr_dim_f][l + 1])),
          lddv1, lddv2,
          // null, ldvs[0], ldvs[1],
          dv + get_idx(ldvs, gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f,
                                     handle.dofs[curr_dim_r][l + 1],
                                     handle.dofs[curr_dim_c][l + 1], 0)),
          lddv1, lddv2,
          // null,ldvs[0], ldvs[1],
          dv + get_idx(ldvs, gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f,
                                     handle.dofs[curr_dim_r][l + 1],
                                     handle.dofs[curr_dim_c][l + 1],
                                     handle.dofs[curr_dim_f][l + 1])),
          lddv1, lddv2,
          // null, ldvs[0], ldvs[1],
          0, 0, 0, handle.dofs[curr_dim_r][l], handle.dofs[curr_dim_c][l],
          handle.dofs[curr_dim_f][l], 0,
          handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

      lwpk<T, D, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l],
                       handle.dw, handle.ldws_d, dv, ldvs_d, 0);

      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[curr_dim_r][l],
      //   handle.dofs[curr_dim_c][l], handle.dofs[curr_dim_f][l],
      //                     dv+i*ldvs[0]*ldvs[1]*ldvs[2], ldvs[0], ldvs[1],
      //                     ldvs[0]);
      // }

      // printf("interpolate 4-5D rev\n");

      curr_dim_r = 4, curr_dim_c = 3, curr_dim_f = 0;
      lddv1 = 1, lddv2 = 1, lddw1 = 1, lddw2 = 1;
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddv1 *= ldvs[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddv2 *= ldvs[s];
      }
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddw1 *= handle.ldws_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddw2 *= handle.ldws_h[s];
      }
      if (D % 2 == 0) {
        unprocessed_dims.pop_back();
        gpk_rev<T, D, 2, true, false, 2>(
            handle, handle.shapes_h[l], handle.shapes_d[l],
            handle.shapes_d[l + 1], handle.ldws_d, ldvs_d,
            unprocessed_dims.size(),
            thrust::raw_pointer_cast(unprocessed_dims.data()), curr_dim_r,
            curr_dim_c, curr_dim_f, handle.ratio[curr_dim_r][l],
            handle.ratio[curr_dim_c][l], handle.ratio[curr_dim_f][l], handle.dw,
            lddw1, lddw2, dv, lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f, 0, 0,
                                       handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f, 0,
                                       handle.dofs[curr_dim_c][l + 1], 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f,
                                       handle.dofs[curr_dim_r][l + 1], 0, 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, 0, handle.dofs[curr_dim_c][l + 1],
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1], 0,
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                 handle.dofs[curr_dim_c][l + 1], 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                 handle.dofs[curr_dim_c][l + 1],
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            0, 0, 0, handle.dofs[curr_dim_r][l], handle.dofs[curr_dim_c][l],
            handle.dofs[curr_dim_f][l], 0,
            handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
      } else {
        unprocessed_dims.pop_back();
        unprocessed_dims.pop_back();
        gpk_rev<T, D, 3, true, false, 2>(
            handle, handle.shapes_h[l], handle.shapes_d[l],
            handle.shapes_d[l + 1], handle.ldws_d, ldvs_d,
            unprocessed_dims.size(),
            thrust::raw_pointer_cast(unprocessed_dims.data()), curr_dim_r,
            curr_dim_c, curr_dim_f, handle.ratio[curr_dim_r][l],
            handle.ratio[curr_dim_c][l], handle.ratio[curr_dim_f][l], handle.dw,
            lddw1, lddw2, dv, lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f, 0, 0,
                                       handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f, 0,
                                       handle.dofs[curr_dim_c][l + 1], 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs, gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                       curr_dim_f,
                                       handle.dofs[curr_dim_r][l + 1], 0, 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, 0, handle.dofs[curr_dim_c][l + 1],
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1], 0,
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                 handle.dofs[curr_dim_c][l + 1], 0)),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            dv + get_idx(ldvs,
                         gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                 curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                 handle.dofs[curr_dim_c][l + 1],
                                 handle.dofs[curr_dim_f][l + 1])),
            lddv1, lddv2,
            // null, lddv1, lddv2,
            0, 0, 0, handle.dofs[curr_dim_r][l], handle.dofs[curr_dim_c][l],
            handle.dofs[curr_dim_f][l], 0,
            handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
      }

      lwpk<T, D, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l],
                       handle.dw, handle.ldws_d, dv, ldvs_d, 0);
      lwpk<T, D, COPY>(handle, shape, handle.dw, handle.ldws_h, dv, ldvs, 0);

      // printf("after interpolate 4D rev\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[curr_dim_r][l],
      //   handle.dofs[curr_dim_c][l], handle.dofs[curr_dim_f][l],
      //                     dv+i*ldvs[0]*ldvs[1]*ldvs[2], ldvs[0], ldvs[1],
      //                     ldvs[0]);
      // }

      // printf("reorder restore 1-3D\n");
      curr_dim_r = 2, curr_dim_c = 1, curr_dim_f = 0;
      lddw1 = 1, lddw2 = 1, lddb1 = 1, lddb2 = 1;
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddw1 *= handle.ldws_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddw2 *= handle.ldws_h[s];
      }
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddb1 *= handle.ldbs_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddb2 *= handle.ldbs_h[s];
      }
      for (int i = 3; i < D; i++)
        unprocessed_dims.push_back(i);

      gpk_rev<T, D, 3, false, false, 1>(
          handle, handle.shapes_h[l], handle.shapes_d[l],
          handle.shapes_d[l + 1], handle.ldws_d, handle.ldbs_d,
          unprocessed_dims.size(),
          thrust::raw_pointer_cast(unprocessed_dims.data()), curr_dim_r,
          curr_dim_c, curr_dim_f, handle.ratio[curr_dim_r][l],
          handle.ratio[curr_dim_c][l], handle.ratio[curr_dim_f][l], handle.dw,
          lddw1, lddw2, handle.db, lddb1, lddb2,
          handle.db + get_idx(handle.ldbs_h,
                              gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f, 0,
                                      0, handle.dofs[curr_dim_f][l + 1])),
          lddb1, lddb2,
          // null, ldvs[0], ldvs[1],
          handle.db + get_idx(handle.ldbs_h,
                              gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f, 0,
                                      handle.dofs[curr_dim_c][l + 1], 0)),
          lddb1, lddb2,
          // null, ldvs[0], ldvs[1],
          handle.db + get_idx(handle.ldbs_h,
                              gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f,
                                      handle.dofs[curr_dim_r][l + 1], 0, 0)),
          lddb1, lddb2,
          // null, ldvs[0], ldvs[1],
          handle.db + get_idx(handle.ldbs_h,
                              gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f, 0,
                                      handle.dofs[curr_dim_c][l + 1],
                                      handle.dofs[curr_dim_f][l + 1])),
          lddb1, lddb2,
          // null, ldvs[0], ldvs[1],
          handle.db + get_idx(handle.ldbs_h,
                              gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f,
                                      handle.dofs[curr_dim_r][l + 1], 0,
                                      handle.dofs[curr_dim_f][l + 1])),
          lddb1, lddb2,
          // null, ldvs[0], ldvs[1],
          handle.db + get_idx(handle.ldbs_h,
                              gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f,
                                      handle.dofs[curr_dim_r][l + 1],
                                      handle.dofs[curr_dim_c][l + 1], 0)),
          lddb1, lddb2,
          // null,ldvs[0], ldvs[1],
          handle.db + get_idx(handle.ldbs_h,
                              gen_idx(D, curr_dim_r, curr_dim_c, curr_dim_f,
                                      handle.dofs[curr_dim_r][l + 1],
                                      handle.dofs[curr_dim_c][l + 1],
                                      handle.dofs[curr_dim_f][l + 1])),
          lddb1, lddb2,
          // null, ldvs[0], ldvs[1],
          0, 0, 0, handle.dofs[curr_dim_r][l], handle.dofs[curr_dim_c][l],
          handle.dofs[curr_dim_f][l], 0,
          handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

      lwpk<T, D, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l],
                       handle.dw, handle.ldws_d, handle.db, handle.ldbs_d, 0);

      // printf("reorder 1-3D rev\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[curr_dim_r][l],
      //   handle.dofs[curr_dim_c][l], handle.dofs[curr_dim_f][l],
      //                     handle.db+i*handle.ldbs_h[0]*handle.ldbs_h[1]*handle.ldbs_h[2],
      //                     handle.ldbs_h[0], handle.ldbs_h[1],
      //                     handle.ldbs_h[0]);
      // }

      // printf("reorder restore nodal values 1-4D\n");

      curr_dim_r = 4, curr_dim_c = 3, curr_dim_f = 0;
      lddv1 = 1, lddv2 = 1, lddb1 = 1, lddb2 = 1;
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddv1 *= ldvs[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddv2 *= ldvs[s];
      }
      for (int s = curr_dim_f; s < curr_dim_c; s++) {
        lddb1 *= handle.ldbs_h[s];
      }
      for (int s = curr_dim_c; s < curr_dim_r; s++) {
        lddb2 *= handle.ldbs_h[s];
      }
      if (D % 2 == 0) {
        unprocessed_dims.pop_back();
        gpk_rev<T, D, 2, false, true, 2>(
            handle, handle.shapes_h[l], handle.shapes_d[l],
            handle.shapes_d[l + 1], ldvs_d, handle.ldbs_d,
            unprocessed_dims.size(),
            thrust::raw_pointer_cast(unprocessed_dims.data()), curr_dim_r,
            curr_dim_c, curr_dim_f, handle.ratio[curr_dim_r][l],
            handle.ratio[curr_dim_c][l], handle.ratio[curr_dim_f][l], dv, lddv1,
            lddv2, handle.db, lddb1, lddb2,
            // null, lddv1, lddv2,
            handle.db +
                get_idx(handle.ldbs_h, gen_idx(handle.D_padded, curr_dim_r,
                                               curr_dim_c, curr_dim_f, 0, 0,
                                               handle.dofs[curr_dim_f][l + 1])),
            lddb1, lddb2,
            // null, lddv1, lddv2,
            handle.db + get_idx(handle.ldbs_h,
                                gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                        curr_dim_f, 0,
                                        handle.dofs[curr_dim_c][l + 1], 0)),
            lddb1, lddb2,
            // null, lddv1, lddv2,
            handle.db + get_idx(handle.ldbs_h,
                                gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                        curr_dim_f,
                                        handle.dofs[curr_dim_r][l + 1], 0, 0)),
            lddb1, lddb2,
            // null, lddv1, lddv2,
            handle.db +
                get_idx(handle.ldbs_h,
                        gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                curr_dim_f, 0, handle.dofs[curr_dim_c][l + 1],
                                handle.dofs[curr_dim_f][l + 1])),
            lddb1, lddb2,
            // null, lddv1, lddv2,
            handle.db +
                get_idx(handle.ldbs_h,
                        gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                curr_dim_f, handle.dofs[curr_dim_r][l + 1], 0,
                                handle.dofs[curr_dim_f][l + 1])),
            lddb1, lddb2,
            // null, lddv1, lddv2,
            handle.db +
                get_idx(handle.ldbs_h,
                        gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                handle.dofs[curr_dim_c][l + 1], 0)),
            lddb1, lddb2,
            // null, lddv1, lddv2,
            handle.db +
                get_idx(handle.ldbs_h,
                        gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                handle.dofs[curr_dim_c][l + 1],
                                handle.dofs[curr_dim_f][l + 1])),
            lddb1, lddb2,
            // null, lddv1, lddv2,
            0, 0, 0, handle.dofs[curr_dim_r][l], handle.dofs[curr_dim_c][l],
            handle.dofs[curr_dim_f][l], 0,
            handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
      } else {
        gpk_rev<T, D, 3, false, true, 2>(
            handle, handle.shapes_h[l], handle.shapes_d[l],
            handle.shapes_d[l + 1], ldvs_d, handle.ldbs_d,
            unprocessed_dims.size(),
            thrust::raw_pointer_cast(unprocessed_dims.data()), curr_dim_r,
            curr_dim_c, curr_dim_f, handle.ratio[curr_dim_r][l],
            handle.ratio[curr_dim_c][l], handle.ratio[curr_dim_f][l], dv, lddv1,
            lddv2, handle.db, lddb1, lddb2,
            // null, lddv1, lddv2,
            handle.db +
                get_idx(handle.ldbs_h, gen_idx(handle.D_padded, curr_dim_r,
                                               curr_dim_c, curr_dim_f, 0, 0,
                                               handle.dofs[curr_dim_f][l + 1])),
            lddb1, lddb2,
            // null, lddv1, lddv2,
            handle.db + get_idx(handle.ldbs_h,
                                gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                        curr_dim_f, 0,
                                        handle.dofs[curr_dim_c][l + 1], 0)),
            lddb1, lddb2,
            // null, lddv1, lddv2,
            handle.db + get_idx(handle.ldbs_h,
                                gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                        curr_dim_f,
                                        handle.dofs[curr_dim_r][l + 1], 0, 0)),
            lddb1, lddb2,
            // null, lddv1, lddv2,
            handle.db +
                get_idx(handle.ldbs_h,
                        gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                curr_dim_f, 0, handle.dofs[curr_dim_c][l + 1],
                                handle.dofs[curr_dim_f][l + 1])),
            lddb1, lddb2,
            // null, lddv1, lddv2,
            handle.db +
                get_idx(handle.ldbs_h,
                        gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                curr_dim_f, handle.dofs[curr_dim_r][l + 1], 0,
                                handle.dofs[curr_dim_f][l + 1])),
            lddb1, lddb2,
            // null, lddv1, lddv2,
            handle.db +
                get_idx(handle.ldbs_h,
                        gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                handle.dofs[curr_dim_c][l + 1], 0)),
            lddb1, lddb2,
            // null, lddv1, lddv2,
            handle.db +
                get_idx(handle.ldbs_h,
                        gen_idx(handle.D_padded, curr_dim_r, curr_dim_c,
                                curr_dim_f, handle.dofs[curr_dim_r][l + 1],
                                handle.dofs[curr_dim_c][l + 1],
                                handle.dofs[curr_dim_f][l + 1])),
            lddb1, lddb2,
            // null, lddv1, lddv2,
            0, 0, 0, handle.dofs[curr_dim_r][l], handle.dofs[curr_dim_c][l],
            handle.dofs[curr_dim_f][l], 0,
            handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
      }

      // printf("after coeff restore 4D rev\n");
      // for (int i = 0; i < handle.dofs[3][l]; i++) {
      //   printf("i = %d\n", i);
      //   print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      //   handle.dofs[0][l],
      //                     dv+i*ldvs[0]*ldvs[1]*ldvs[2], ldvs[0], ldvs[1],
      //                     ldvs[0]);
      // }
    }
  }
}

#define KERNELS(T, D)                                                          \
  template void refactor_reo<T, D>(Handle<T, D> & handle, T * dv,              \
                                   std::vector<int> ldvs, int l_target);       \
  template void recompose_reo<T, D>(Handle<T, D> & handle, T * dv,             \
                                    std::vector<int> ldvs, int l_target);

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
