/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cuda/CommonInternal.h"
#include "cuda/SubArray.h"

#include "cuda/GridProcessingKernel.h"
#include "cuda/GridProcessingKernel3D.h"
#include "cuda/GridProcessingKernel3D.hpp"
#include "cuda/IterativeProcessingKernel.h"
#include "cuda/IterativeProcessingKernel3D.h"
#include "cuda/LevelwiseProcessingKernel.h"
#include "cuda/LinearProcessingKernel.h"
#include "cuda/LinearProcessingKernel3D.h"

#include "cuda/DataRefactoring.h"

// #include "cuda/Testing/ReorderToolsGPU.hpp"

#include <iostream>

#include <chrono>
namespace mgard_cuda {

static bool store = false;
static bool verify = false;
static bool debug_print = false;

template <DIM D, typename T>
void calc_coeff_pointers(Handle<D, T> &handle, DIM curr_dims[3], DIM l,
                         SubArray<D, T> doutput, SubArray<D, T> &dcoarse,
                         SubArray<D, T> &dcoeff_f, SubArray<D, T> &dcoeff_c,
                         SubArray<D, T> &dcoeff_r, SubArray<D, T> &dcoeff_cf,
                         SubArray<D, T> &dcoeff_rf, SubArray<D, T> &dcoeff_rc,
                         SubArray<D, T> &dcoeff_rcf) {

  SIZE n[3];
  SIZE nn[3];
  for (DIM d = 0; d < 3; d++) {
    n[d] = handle.dofs[curr_dims[d]][l];
    nn[d] = handle.dofs[curr_dims[d]][l + 1];
  }

  dcoarse = doutput;
  dcoarse.resize(curr_dims[0], nn[0]);
  dcoarse.resize(curr_dims[1], nn[1]);
  dcoarse.resize(curr_dims[2], nn[2]);

  dcoeff_f = doutput;
  dcoeff_f.offset(curr_dims[0], nn[0]);
  dcoeff_f.resize(curr_dims[0], n[0] - nn[0]);
  dcoeff_f.resize(curr_dims[1], nn[1]);
  dcoeff_f.resize(curr_dims[2], nn[2]);

  dcoeff_c = doutput;
  dcoeff_c.offset(curr_dims[1], nn[1]);
  dcoeff_c.resize(curr_dims[0], nn[0]);
  dcoeff_c.resize(curr_dims[1], n[1] - nn[1]);
  dcoeff_c.resize(curr_dims[2], nn[2]);

  dcoeff_r = doutput;
  dcoeff_r.offset(curr_dims[2], nn[2]);
  dcoeff_r.resize(curr_dims[0], nn[0]);
  dcoeff_r.resize(curr_dims[1], nn[1]);
  dcoeff_r.resize(curr_dims[2], n[2] - nn[2]);

  dcoeff_cf = doutput;
  dcoeff_cf.offset(curr_dims[0], nn[0]);
  dcoeff_cf.offset(curr_dims[1], nn[1]);
  dcoeff_cf.resize(curr_dims[0], n[0] - nn[0]);
  dcoeff_cf.resize(curr_dims[1], n[1] - nn[1]);
  dcoeff_cf.resize(curr_dims[2], nn[2]);

  dcoeff_rf = doutput;
  dcoeff_rf.offset(curr_dims[0], nn[0]);
  dcoeff_rf.offset(curr_dims[2], nn[2]);
  dcoeff_rf.resize(curr_dims[0], n[0] - nn[0]);
  dcoeff_rf.resize(curr_dims[1], nn[1]);
  dcoeff_rf.resize(curr_dims[2], n[2] - nn[2]);

  dcoeff_rc = doutput;
  dcoeff_rc.offset(curr_dims[1], nn[1]);
  dcoeff_rc.offset(curr_dims[2], nn[2]);
  dcoeff_rc.resize(curr_dims[0], nn[0]);
  dcoeff_rc.resize(curr_dims[1], n[1] - nn[1]);
  dcoeff_rc.resize(curr_dims[2], n[2] - nn[2]);

  dcoeff_rcf = doutput;
  dcoeff_rcf.offset(curr_dims[0], nn[0]);
  dcoeff_rcf.offset(curr_dims[1], nn[1]);
  dcoeff_rcf.offset(curr_dims[2], nn[2]);
  dcoeff_rcf.resize(curr_dims[0], n[0] - nn[0]);
  dcoeff_rcf.resize(curr_dims[1], n[1] - nn[1]);
  dcoeff_rcf.resize(curr_dims[2], n[2] - nn[2]);
}

template <DIM D, typename T>
void calc_coefficients_3d(Handle<D, T> &handle, SubArray<D, T> dinput,
                          SubArray<D, T> &doutput, SIZE l, int queue_idx) {

  int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
  int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(handle.shapes_h[0][d]) + "_";

  dinput.project(0, 1, 2);
  doutput.project(0, 1, 2);

  SIZE f = handle.dofs[0][l];
  SIZE c = handle.dofs[1][l];
  SIZE r = handle.dofs[2][l];
  SIZE ff = handle.dofs[0][l + 1];
  SIZE cc = handle.dofs[1][l + 1];
  SIZE rr = handle.dofs[2][l + 1];

  SubArray<D, T> dcoarse = doutput;
  dcoarse.resize({ff, cc, rr});
  SubArray<D, T> dcoeff_f = doutput;
  dcoeff_f.offset({ff, 0, 0});
  dcoeff_f.resize({f - ff, cc, rr});
  SubArray<D, T> dcoeff_c = doutput;
  dcoeff_c.offset({0, cc, 0});
  dcoeff_c.resize({ff, c - cc, rr});
  SubArray<D, T> dcoeff_r = doutput;
  dcoeff_r.offset({0, 0, rr});
  dcoeff_r.resize({ff, cc, r - rr});
  SubArray<D, T> dcoeff_cf = doutput;
  dcoeff_cf.offset({ff, cc, 0});
  dcoeff_cf.resize({f - ff, c - cc, rr});
  SubArray<D, T> dcoeff_rf = doutput;
  dcoeff_rf.offset({ff, 0, rr});
  dcoeff_rf.resize({f - ff, cc, r - rr});
  SubArray<D, T> dcoeff_rc = doutput;
  dcoeff_rc.offset({0, cc, rr});
  dcoeff_rc.resize({ff, c - cc, r - rr});
  SubArray<D, T> dcoeff_rcf = doutput;
  dcoeff_rcf.offset({ff, cc, rr});
  dcoeff_rcf.resize({f - ff, c - cc, r - rr});

  SubArray<1, T> ratio_r({handle.dofs[2][l]}, handle.ratio[2][l]);
  SubArray<1, T> ratio_c({handle.dofs[1][l]}, handle.ratio[1][l]);
  SubArray<1, T> ratio_f({handle.dofs[0][l]}, handle.ratio[0][l]);

  T *null = NULL;
  // // GpkReo3D<Handle<D, T>, D, T, CUDA>(handle).Execute(
  // //     handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
  // //     handle.dofs[2][l+1], handle.dofs[1][l+1], handle.dofs[0][l+1],
  // //     ratio_r, ratio_c, ratio_f,
  // //     dinput, dcoarse,
  // //     dcoeff_f, dcoeff_c, dcoeff_r,
  // //     dcoeff_cf, dcoeff_rf, dcoeff_rc,
  // //     dcoeff_rcf,
  // //     queue_idx);
  // // handle.sync_all();
  // if (debug_print) {
  //   printf("after pi_Ql_reo\n");
  //   print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
  //                     doutput.dv, doutput.ldvs_h[0], doutput.ldvs_h[1],
  //                     doutput.ldvs_h[0]);
  // }

  gpk_reo_3d(
      handle, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
      handle.ratio[2][l], handle.ratio[1][l], handle.ratio[0][l], dinput.dv,
      dinput.lddv1, dinput.lddv2, dcoarse.dv, dcoarse.lddv1, dcoarse.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      queue_idx, handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
  // handle.sync_all();
  verify_matrix_cuda(
      handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l], doutput.dv,
      doutput.ldvs_h[0], doutput.ldvs_h[1], doutput.ldvs_h[0],
      prefix + "gpk_reo_3d" + "_level_" + std::to_string(l), store, verify);

  if (debug_print) {
    printf("after pi_Ql_reo\n");
    print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
                      doutput.dv, doutput.ldvs_h[0], doutput.ldvs_h[1],
                      doutput.ldvs_h[0]);
  }
}

template <DIM D, typename T>
void coefficients_restore_3d(Handle<D, T> &handle, SubArray<D, T> dinput,
                             SubArray<D, T> &doutput, SIZE l, int queue_idx) {

  int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
  int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(handle.shapes_h[0][d]) + "_";

  dinput.project(0, 1, 2);
  doutput.project(0, 1, 2);

  SIZE f = handle.dofs[0][l];
  SIZE c = handle.dofs[1][l];
  SIZE r = handle.dofs[2][l];
  SIZE ff = handle.dofs[0][l + 1];
  SIZE cc = handle.dofs[1][l + 1];
  SIZE rr = handle.dofs[2][l + 1];

  SubArray<D, T> dcoarse = dinput;
  dcoarse.resize({ff, cc, rr});
  SubArray<D, T> dcoeff_f = dinput;
  dcoeff_f.offset({ff, 0, 0});
  dcoeff_f.resize({f - ff, cc, rr});
  SubArray<D, T> dcoeff_c = dinput;
  dcoeff_c.offset({0, cc, 0});
  dcoeff_c.resize({ff, c - cc, rr});
  SubArray<D, T> dcoeff_r = dinput;
  dcoeff_r.offset({0, 0, rr});
  dcoeff_r.resize({ff, cc, r - rr});
  SubArray<D, T> dcoeff_cf = dinput;
  dcoeff_cf.offset({ff, cc, 0});
  dcoeff_cf.resize({f - ff, c - cc, rr});
  SubArray<D, T> dcoeff_rf = dinput;
  dcoeff_rf.offset({ff, 0, rr});
  dcoeff_rf.resize({f - ff, cc, r - rr});
  SubArray<D, T> dcoeff_rc = dinput;
  dcoeff_rc.offset({0, cc, rr});
  dcoeff_rc.resize({ff, c - cc, r - rr});
  SubArray<D, T> dcoeff_rcf = dinput;
  dcoeff_rcf.offset({ff, cc, rr});
  dcoeff_rcf.resize({f - ff, c - cc, r - rr});

  SubArray<1, T> ratio_r({handle.dofs[2][l]}, handle.ratio[2][l]);
  SubArray<1, T> ratio_c({handle.dofs[1][l]}, handle.ratio[1][l]);
  SubArray<1, T> ratio_f({handle.dofs[0][l]}, handle.ratio[0][l]);

  // GpkRev3D<Handle<D, T>, D, T, CUDA>(handle).Execute(
  //     handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
  //     handle.dofs[2][l+1], handle.dofs[1][l+1], handle.dofs[0][l+1],
  //     ratio_r, ratio_c, ratio_f,
  //     doutput, dcoarse,
  //     dcoeff_f, dcoeff_c, dcoeff_r,
  //     dcoeff_cf, dcoeff_rf, dcoeff_rc,
  //     dcoeff_rcf,
  //     0, 0, 0,
  //     handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
  //     queue_idx);

  T *null = NULL;
  gpk_rev_3d(
      handle, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
      handle.ratio[2][l], handle.ratio[1][l], handle.ratio[0][l], doutput.dv,
      doutput.lddv1, doutput.lddv2, dcoarse.dv, dcoarse.lddv1, dcoarse.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
      // null, ldvs_h[0], ldvs_h[1],
      0, 0, 0, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
      queue_idx, handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

  // handle.sync(0);
  verify_matrix_cuda(
      handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l], doutput.dv,
      doutput.ldvs_h[0], doutput.ldvs_h[1], doutput.ldvs_h[0],
      prefix + "gpk_rev_3d" + "_level_" + std::to_string(l), store, verify);

  // gpk_rev<D, T, D, true, false, 1>(handle,
  //             shape, shape_c, handle.ldws_h, ldvs_h, unprocessed_dims,
  //             2, 1, 0,
  //             handle.ratio[2][l], handle.ratio[1][l], handle.ratio[0][l],
  //             handle.dw, handle.ldws_h[0], handle.ldws_h[1],
  //             dv, ldvs_h[0], ldvs_h[1],
  //             dv+get_idx(ldvs_h[0], ldvs_h[1],  0, 0, handle.dofs[0][l+1]),
  //             ldvs_h[0], ldvs_h[1],
  //             // null, ldvs_h[0], ldvs_h[1],
  //             dv+get_idx(ldvs_h[0], ldvs_h[1],  0, handle.dofs[1][l+1], 0),
  //             ldvs_h[0], ldvs_h[1],
  //             // null, ldvs_h[0], ldvs_h[1],
  //             dv+get_idx(ldvs_h[0], ldvs_h[1],  handle.dofs[2][l+1], 0, 0),
  //             ldvs_h[0], ldvs_h[1],
  //             // null, ldvs_h[0], ldvs_h[1],
  //             dv+get_idx(ldvs_h[0], ldvs_h[1],  0, handle.dofs[1][l+1],
  //             handle.dofs[0][l+1]), ldvs_h[0], ldvs_h[1],
  //             // null, ldvs_h[0], ldvs_h[1],
  //             dv+get_idx(ldvs_h[0], ldvs_h[1],  handle.dofs[2][l+1], 0,
  //             handle.dofs[0][l+1]), ldvs_h[0], ldvs_h[1],
  //             // null, ldvs_h[0], ldvs_h[1],
  //             dv+get_idx(ldvs_h[0], ldvs_h[1],  handle.dofs[2][l+1],
  //             handle.dofs[1][l+1], 0), ldvs_h[0], ldvs_h[1],
  //             // null,ldvs_h[0], ldvs_h[1],
  //             dv+get_idx(ldvs_h[0], ldvs_h[1],  handle.dofs[2][l+1],
  //             handle.dofs[1][l+1], handle.dofs[0][l+1]), ldvs_h[0],
  //             ldvs_h[1],
  //             // null, ldvs_h[0], ldvs_h[1],
  //             0, 0, 0, handle.dofs[2][l], handle.dofs[1][l],
  //             handle.dofs[0][l], 0,
  //             handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

  // print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
  // handle.dofs[0][l], doutput.dv, doutput.ldvs_h[0], doutput.ldvs_h[1],
  // doutput.ldvs_h[0],);

  // gpk_rev<D, T, D, false, true, 1>(handle,
  //             shape, shape_c, handle.ldws_h, ldvs_h, unprocessed_dims,
  //             2, 1, 0,
  //             handle.ratio[2][l], handle.ratio[1][l], handle.ratio[0][l],
  //             handle.dw, handle.ldws_h[0], handle.ldws_h[1],
  //             dv, ldvs_h[0], ldvs_h[1],
  //             dv+get_idx(ldvs_h[0], ldvs_h[1],  0, 0, handle.dofs[0][l+1]),
  //             ldvs_h[0], ldvs_h[1],
  //             // null, ldvs_h[0], ldvs_h[1],
  //             dv+get_idx(ldvs_h[0], ldvs_h[1],  0, handle.dofs[1][l+1], 0),
  //             ldvs_h[0], ldvs_h[1],
  //             // null, ldvs_h[0], ldvs_h[1],
  //             dv+get_idx(ldvs_h[0], ldvs_h[1],  handle.dofs[2][l+1], 0, 0),
  //             ldvs_h[0], ldvs_h[1],
  //             // null, ldvs_h[0], ldvs_h[1],
  //             dv+get_idx(ldvs_h[0], ldvs_h[1],  0, handle.dofs[1][l+1],
  //             handle.dofs[0][l+1]), ldvs_h[0], ldvs_h[1],
  //             // null, ldvs_h[0], ldvs_h[1],
  //             dv+get_idx(ldvs_h[0], ldvs_h[1],  handle.dofs[2][l+1], 0,
  //             handle.dofs[0][l+1]), ldvs_h[0], ldvs_h[1],
  //             // null, ldvs_h[0], ldvs_h[1],
  //             dv+get_idx(ldvs_h[0], ldvs_h[1],  handle.dofs[2][l+1],
  //             handle.dofs[1][l+1], 0), ldvs_h[0], ldvs_h[1],
  //             // null,ldvs_h[0], ldvs_h[1],
  //             dv+get_idx(ldvs_h[0], ldvs_h[1],  handle.dofs[2][l+1],
  //             handle.dofs[1][l+1], handle.dofs[0][l+1]), ldvs_h[0],
  //             ldvs_h[1],
  //             // null, ldvs_h[0], ldvs_h[1],
  //             0, 0, 0, handle.dofs[2][l], handle.dofs[1][l],
  //             handle.dofs[0][l], 0,
  //             handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

  if (debug_print) {
    printf("after coeff-restore\n");
    print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
                      doutput.dv, doutput.ldvs_h[0], doutput.ldvs_h[1],
                      doutput.ldvs_h[0]);
  }
}

template <DIM D, typename T>
void calc_correction_3d(Handle<D, T> &handle, SubArray<D, T> dcoeff,
                        SubArray<D, T> &dcorrection, SIZE l, int queue_idx) {

  int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
  int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(handle.shapes_h[0][d]) + "_";

  SubArray<D, T> dw_in1, dw_in2, dw_out;
  if (D >= 1) {
    dw_in1 = dcoeff;
    dw_in1.resize(
        {handle.dofs[0][l + 1], handle.dofs[1][l], handle.dofs[2][l]});
    dw_in2 = dcoeff;
    dw_in2.offset({handle.dofs[0][l + 1], 0, 0});
    dw_in2.resize({handle.dofs[0][l] - handle.dofs[0][l + 1], handle.dofs[1][l],
                   handle.dofs[2][l]});
    dw_out = dcorrection;
    dw_out.resize({handle.dofs[0][l + 1], handle.dofs[1][l], handle.dofs[2][l]});

    lpk_reo_1_3d(
        handle, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
        handle.dofs[0][l + 1], handle.dofs[2][l + 1], handle.dofs[1][l + 1],
        handle.dofs[0][l + 1], handle.dist[0][l], handle.ratio[0][l], dw_in1.dv,
        dw_in1.ldvs_h[0], dw_in1.ldvs_h[1], dw_in2.dv, dw_in2.ldvs_h[0],
        dw_in2.ldvs_h[1], dw_out.dv, dw_out.ldvs_h[0], dw_out.ldvs_h[1],
        queue_idx,
        handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

    verify_matrix_cuda(
        handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l + 1], dw_out.dv,
        dw_out.ldvs_h[0], dw_out.ldvs_h[1], dw_out.ldvs_h[0],
        prefix + "lpk_reo_1_3d" + "_level_" + std::to_string(l), store, verify);

    if (debug_print) {
      printf("after mass_trans_multiply_1_cpt:\n");
      print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
                        handle.dofs[0][l + 1], dw_out.dv, dw_out.ldvs_h[0],
                        dw_out.ldvs_h[1], dw_out.ldvs_h[0]);
    }

    // PrintSubarray("after mass_trans_multiply_1_cpt::dw_in1", dw_in1);
    // PrintSubarray("after mass_trans_multiply_1_cpt::dw_in2", dw_in2);
    // PrintSubarray("after mass_trans_multiply_1_cpt::dw_out", dw_out);

  }

  if (D >= 2) {
    dw_in1 = dw_out;
    dw_in1.resize(
        {handle.dofs[0][l + 1], handle.dofs[1][l + 1], handle.dofs[2][l]});
    dw_in2 = dw_out;
    dw_in2.offset({0, handle.dofs[1][l + 1], 0});
    dw_in2.resize({handle.dofs[0][l + 1],
                   handle.dofs[1][l] - handle.dofs[1][l + 1],
                   handle.dofs[2][l]});
    dw_out.offset({handle.dofs[0][l + 1], 0, 0});
    dw_out.resize(
        {handle.dofs[0][l + 1], handle.dofs[1][l + 1], handle.dofs[2][l]});

    lpk_reo_2_3d(
        handle, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l + 1],
        handle.dofs[1][l + 1], handle.dist[1][l], handle.ratio[1][l], dw_in1.dv,
        dw_in1.ldvs_h[0], dw_in1.ldvs_h[1], dw_in2.dv, dw_in2.ldvs_h[0],
        dw_in2.ldvs_h[1], dw_out.dv, dw_out.ldvs_h[0], dw_out.ldvs_h[1],
        queue_idx,
        handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

    // handle.sync(0);
    verify_matrix_cuda(
        handle.dofs[2][l], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
        dw_out.dv, dw_out.ldvs_h[0], dw_out.ldvs_h[1], dw_out.ldvs_h[0],
        prefix + "lpk_reo_2_3d" + "_level_" + std::to_string(l), store, verify);

    if (debug_print) {
      printf("after mass_trans_multiply_2_cpt\n");
      print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l + 1],
                        handle.dofs[0][l + 1], dw_out.dv, dw_out.ldvs_h[0],
                        dw_out.ldvs_h[1], dw_out.ldvs_h[0]);
    }
  }

  if (D == 3) {
    dw_in1 = dw_out;
    dw_in1.resize(
        {handle.dofs[0][l + 1], handle.dofs[1][l + 1], handle.dofs[2][l + 1]});
    dw_in2 = dw_out;
    dw_in2.offset({0, 0, handle.dofs[2][l + 1]});
    dw_in2.resize({handle.dofs[0][l + 1], handle.dofs[1][l + 1],
                   handle.dofs[2][l] - handle.dofs[2][l + 1]});
    dw_out.offset({handle.dofs[0][l + 1], handle.dofs[1][l + 1], 0});
    dw_out.resize(
        {handle.dofs[0][l + 1], handle.dofs[1][l + 1], handle.dofs[2][l + 1]});

    lpk_reo_3_3d(
        handle, handle.dofs[2][l], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
        handle.dofs[2][l + 1], handle.dist[2][l], handle.ratio[2][l], dw_in1.dv,
        dw_in1.ldvs_h[0], dw_in1.ldvs_h[1], dw_in2.dv, dw_in2.ldvs_h[0],
        dw_in2.ldvs_h[1], dw_out.dv, dw_out.ldvs_h[0], dw_out.ldvs_h[1],
        queue_idx,
        handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

    // handle.sync(0);
    verify_matrix_cuda(
        handle.dofs[2][l + 1], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
        dw_out.dv, dw_out.ldvs_h[0], dw_out.ldvs_h[1], dw_out.ldvs_h[0],
        prefix + "lpk_reo_3_3d" + "_level_" + std::to_string(l), store, verify);

    if (debug_print) {
      printf("after mass_trans_multiply_3_cpt\n");
      print_matrix_cuda(handle.dofs[2][l + 1], handle.dofs[1][l + 1],
                        handle.dofs[0][l + 1], dw_out.dv, dw_out.ldvs_h[0],
                        dw_out.ldvs_h[1], dw_out.ldvs_h[0]);
    }
  }

  if (D >= 1) {
    ipk_1_3d(handle, handle.dofs[2][l + 1], handle.dofs[1][l + 1],
             handle.dofs[0][l + 1], handle.am[0][l + 1], handle.bm[0][l + 1],
             handle.dist[0][l + 1], dw_out.dv, dw_out.ldvs_h[0],
             dw_out.ldvs_h[1], queue_idx,
             handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);

    // //handle.sync(0);
    verify_matrix_cuda(
        handle.dofs[2][l + 1], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
        dw_out.dv, dw_out.ldvs_h[0], dw_out.ldvs_h[1], dw_out.ldvs_h[0],
        prefix + "ipk_1_3d" + "_level_" + std::to_string(l), store, verify);

    if (debug_print) {
      printf("after solve_tridiag_1_cpt\n");
      print_matrix_cuda(handle.dofs[2][l + 1], handle.dofs[1][l + 1],
                        handle.dofs[0][l + 1], dw_out.dv, dw_out.ldvs_h[0],
                        dw_out.ldvs_h[1], dw_out.ldvs_h[0]);
    }
  }
  if (D >= 2) {
    ipk_2_3d(handle, handle.dofs[2][l + 1], handle.dofs[1][l + 1],
             handle.dofs[0][l + 1], handle.am[1][l + 1], handle.bm[1][l + 1],
             handle.dist[1][l + 1], dw_out.dv, dw_out.ldvs_h[0],
             dw_out.ldvs_h[1], queue_idx,
             handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);

    // handle.sync(0);
    verify_matrix_cuda(
        handle.dofs[2][l + 1], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
        dw_out.dv, dw_out.ldvs_h[0], dw_out.ldvs_h[1], dw_out.ldvs_h[0],
        prefix + "ipk_2_3d" + "_level_" + std::to_string(l), store, verify);

    if (debug_print) {
      printf("after solve_tridiag_2_cpt\n");
      print_matrix_cuda(handle.dofs[2][l + 1], handle.dofs[1][l + 1],
                        handle.dofs[0][l + 1], dw_out.dv, dw_out.ldvs_h[0],
                        dw_out.ldvs_h[1], dw_out.ldvs_h[0]);
    }
  }

  if (D == 3) {
    ipk_3_3d(handle, handle.dofs[2][l + 1], handle.dofs[1][l + 1],
             handle.dofs[0][l + 1], handle.am[2][l + 1], handle.bm[2][l + 1],
             handle.dist[2][l + 1], dw_out.dv, dw_out.ldvs_h[0],
             dw_out.ldvs_h[1], queue_idx,
             handle.auto_tuning_ts3[handle.arch][handle.precision][range_lp1]);

    // handle.sync(0);
    verify_matrix_cuda(
        handle.dofs[2][l + 1], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
        dw_out.dv, dw_out.ldvs_h[0], dw_out.ldvs_h[1], dw_out.ldvs_h[0],
        prefix + "ipk_3_3d" + "_level_" + std::to_string(l), store, verify);

    if (debug_print) {
      printf("after solve_tridiag_3_cpt\n");
      print_matrix_cuda(handle.dofs[2][l + 1], handle.dofs[1][l + 1],
                        handle.dofs[0][l + 1], dw_out.dv, dw_out.ldvs_h[0],
                        dw_out.ldvs_h[1], dw_out.ldvs_h[0]);
    }
  }

  // final correction output
  dcorrection = dw_out;
}

template <DIM D, typename T>
void calc_coefficients_nd(Handle<D, T> &handle, SubArray<D, T> dinput1,
                          SubArray<D, T> dinput2, SubArray<D, T> &doutput,
                          SIZE l, int queue_idx) {

  int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
  int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(handle.shapes_h[0][d]) + "_";
  // printf("interpolate 1-3D\n");

  SubArray<D, T> dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf,
      dcoeff_rc, dcoeff_rcf;

  DIM curr_dims[3];

  int unprocessed_idx = 0;
  curr_dims[0] = 0;
  curr_dims[1] = 1;
  curr_dims[2] = 2;
  dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  calc_coeff_pointers(handle, curr_dims, l, doutput, dcoarse, dcoeff_f,
                      dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                      dcoeff_rcf);

  gpk_reo<D, 3, T, true, false, 1>(
      handle, handle.shapes_h[l], handle.shapes_d[l], handle.shapes_d[l + 1],
      dinput1.ldvs_d, doutput.ldvs_d, handle.unprocessed_n[unprocessed_idx],
      handle.unprocessed_dims_d[unprocessed_idx], curr_dims[2], curr_dims[1],
      curr_dims[0], handle.ratio[curr_dims[2]][l],
      handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], dinput1.dv,
      dinput1.lddv1, dinput1.lddv2, dcoarse.dv, dcoarse.lddv1, dcoarse.lddv2,
      // null, lddv1, lddv2,
      dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
      // null, lddv1, lddv2,
      dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
      // null, lddv1, lddv2,
      dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
      // null, lddv1, lddv2,
      dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
      // null, lddv1, lddv2,
      dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
      // null, lddv1, lddv2,
      dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
      // null, lddv1, lddv2,
      dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
      // null, lddv1, lddv2,
      queue_idx, handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

  for (DIM d = 3; d < D; d += 2) {
    // copy back to input1 for interpolation again
    lwpk<D, T, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l], doutput.dv,
                     doutput.ldvs_d, dinput1.dv, dinput1.ldvs_d, queue_idx);

    // printf("interpolate %u-%uD\n", d+1, d+2);
    curr_dims[0] = 0;
    curr_dims[1] = d;
    curr_dims[2] = d + 1;
    dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    calc_coeff_pointers(handle, curr_dims, l, doutput, dcoarse, dcoeff_f,
                        dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                        dcoeff_rcf);

    // printf("lddv1(%d), lddv2(%d), lddw1(%d), lddw2(%d)\n", lddv1, lddv2,
    // lddw1, lddw2);
    if (D - d == 1) {
      unprocessed_idx += 1;
      gpk_reo<D, 2, T, true, false, 2>(
          handle, handle.shapes_h[l], handle.shapes_d[l],
          handle.shapes_d[l + 1], dinput1.ldvs_d, doutput.ldvs_d,
          handle.unprocessed_n[unprocessed_idx],
          handle.unprocessed_dims_d[unprocessed_idx], curr_dims[2],
          curr_dims[1], curr_dims[0], handle.ratio[curr_dims[2]][l],
          handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l],
          dinput1.dv, dinput1.lddv1, dinput1.lddv2, dcoarse.dv, dcoarse.lddv1,
          dcoarse.lddv2,
          // null, lddv1, lddv2,
          dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
          // null, lddv1, lddv2,
          dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
          // null, lddv1, lddv2,
          dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
          // null, lddv1, lddv2,
          dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
          // null, lddv1, lddv2,
          dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
          // null, lddv1, lddv2,
          dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
          // null, lddv1, lddv2,
          dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
          // null, lddv1, lddv2,
          queue_idx,
          handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
    } else { // D - d >= 2
      unprocessed_idx += 2;
      gpk_reo<D, 3, T, true, false, 2>(
          handle, handle.shapes_h[l], handle.shapes_d[l],
          handle.shapes_d[l + 1], dinput1.ldvs_d, doutput.ldvs_d,
          handle.unprocessed_n[unprocessed_idx],
          handle.unprocessed_dims_d[unprocessed_idx], curr_dims[2],
          curr_dims[1], curr_dims[0], handle.ratio[curr_dims[2]][l],
          handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l],
          dinput1.dv, dinput1.lddv1, dinput1.lddv2, dcoarse.dv, dcoarse.lddv1,
          dcoarse.lddv2,
          // null, lddv1, lddv2,
          dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
          // null, lddv1, lddv2,
          dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
          // null, lddv1, lddv2,
          dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
          // null, lddv1, lddv2,
          dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
          // null, lddv1, lddv2,
          dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
          // null, lddv1, lddv2,
          dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
          // null, lddv1, lddv2,
          dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
          // null, lddv1, lddv2,
          queue_idx,
          handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
    }
  }

  if (debug_print) { // debug
    printf(" after interpolation\n");
    for (int k = 0; k < doutput.shape[4]; k++) {
      for (int j = 0; j < doutput.shape[3]; j++) {
        printf("i,j = %d,%d\n", k, j);
        print_matrix_cuda(
            doutput.shape[2], doutput.shape[1], doutput.shape[0],
            doutput.dv +
                k * doutput.ldvs_h[0] * doutput.ldvs_h[1] * doutput.ldvs_h[2] *
                    doutput.ldvs_h[3] +
                j * doutput.ldvs_h[0] * doutput.ldvs_h[1] * doutput.ldvs_h[2],
            doutput.ldvs_h[0], doutput.ldvs_h[1], doutput.ldvs_h[0]);
      }
    }
  } // debug

  unprocessed_idx = 0;
  // printf("reorder 1-3D\n");
  curr_dims[0] = 0;
  curr_dims[1] = 1;
  curr_dims[2] = 2;
  dinput2.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  dinput1.project(curr_dims[0], curr_dims[1],
                  curr_dims[2]); // reuse input1 as temp output
  calc_coeff_pointers(handle, curr_dims, l, dinput1, dcoarse, dcoeff_f,
                      dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                      dcoeff_rcf);

  gpk_reo<D, 3, T, false, false, 1>(
      handle, handle.shapes_h[l], handle.shapes_d[l], handle.shapes_d[l + 1],
      dinput2.ldvs_d, dinput1.ldvs_d, handle.unprocessed_n[unprocessed_idx],
      handle.unprocessed_dims_d[unprocessed_idx], curr_dims[2], curr_dims[1],
      curr_dims[0], handle.ratio[curr_dims[2]][l],
      handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], dinput2.dv,
      dinput2.lddv1, dinput2.lddv2, dcoarse.dv, dcoarse.lddv1, dcoarse.lddv2,
      // null, lddv1, lddv2,
      dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
      // null, lddv1, lddv2,
      dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
      // null, lddv1, lddv2,
      dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
      // null, lddv1, lddv2,
      dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
      // null, lddv1, lddv2,
      dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
      // null, lddv1, lddv2,
      dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
      // null, lddv1, lddv2,
      dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
      // null, lddv1, lddv2,
      queue_idx, handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

  DIM D_reduced = D % 2 == 0 ? D - 1 : D - 2;
  for (DIM d = 3; d < D_reduced; d += 2) {
    // copy back to input2 for reordering again
    lwpk<D, T, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l], dinput1.dv,
                     dinput1.ldvs_d, dinput2.dv, dinput2.ldvs_d, queue_idx);

    // printf("reorder %u-%uD\n", d+1, d+2);
    curr_dims[0] = 0;
    curr_dims[1] = d;
    curr_dims[2] = d + 1;
    dinput2.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    dinput1.project(curr_dims[0], curr_dims[1],
                    curr_dims[2]); // reuse input1 as temp output
    calc_coeff_pointers(handle, curr_dims, l, dinput1, dcoarse, dcoeff_f,
                        dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                        dcoeff_rcf);
    unprocessed_idx += 2;
    gpk_reo<D, 3, T, false, false, 2>(
        handle, handle.shapes_h[l], handle.shapes_d[l], handle.shapes_d[l + 1],
        dinput2.ldvs_d, dinput1.ldvs_d, handle.unprocessed_n[unprocessed_idx],
        handle.unprocessed_dims_d[unprocessed_idx], curr_dims[2], curr_dims[1],
        curr_dims[0], handle.ratio[curr_dims[2]][l],
        handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l],
        dinput2.dv, dinput2.lddv1, dinput2.lddv2, dcoarse.dv, dcoarse.lddv1,
        dcoarse.lddv2,
        // null, lddv1, lddv2,
        dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
        // null, lddv1, lddv2,
        dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
        // null, lddv1, lddv2,
        dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
        // null, lddv1, lddv2,
        dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
        // null, lddv1, lddv2,
        queue_idx,
        handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
  }

  // printf("calc coeff %u-%dD\n", D_reduced+1, D_reduced+2);
  curr_dims[0] = 0;
  curr_dims[1] = D_reduced;
  curr_dims[2] = D_reduced + 1;
  dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  doutput.project(curr_dims[0], curr_dims[1],
                  curr_dims[2]); // reuse input1 as temp output
  calc_coeff_pointers(handle, curr_dims, l, doutput, dcoarse, dcoeff_f,
                      dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                      dcoeff_rcf);
  if (D - D_reduced == 1) {
    // unprocessed_dims.pop_back();
    unprocessed_idx += 1;
    gpk_reo<D, 2, T, false, true, 2>(
        handle, handle.shapes_h[l], handle.shapes_d[l], handle.shapes_d[l + 1],
        dinput1.ldvs_d, doutput.ldvs_d, handle.unprocessed_n[unprocessed_idx],
        handle.unprocessed_dims_d[unprocessed_idx], curr_dims[2], curr_dims[1],
        curr_dims[0], handle.ratio[curr_dims[2]][l],
        handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l],
        dinput1.dv, dinput1.lddv1, dinput1.lddv2, dcoarse.dv, dcoarse.lddv1,
        dcoarse.lddv2,
        // null, lddv1, lddv2,
        dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
        // null, lddv1, lddv2,
        dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
        // null, lddv1, lddv2,
        dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
        // null, lddv1, lddv2,
        dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
        // null, lddv1, lddv2,
        queue_idx,
        handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

  } else { // D-D_reduced == 2
    unprocessed_idx += 2;
    gpk_reo<D, 3, T, false, true, 2>(
        handle, handle.shapes_h[l], handle.shapes_d[l], handle.shapes_d[l + 1],
        dinput1.ldvs_d, doutput.ldvs_d, handle.unprocessed_n[unprocessed_idx],
        handle.unprocessed_dims_d[unprocessed_idx], curr_dims[2], curr_dims[1],
        curr_dims[0], handle.ratio[curr_dims[2]][l],
        handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l],
        dinput1.dv, dinput1.lddv1, dinput1.lddv2, dcoarse.dv, dcoarse.lddv1,
        dcoarse.lddv2,
        // null, lddv1, lddv2,
        dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
        // null, lddv1, lddv2,
        dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
        // null, lddv1, lddv2,
        dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
        // null, lddv1, lddv2,
        dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
        // null, lddv1, lddv2,
        queue_idx,
        handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
  }

  if (debug_print) { // debug
    printf(" after calc coeff\n");
    for (int k = 0; k < doutput.shape[4]; k++) {
      for (int j = 0; j < doutput.shape[3]; j++) {
        printf("i,j = %d,%d\n", k, j);
        print_matrix_cuda(
            doutput.shape[2], doutput.shape[1], doutput.shape[0],
            doutput.dv +
                k * doutput.ldvs_h[0] * doutput.ldvs_h[1] * doutput.ldvs_h[2] *
                    doutput.ldvs_h[3] +
                j * doutput.ldvs_h[0] * doutput.ldvs_h[1] * doutput.ldvs_h[2],
            doutput.ldvs_h[0], doutput.ldvs_h[1], doutput.ldvs_h[0]);
      }
    }
  } // debug
}

template <DIM D, typename T>
void coefficients_restore_nd(Handle<D, T> &handle, SubArray<D, T> dinput1,
                             SubArray<D, T> dinput2, SubArray<D, T> &doutput,
                             SIZE l, int queue_idx) {

  int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
  int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(handle.shapes_h[0][d]) + "_";

  SubArray<D, T> dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf,
      dcoeff_rc, dcoeff_rcf;

  DIM curr_dims[3];
  int unprocessed_idx = 0;

  // printf("interpolate-restore 1-3D\n");
  curr_dims[0] = 0;
  curr_dims[1] = 1;
  curr_dims[2] = 2;
  dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  calc_coeff_pointers(handle, curr_dims, l, dinput1, dcoarse, dcoeff_f,
                      dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                      dcoeff_rcf);

  gpk_rev<D, 3, T, true, false, 1>(
      handle, handle.shapes_h[l], handle.shapes_d[l], handle.shapes_d[l + 1],
      doutput.ldvs_d, dinput1.ldvs_d, handle.unprocessed_n[unprocessed_idx],
      handle.unprocessed_dims_d[unprocessed_idx], curr_dims[2], curr_dims[1],
      curr_dims[0], handle.ratio[curr_dims[2]][l],
      handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], doutput.dv,
      doutput.lddv1, doutput.lddv2, dcoarse.dv, dcoarse.lddv1, dcoarse.lddv2,
      // null, lddv1, lddv2,
      dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
      // null, lddv1, lddv2,
      dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
      // null, lddv1, lddv2,
      dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
      // null, lddv1, lddv2,
      dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
      // null, lddv1, lddv2,
      dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
      // null, lddv1, lddv2,
      dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
      // null, lddv1, lddv2,
      dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
      // null, lddv1, lddv2,
      0, 0, 0, handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l],
      handle.dofs[curr_dims[0]][l], queue_idx,
      handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

  for (DIM d = 3; d < D; d += 2) {
    lwpk<D, T, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l], doutput.dv,
                     doutput.ldvs_d, dinput1.dv, dinput1.ldvs_d, queue_idx);

    // printf("interpolate-restore %u-%uD\n", d+1, d+2);
    curr_dims[0] = 0;
    curr_dims[1] = d;
    curr_dims[2] = d + 1;
    dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    calc_coeff_pointers(handle, curr_dims, l, dinput1, dcoarse, dcoeff_f,
                        dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                        dcoeff_rcf);

    if (D - d == 1) {
      unprocessed_idx += 1;
      gpk_rev<D, 2, T, true, false, 2>(
          handle, handle.shapes_h[l], handle.shapes_d[l],
          handle.shapes_d[l + 1], doutput.ldvs_d, dinput1.ldvs_d,
          handle.unprocessed_n[unprocessed_idx],
          handle.unprocessed_dims_d[unprocessed_idx], curr_dims[2],
          curr_dims[1], curr_dims[0], handle.ratio[curr_dims[2]][l],
          handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l],
          doutput.dv, doutput.lddv1, doutput.lddv2, dcoarse.dv, dcoarse.lddv1,
          dcoarse.lddv2,
          // null, lddv1, lddv2,
          dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
          // null, lddv1, lddv2,
          dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
          // null, lddv1, lddv2,
          dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
          // null, lddv1, lddv2,
          dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
          // null, lddv1, lddv2,
          dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
          // null, lddv1, lddv2,
          dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
          // null, lddv1, lddv2,
          dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
          // null, lddv1, lddv2,
          0, 0, 0, handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l],
          handle.dofs[curr_dims[0]][l], queue_idx,
          handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
    } else { // D - d >= 2
      unprocessed_idx += 2;
      gpk_rev<D, 3, T, true, false, 2>(
          handle, handle.shapes_h[l], handle.shapes_d[l],
          handle.shapes_d[l + 1], doutput.ldvs_d, dinput1.ldvs_d,
          handle.unprocessed_n[unprocessed_idx],
          handle.unprocessed_dims_d[unprocessed_idx], curr_dims[2],
          curr_dims[1], curr_dims[0], handle.ratio[curr_dims[2]][l],
          handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l],
          doutput.dv, doutput.lddv1, doutput.lddv2, dcoarse.dv, dcoarse.lddv1,
          dcoarse.lddv2,
          // null, lddv1, lddv2,
          dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
          // null, lddv1, lddv2,
          dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
          // null, lddv1, lddv2,
          dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
          // null, lddv1, lddv2,
          dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
          // null, lddv1, lddv2,
          dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
          // null, lddv1, lddv2,
          dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
          // null, lddv1, lddv2,
          dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
          // null, lddv1, lddv2,
          0, 0, 0, handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l],
          handle.dofs[curr_dims[0]][l], queue_idx,
          handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
    }
  }
  // Done interpolation-restore on doutput

  if (debug_print) { // debug
    printf("After interpolation reverse-reorder\n");
    for (int k = 0; k < doutput.shape[4]; k++) {
      for (int j = 0; j < doutput.shape[3]; j++) {
        printf("i,j = %d,%d\n", k, j);
        print_matrix_cuda(
            doutput.shape[2], doutput.shape[1], doutput.shape[0],
            doutput.dv +
                k * doutput.ldvs_h[0] * doutput.ldvs_h[1] * doutput.ldvs_h[2] *
                    doutput.ldvs_h[3] +
                j * doutput.ldvs_h[0] * doutput.ldvs_h[1] * doutput.ldvs_h[2],
            doutput.ldvs_h[0], doutput.ldvs_h[1], doutput.ldvs_h[0]);
      }
    }
  } // debug

  unprocessed_idx = 0;

  // printf("reorder-restore 1-3D\n");
  curr_dims[0] = 0;
  curr_dims[1] = 1;
  curr_dims[2] = 2;
  dinput2.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  dinput1.project(curr_dims[0], curr_dims[1],
                  curr_dims[2]); // reuse input1 as temp space
  calc_coeff_pointers(handle, curr_dims, l, dinput2, dcoarse, dcoeff_f,
                      dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                      dcoeff_rcf);

  gpk_rev<D, 3, T, false, false, 1>(
      handle, handle.shapes_h[l], handle.shapes_d[l], handle.shapes_d[l + 1],
      dinput1.ldvs_d, dinput2.ldvs_d, handle.unprocessed_n[unprocessed_idx],
      handle.unprocessed_dims_d[unprocessed_idx], curr_dims[2], curr_dims[1],
      curr_dims[0], handle.ratio[curr_dims[2]][l],
      handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], dinput1.dv,
      dinput1.lddv1, dinput1.lddv2, dcoarse.dv, dcoarse.lddv1, dcoarse.lddv2,
      // null, lddv1, lddv2,
      dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
      // null, lddv1, lddv2,
      dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
      // null, lddv1, lddv2,
      dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
      // null, lddv1, lddv2,
      dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
      // null, lddv1, lddv2,
      dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
      // null, lddv1, lddv2,
      dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
      // null, lddv1, lddv2,
      dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
      // null, lddv1, lddv2,
      0, 0, 0, handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l],
      handle.dofs[curr_dims[0]][l], queue_idx,
      handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

  DIM D_reduced = D % 2 == 0 ? D - 1 : D - 2;
  for (DIM d = 3; d < D_reduced; d += 2) {
    // printf("reorder-reverse\n");
    // copy back to input2 for reordering again
    lwpk<D, T, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l], dinput1.dv,
                     dinput1.ldvs_d, dinput2.dv, dinput2.ldvs_d, queue_idx);
    // printf("reorder-restore %u-%uD\n", d+1, d+2);
    curr_dims[0] = 0;
    curr_dims[1] = d;
    curr_dims[2] = d + 1;
    dinput2.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    dinput1.project(curr_dims[0], curr_dims[1],
                    curr_dims[2]); // reuse input1 as temp output
    calc_coeff_pointers(handle, curr_dims, l, dinput2, dcoarse, dcoeff_f,
                        dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                        dcoeff_rcf);

    unprocessed_idx += 2;
    gpk_rev<D, 3, T, false, false, 2>(
        handle, handle.shapes_h[l], handle.shapes_d[l], handle.shapes_d[l + 1],
        dinput1.ldvs_d, dinput2.ldvs_d, handle.unprocessed_n[unprocessed_idx],
        handle.unprocessed_dims_d[unprocessed_idx], curr_dims[2], curr_dims[1],
        curr_dims[0], handle.ratio[curr_dims[2]][l],
        handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l],
        dinput1.dv, dinput1.lddv1, dinput1.lddv2, dcoarse.dv, dcoarse.lddv1,
        dcoarse.lddv2,
        // null, lddv1, lddv2,
        dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
        // null, lddv1, lddv2,
        dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
        // null, lddv1, lddv2,
        dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
        // null, lddv1, lddv2,
        dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
        // null, lddv1, lddv2,
        0, 0, 0, handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l],
        handle.dofs[curr_dims[0]][l], queue_idx,
        handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
  }

  // printf("coeff-restore %u-%dD\n", D_reduced+1, D_reduced+2);
  curr_dims[0] = 0;
  curr_dims[1] = D_reduced;
  curr_dims[2] = D_reduced + 1;
  dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  calc_coeff_pointers(handle, curr_dims, l, dinput1, dcoarse, dcoeff_f,
                      dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                      dcoeff_rcf);

  if (D - D_reduced == 1) {
    // printf("coeff-restore %u-%dD\n", D_reduced+1, D_reduced+1);
    unprocessed_idx += 1;
    gpk_rev<D, 2, T, false, true, 2>(
        handle, handle.shapes_h[l], handle.shapes_d[l], handle.shapes_d[l + 1],
        doutput.ldvs_d, dinput1.ldvs_d, handle.unprocessed_n[unprocessed_idx],
        handle.unprocessed_dims_d[unprocessed_idx], curr_dims[2], curr_dims[1],
        curr_dims[0], handle.ratio[curr_dims[2]][l],
        handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l],
        doutput.dv, doutput.lddv1, doutput.lddv2, dcoarse.dv, dcoarse.lddv1,
        dcoarse.lddv2,
        // null, lddv1, lddv2,
        dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
        // null, lddv1, lddv2,
        dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
        // null, lddv1, lddv2,
        dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
        // null, lddv1, lddv2,
        dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
        // null, lddv1, lddv2,
        0, 0, 0, handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l],
        handle.dofs[curr_dims[0]][l], queue_idx,
        handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
  } else { // D - D_reduced >= 2
    // printf("coeff-restore %u-%dD\n", D_reduced+1, D_reduced+2);
    unprocessed_idx += 2;
    gpk_rev<D, 3, T, false, true, 2>(
        handle, handle.shapes_h[l], handle.shapes_d[l], handle.shapes_d[l + 1],
        doutput.ldvs_d, dinput1.ldvs_d, handle.unprocessed_n[unprocessed_idx],
        handle.unprocessed_dims_d[unprocessed_idx], curr_dims[2], curr_dims[1],
        curr_dims[0], handle.ratio[curr_dims[2]][l],
        handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l],
        doutput.dv, doutput.lddv1, doutput.lddv2, dcoarse.dv, dcoarse.lddv1,
        dcoarse.lddv2,
        // null, lddv1, lddv2,
        dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2,
        // null, lddv1, lddv2,
        dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2,
        // null, lddv1, lddv2,
        dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2,
        // null, lddv1, lddv2,
        dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2,
        // null, lddv1, lddv2,
        dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2,
        // null, lddv1, lddv2,
        0, 0, 0, handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l],
        handle.dofs[curr_dims[0]][l], queue_idx,
        handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
  }

  if (debug_print) { // debug
    printf("After coeff restore\n");
    for (int k = 0; k < doutput.shape[4]; k++) {
      for (int j = 0; j < doutput.shape[3]; j++) {
        printf("i,j = %d,%d\n", k, j);
        print_matrix_cuda(
            doutput.shape[2], doutput.shape[1], doutput.shape[0],
            doutput.dv +
                k * doutput.ldvs_h[0] * doutput.ldvs_h[1] * doutput.ldvs_h[2] *
                    doutput.ldvs_h[3] +
                j * doutput.ldvs_h[0] * doutput.ldvs_h[1] * doutput.ldvs_h[2],
            doutput.ldvs_h[0], doutput.ldvs_h[1], doutput.ldvs_h[0]);
      }
    }
  } // debug
}

template <DIM D, typename T>
void calc_correction_nd(Handle<D, T> &handle, SubArray<D, T> dcoeff,
                        SubArray<D, T> &dcorrection, SIZE l, int queue_idx) {
  int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
  int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(handle.shapes_h[0][d]) + "_";

  SubArray<D, T> dw_in1 = dcoeff;
  SubArray<D, T> dw_in2 = dcoeff;
  SubArray<D, T> dw_out = dcorrection;

  // start correction calculation
  int prev_dim_r, prev_dim_c, prev_dim_f;
  int curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;

  dw_in1.resize(curr_dim_f, handle.dofs[curr_dim_f][l + 1]);
  dw_in2.offset(curr_dim_f, handle.dofs[curr_dim_f][l + 1]);
  dw_in2.resize(curr_dim_f,
                handle.dofs[curr_dim_f][l] - handle.dofs[curr_dim_f][l + 1]);
  dw_out.resize(curr_dim_f, handle.dofs[curr_dim_f][l + 1]);

  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

  // printf("mass trans 1D\n");
  lpk_reo_1<D, T>(
      handle, handle.shapes_h[l], handle.shapes_h[l + 1], handle.shapes_d[l],
      handle.shapes_d[l + 1], dw_in1.ldvs_d, dw_out.ldvs_d,
      handle.processed_n[0], handle.processed_dims_h[0],
      handle.processed_dims_d[0], curr_dim_r, curr_dim_c, curr_dim_f,
      handle.dist[curr_dim_f][l], handle.ratio[curr_dim_f][l], dw_in1.dv,
      dw_in1.lddv1, dw_in1.lddv2, dw_in2.dv, dw_in2.lddv1, dw_in2.lddv2,
      dw_out.dv, dw_out.lddv1, dw_out.lddv2, queue_idx,
      handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

  if (debug_print) { // debug
    printf("decomposition: after MR-1D[%d]\n", l);
    for (int i = 0; i < dw_out.shape[3]; i++) {
      printf("i = %d\n", i);
      print_matrix_cuda(dw_out.shape[2], dw_out.shape[1], dw_out.shape[0],
                        dw_out.dv + i * dw_out.ldvs_h[0] * dw_out.ldvs_h[1] *
                                        dw_out.ldvs_h[2],
                        dw_out.ldvs_h[0], dw_out.ldvs_h[1], dw_out.ldvs_h[0]);
    }
  }

  // mass trans 2D
  prev_dim_f = curr_dim_f;
  prev_dim_c = curr_dim_c;
  prev_dim_r = curr_dim_r;
  curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;

  dw_in1 = dw_out;
  dw_in2 = dw_out;
  dw_in1.resize(curr_dim_c, handle.dofs[curr_dim_c][l + 1]);
  dw_in2.offset(curr_dim_c, handle.dofs[curr_dim_c][l + 1]);
  dw_in2.resize(curr_dim_c,
                handle.dofs[curr_dim_c][l] - handle.dofs[curr_dim_c][l + 1]);
  dw_out.offset(prev_dim_f, handle.dofs[curr_dim_f][l + 1]);
  dw_out.resize(curr_dim_c, handle.dofs[curr_dim_c][l + 1]);

  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

  // printf("mass trans 2D\n");
  lpk_reo_2<D, T>(
      handle, handle.shapes_h[l], handle.shapes_h[l + 1], handle.shapes_d[l],
      handle.shapes_d[l + 1], dw_in1.ldvs_d, dw_out.ldvs_d,
      handle.processed_n[1], handle.processed_dims_h[1],
      handle.processed_dims_d[1], curr_dim_r, curr_dim_c, curr_dim_f,
      handle.dist[curr_dim_c][l], handle.ratio[curr_dim_c][l], dw_in1.dv,
      dw_in1.lddv1, dw_in1.lddv2, dw_in2.dv, dw_in2.lddv1, dw_in2.lddv2,
      dw_out.dv, dw_out.lddv1, dw_out.lddv2, queue_idx,
      handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

  if (debug_print) { // debug
    printf("decomposition: after MR-2D[%d]\n", l);
    for (int i = 0; i < dw_out.shape[3]; i++) {
      printf("i = %d\n", i);
      print_matrix_cuda(dw_out.shape[2], dw_out.shape[1], dw_out.shape[0],
                        dw_out.dv + i * dw_out.ldvs_h[0] * dw_out.ldvs_h[1] *
                                        dw_out.ldvs_h[2],
                        dw_out.ldvs_h[0], dw_out.ldvs_h[1], dw_out.ldvs_h[0]);
    }
  }

  // mass trans 3D

  prev_dim_f = curr_dim_f;
  prev_dim_c = curr_dim_c;
  prev_dim_r = curr_dim_r;
  curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;

  dw_in1 = dw_out;
  dw_in2 = dw_out;
  dw_in1.resize(curr_dim_r, handle.dofs[curr_dim_r][l + 1]);
  dw_in2.offset(curr_dim_r, handle.dofs[curr_dim_r][l + 1]);
  dw_in2.resize(curr_dim_r,
                handle.dofs[curr_dim_r][l] - handle.dofs[curr_dim_r][l + 1]);
  dw_out.offset(prev_dim_c, handle.dofs[curr_dim_c][l + 1]);
  dw_out.resize(curr_dim_r, handle.dofs[curr_dim_r][l + 1]);

  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

  // printf("mass trans 3D\n");
  lpk_reo_3<D, T>(
      handle, handle.shapes_h[l], handle.shapes_h[l + 1], handle.shapes_d[l],
      handle.shapes_d[l + 1], dw_in1.ldvs_d, dw_out.ldvs_d,
      handle.processed_n[2], handle.processed_dims_h[2],
      handle.processed_dims_d[2], curr_dim_r, curr_dim_c, curr_dim_f,
      handle.dist[curr_dim_r][l], handle.ratio[curr_dim_r][l], dw_in1.dv,
      dw_in1.lddv1, dw_in1.lddv2, dw_in2.dv, dw_in2.lddv1, dw_in2.lddv2,
      dw_out.dv, dw_out.lddv1, dw_out.lddv2, queue_idx,
      handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

  if (debug_print) { // debug
    printf("decomposition: after MR-3D[%d]\n", l);
    for (int i = 0; i < dw_out.shape[3]; i++) {
      printf("i = %d\n", i);
      print_matrix_cuda(dw_out.shape[2], dw_out.shape[1], dw_out.shape[0],
                        dw_out.dv + i * dw_out.ldvs_h[0] * dw_out.ldvs_h[1] *
                                        dw_out.ldvs_h[2],
                        dw_out.ldvs_h[0], dw_out.ldvs_h[1], dw_out.ldvs_h[0]);
    }
  }

  // mass trans 4D+
  for (int i = 3; i < D; i++) {
    prev_dim_f = curr_dim_f;
    prev_dim_c = curr_dim_c;
    prev_dim_r = curr_dim_r;
    curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = i;
    dw_in1 = dw_out;
    dw_in2 = dw_out;
    dw_in1.resize(curr_dim_r, handle.dofs[curr_dim_r][l + 1]);
    dw_in2.offset(curr_dim_r, handle.dofs[curr_dim_r][l + 1]);
    dw_in2.resize(curr_dim_r,
                  handle.dofs[curr_dim_r][l] - handle.dofs[curr_dim_r][l + 1]);
    dw_out.offset(prev_dim_r, handle.dofs[prev_dim_r][l + 1]);
    dw_out.resize(curr_dim_r, handle.dofs[curr_dim_r][l + 1]);

    dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
    dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
    dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

    // printf("mass trans %dD\n", i+1);
    lpk_reo_3<D, T>(
        handle, handle.shapes_h[l], handle.shapes_h[l + 1], handle.shapes_d[l],
        handle.shapes_d[l + 1], dw_in1.ldvs_d, dw_out.ldvs_d,
        handle.processed_n[i], handle.processed_dims_h[i],
        handle.processed_dims_d[i], curr_dim_r, curr_dim_c, curr_dim_f,
        handle.dist[curr_dim_r][l], handle.ratio[curr_dim_r][l], dw_in1.dv,
        dw_in1.lddv1, dw_in1.lddv2, dw_in2.dv, dw_in2.lddv1, dw_in2.lddv2,
        dw_out.dv, dw_out.lddv1, dw_out.lddv2, queue_idx,
        handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

    if (debug_print) { // debug
      printf("decomposition: after MR-%dD[%d]\n", i + 1, l);
      for (int k = 0; k < dw_out.shape[4]; k++) {
        for (int j = 0; j < dw_out.shape[3]; j++) {
          printf("i,j = %d,%d\n", k, j);
          print_matrix_cuda(
              dw_out.shape[2], dw_out.shape[1], dw_out.shape[0],
              dw_out.dv +
                  k * dw_out.ldvs_h[0] * dw_out.ldvs_h[1] * dw_out.ldvs_h[2] *
                      dw_out.ldvs_h[3] +
                  j * dw_out.ldvs_h[0] * dw_out.ldvs_h[1] * dw_out.ldvs_h[2],
              dw_out.ldvs_h[0], dw_out.ldvs_h[1], dw_out.ldvs_h[0]);
        }
      }
    }
  }

  curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

  // printf("solve tridiag 1D\n");
  ipk_1<D, T>(handle, handle.shapes_h[l], handle.shapes_h[l + 1],
              handle.shapes_d[l], handle.shapes_d[l + 1], dw_out.ldvs_d,
              dw_out.ldvs_d, handle.processed_n[0], handle.processed_dims_h[0],
              handle.processed_dims_d[0], curr_dim_r, curr_dim_c, curr_dim_f,
              handle.am[curr_dim_f][l + 1], handle.bm[curr_dim_f][l + 1],
              handle.dist[curr_dim_f][l + 1], dw_out.dv, dw_out.lddv1,
              dw_out.lddv2, queue_idx,
              handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);

  if (debug_print) { // debug
    printf("decomposition: after TR-1D[%d]\n", l);
    for (int k = 0; k < dw_out.shape[4]; k++) {
      for (int j = 0; j < dw_out.shape[3]; j++) {
        printf("i,j = %d,%d\n", k, j);
        print_matrix_cuda(dw_out.shape[2], dw_out.shape[1], dw_out.shape[0],
                          dw_out.dv +
                              k * dw_out.ldvs_h[0] * dw_out.ldvs_h[1] *
                                  dw_out.ldvs_h[2] * dw_out.ldvs_h[3] +
                              j * dw_out.ldvs_h[0] * dw_out.ldvs_h[1] *
                                  dw_out.ldvs_h[2],
                          dw_out.ldvs_h[0], dw_out.ldvs_h[1], dw_out.ldvs_h[0]);
      }
    }
  } // debug

  curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

  // printf("solve tridiag 2D\n");
  ipk_2<D, T>(handle, handle.shapes_h[l], handle.shapes_h[l + 1],
              handle.shapes_d[l], handle.shapes_d[l + 1], dw_out.ldvs_d,
              dw_out.ldvs_d, handle.processed_n[1], handle.processed_dims_h[1],
              handle.processed_dims_d[1], curr_dim_r, curr_dim_c, curr_dim_f,
              handle.am[curr_dim_c][l + 1], handle.bm[curr_dim_c][l + 1],
              handle.dist[curr_dim_c][l + 1], dw_out.dv, dw_out.lddv1,
              dw_out.lddv2, queue_idx,
              handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);

  if (debug_print) { // debug
    printf("decomposition: after TR-2D[%d]\n", l);
    for (int k = 0; k < dw_out.shape[4]; k++) {
      for (int j = 0; j < dw_out.shape[3]; j++) {
        printf("i,j = %d,%d\n", k, j);
        print_matrix_cuda(dw_out.shape[2], dw_out.shape[1], dw_out.shape[0],
                          dw_out.dv +
                              k * dw_out.ldvs_h[0] * dw_out.ldvs_h[1] *
                                  dw_out.ldvs_h[2] * dw_out.ldvs_h[3] +
                              j * dw_out.ldvs_h[0] * dw_out.ldvs_h[1] *
                                  dw_out.ldvs_h[2],
                          dw_out.ldvs_h[0], dw_out.ldvs_h[1], dw_out.ldvs_h[0]);
      }
    }
  } // debug

  curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

  // printf("solve tridiag 3D\n");
  ipk_3<D, T>(handle, handle.shapes_h[l], handle.shapes_h[l + 1],
              handle.shapes_d[l], handle.shapes_d[l + 1], dw_out.ldvs_d,
              dw_out.ldvs_d, handle.processed_n[2], handle.processed_dims_h[2],
              handle.processed_dims_d[2], curr_dim_r, curr_dim_c, curr_dim_f,
              handle.am[curr_dim_r][l + 1], handle.bm[curr_dim_r][l + 1],
              handle.dist[curr_dim_r][l + 1], dw_out.dv, dw_out.lddv1,
              dw_out.lddv2, queue_idx,
              handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);

  if (debug_print) { // debug
    printf("decomposition: after TR-3D[%d]\n", l);
    for (int k = 0; k < dw_out.shape[4]; k++) {
      for (int j = 0; j < dw_out.shape[3]; j++) {
        printf("i,j = %d,%d\n", k, j);
        print_matrix_cuda(dw_out.shape[2], dw_out.shape[1], dw_out.shape[0],
                          dw_out.dv +
                              k * dw_out.ldvs_h[0] * dw_out.ldvs_h[1] *
                                  dw_out.ldvs_h[2] * dw_out.ldvs_h[3] +
                              j * dw_out.ldvs_h[0] * dw_out.ldvs_h[1] *
                                  dw_out.ldvs_h[2],
                          dw_out.ldvs_h[0], dw_out.ldvs_h[1], dw_out.ldvs_h[0]);
      }
    }
  } // debug

  // mass trans 4D+
  for (int i = 3; i < D; i++) {
    curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = i;
    dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
    dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
    dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);
    // printf("solve tridiag %dD\n", i+1);
    ipk_3<D, T>(
        handle, handle.shapes_h[l], handle.shapes_h[l + 1], handle.shapes_d[l],
        handle.shapes_d[l + 1], dw_out.ldvs_d, dw_out.ldvs_d,
        handle.processed_n[i], handle.processed_dims_h[i],
        handle.processed_dims_d[i], curr_dim_r, curr_dim_c, curr_dim_f,
        handle.am[curr_dim_r][l + 1], handle.bm[curr_dim_r][l + 1],
        handle.dist[curr_dim_r][l + 1], dw_out.dv, dw_out.lddv1, dw_out.lddv2,
        queue_idx,
        handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);
    if (debug_print) { // debug
      printf("decomposition: after TR-%dD[%d]\n", i + 1, l);
      for (int k = 0; k < dw_out.shape[4]; k++) {
        for (int j = 0; j < dw_out.shape[3]; j++) {
          printf("i,j = %d,%d\n", k, j);
          print_matrix_cuda(
              dw_out.shape[2], dw_out.shape[1], dw_out.shape[0],
              dw_out.dv +
                  k * dw_out.ldvs_h[0] * dw_out.ldvs_h[1] * dw_out.ldvs_h[2] *
                      dw_out.ldvs_h[3] +
                  j * dw_out.ldvs_h[0] * dw_out.ldvs_h[1] * dw_out.ldvs_h[2],
              dw_out.ldvs_h[0], dw_out.ldvs_h[1], dw_out.ldvs_h[0]);
        }
      }
    } // debug
  }

  dcorrection = dw_out;

  // { // debug
  //     printf("decomposition: after TR[%d]\n", l);
  //     for (int k = 0; k < dw_out.shape[4]; k++) {
  //       for (int j = 0; j < dw_out.shape[3]; j++) {
  //         printf("i,j = %d,%d\n", k,j);
  //         print_matrix_cuda(dw_out.shape[2], dw_out.shape[1],
  //         dw_out.shape[0],
  //                           dw_out.dv+k*dw_out.ldvs_h[0]*dw_out.ldvs_h[1]*dw_out.ldvs_h[2]*dw_out.ldvs_h[3]+j*dw_out.ldvs_h[0]*dw_out.ldvs_h[1]*dw_out.ldvs_h[2],
  //                           dw_out.ldvs_h[0], dw_out.ldvs_h[1],
  //                           dw_out.ldvs_h[0]);
  //       }
  //     }
  //   } //debug
}

template <DIM D, typename T>
void decompose(Handle<D, T> &handle, T *dv, std::vector<SIZE> ldvs_h,
               SIZE *ldvs_d, SIZE l_target, int queue_idx) {

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(handle.shapes_h[0][d]) + "_";
  // std::cout << prefix << std::endl;

  if (D <= 3) {
    for (int l = 0; l < l_target; ++l) {
      // printf("[gpu] l = %d\n", l);
      int stride = std::pow(2, l);
      int Cstride = stride * 2;
      int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
      int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);

      // for calculate corrections
      T *dw_out = NULL;
      T *dw_in1 = NULL;
      T *dw_in2 = NULL;

      // printf("range_l: %d, range_lp1: %d\n", range_l, range_lp1);

      if (debug_print) {
        printf("input v\n");
        print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
                          handle.dofs[0][l], dv, ldvs_h[0], ldvs_h[1],
                          ldvs_h[0]);
      }

      // verify_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
      //                    handle.dofs[0][l], dv, ldvs_h[0], ldvs_h[1],
      //                    ldvs_h[0], prefix + "begin" + "_level_" +
      //                    std::to_string(l), store, verify);
      lwpk<D, T, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l], dv,
                       ldvs_d, handle.dw, handle.ldws_d, queue_idx);

      SubArray<D, T> dinput(
          {handle.dofs[0][l], handle.dofs[1][l], handle.dofs[2][l]}, handle.dw,
          handle.ldws_h, handle.ldws_d);
      SubArray<D, T> doutput(
          {handle.dofs[0][l], handle.dofs[1][l], handle.dofs[2][l]}, dv, ldvs_h,
          ldvs_d);

      calc_coefficients_3d(handle, dinput, doutput, l, 0);

      SubArray<D, T> dcoeff(
          {handle.dofs[0][l], handle.dofs[1][l], handle.dofs[2][l]}, dv, ldvs_h,
          ldvs_d);
      SubArray<D, T> dcorrection(
          {handle.dofs[0][l] + 1, handle.dofs[1][l] + 1, handle.dofs[2][l] + 1},
          handle.dw, handle.ldws_h, handle.ldws_d);

      calc_correction_3d(handle, dcoeff, dcorrection, l, 0);

      lwpk<D, T, ADD>(handle, handle.shapes_h[l + 1], handle.shapes_d[l + 1],
                      dcorrection.dv, dcorrection.ldvs_d, dv, ldvs_d,
                      queue_idx);

      if (debug_print) {
        printf("after add\n");
        print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
                          handle.dofs[0][l], dv, ldvs_h[0], ldvs_h[1],
                          ldvs_h[0]);
      }

    } // end of loop

    if (debug_print) {
      printf("output of decomposition\n");
      print_matrix_cuda(handle.dofs[2][0], handle.dofs[1][0], handle.dofs[0][0],
                        dv, ldvs_h[0], ldvs_h[1], ldvs_h[0]);
    }
  }

  if (D > 3) {

    for (int l = 0; l < l_target; ++l) {
      // printf("[gpu] l = %d\n", l);
      int stride = std::pow(2, l);
      int Cstride = stride * 2;
      int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
      int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);
      bool f_padding = handle.dofs[0][l] % 2 == 0;
      bool c_padding = handle.dofs[1][l] % 2 == 0;
      bool r_padding = handle.dofs[2][l] % 2 == 0;

      DIM curr_dim_r, curr_dim_c, curr_dim_f;
      LENGTH lddv1, lddv2;
      LENGTH lddw1, lddw2;
      LENGTH lddb1, lddb2;

      int unprocessed_idx = 0;

      if (debug_print) { // debug
        printf("decomposition: before coeff\n");
        for (int i = 0; i < handle.dofs[3][0]; i++) {
          printf("i = %d\n", i);
          print_matrix_cuda(handle.dofs[2][0], handle.dofs[1][0],
                            handle.dofs[0][0],
                            dv + i * ldvs_h[0] * ldvs_h[1] * ldvs_h[2],
                            ldvs_h[0], ldvs_h[1], ldvs_h[0]);
        }
      }

      lwpk<D, T, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l], dv,
                       ldvs_d, handle.dw, handle.ldws_d, queue_idx);
      lwpk<D, T, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l], dv,
                       ldvs_d, handle.db, handle.ldbs_d, queue_idx);

      std::vector<SIZE> shape(handle.D_padded);
      for (DIM d = 0; d < handle.D_padded; d++)
        shape[d] = handle.shapes_h[l][d];

      SubArray<D, T> dinput1(shape, handle.dw, handle.ldws_h, handle.ldws_d);
      SubArray<D, T> dinput2(shape, handle.db, handle.ldbs_h, handle.ldbs_d);
      SubArray<D, T> doutput(shape, dv, ldvs_h, ldvs_d);

      calc_coefficients_nd(handle, dinput1, dinput2, doutput, l, queue_idx);

      // printf ("cjy3113\n");

      if (debug_print) { // debug
        printf("decomposition: after coeff[%d]\n", l);
        for (int k = 0; k < doutput.shape[4]; k++) {
          for (int j = 0; j < doutput.shape[3]; j++) {
            printf("i,j = %d,%d\n", k, j);
            print_matrix_cuda(
                doutput.shape[2], doutput.shape[1], doutput.shape[0],
                doutput.dv +
                    k * doutput.ldvs_h[0] * doutput.ldvs_h[1] *
                        doutput.ldvs_h[2] * doutput.ldvs_h[3] +
                    j * doutput.ldvs_h[0] * doutput.ldvs_h[1] *
                        doutput.ldvs_h[2],
                doutput.ldvs_h[0], doutput.ldvs_h[1], doutput.ldvs_h[0]);
          }
        }
      } // debug

      SubArray<D, T> dcoeff(shape, dv, ldvs_h, ldvs_d);
      SubArray<D, T> dcorrection(shape, handle.dw, handle.ldws_h,
                                 handle.ldws_d);

      calc_correction_nd(handle, dcoeff, dcorrection, l, 0);

      lwpk<D, T, ADD>(handle, handle.shapes_h[l + 1], handle.shapes_d[l + 1],
                      dcorrection.dv, dcorrection.ldvs_d, dv, ldvs_d,
                      queue_idx);
      if (debug_print) { // debug
        printf("decomposition: after apply correction[%d]\n", l);
        for (int k = 0; k < doutput.shape[4]; k++) {
          for (int j = 0; j < doutput.shape[3]; j++) {
            printf("i,j = %d,%d\n", k, j);
            print_matrix_cuda(
                doutput.shape[2], doutput.shape[1], doutput.shape[0],
                doutput.dv +
                    k * doutput.ldvs_h[0] * doutput.ldvs_h[1] *
                        doutput.ldvs_h[2] * doutput.ldvs_h[3] +
                    j * doutput.ldvs_h[0] * doutput.ldvs_h[1] *
                        doutput.ldvs_h[2],
                doutput.ldvs_h[0], doutput.ldvs_h[1], doutput.ldvs_h[0]);
          }
        }
      } // debug
    }

    //   { // debug
    //     lwpk<D, T, COPY>(handle, handle.shapes_h[0], handle.shapes_d[0], dv,
    //                        ldvs_d, handle.db, handle.ldbs_d, queue_idx);
    //     std::vector<SIZE> shape(D);
    //     for (DIM d = 0; d < D; d++) shape[d] = handle.shapes_h[0][d];
    //     SubArray<D, T> dcoeff(shape, handle.db, handle.ldbs_h,
    //     handle.ldbs_d); SubArray<D, T> doutput(shape, handle.dw,
    //     handle.ldws_h, handle.ldws_d); ReverseReorderGPU(handle, dcoeff,
    //     doutput, 0);

    // printf("decomposition: after applying correction\n");
    // for (int i = 0; i < handle.dofs[3][0]; i++) {
    //   printf("i = %d\n", i);
    //   print_matrix_cuda(handle.dofs[2][0], handle.dofs[1][0],
    //   handle.dofs[0][0],
    //                     dv+i*ldvs_h[0]*ldvs_h[1]*ldvs_h[2], ldvs_h[0],
    //                     ldvs_h[1], ldvs_h[0]);
    // }

    //     printf("after coeff reverse\n");
    //     for (int i = 0; i < handle.dofs[3][0]; i++) {
    //       printf("i = %d\n", i);
    //       print_matrix_cuda(handle.dofs[2][0], handle.dofs[1][0],
    //       handle.dofs[0][0],
    //                         doutput.dv+i*doutput.ldvs_h[0]*doutput.ldvs_h[1]*doutput.ldvs_h[2],
    //                         doutput.ldvs_h[0], doutput.ldvs_h[1],
    //                         doutput.ldvs_h[0]);
    //   }
    // }
  }
}

template <DIM D, typename T>
void recompose(Handle<D, T> &handle, T *dv, std::vector<SIZE> ldvs_h,
               SIZE *ldvs_d, SIZE l_target, int queue_idx) {

  if (D <= 3) {

    if (debug_print) {
      printf("input of recomposition\n");
      print_matrix_cuda(handle.dofs[2][0], handle.dofs[1][0], handle.dofs[0][0],
                        dv, ldvs_h[0], ldvs_h[1], ldvs_h[0]);
    }

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
      //                   dv, ldvs_h[0], ldvs_h[1], ldvs_h[0]);

      SubArray<D, T> dcoeff(
          {handle.dofs[0][l], handle.dofs[1][l], handle.dofs[2][l]}, dv, ldvs_h,
          ldvs_d);
      SubArray<D, T> dcorrection(
          {handle.dofs[0][l] + 1, handle.dofs[1][l] + 1, handle.dofs[2][l] + 1},
          handle.dw, handle.ldws_h, handle.ldws_d);

      calc_correction_3d(handle, dcoeff, dcorrection, l, 0);

      lwpk<D, T, SUBTRACT>(handle, handle.shapes_h[l + 1],
                           handle.shapes_d[l + 1], dcorrection.dv,
                           dcorrection.ldvs_d, dv, ldvs_d, queue_idx);

      SubArray<D, T> dinput(
          {handle.dofs[0][l], handle.dofs[1][l], handle.dofs[2][l]}, dv, ldvs_h,
          ldvs_d);

      SubArray<D, T> doutput(
          {handle.dofs[0][l], handle.dofs[1][l], handle.dofs[2][l]}, handle.dw,
          handle.ldws_h, handle.ldws_d);

      coefficients_restore_3d(handle, dinput, doutput, l, 0);

      lwpk<D, T, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l],
                       handle.dw, handle.ldws_d, dv, ldvs_d, queue_idx);

      if (debug_print) {
        printf("output of recomposition:\n");
        print_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l],
                          handle.dofs[0][l], dv, ldvs_h[0], ldvs_h[1],
                          ldvs_h[0]);
      }
    }
  }
  if (D > 3) {
    for (int l = l_target - 1; l >= 0; l--) {
      // printf("[gpu] l = %d\n", l);
      int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
      int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);
      bool f_padding = handle.dofs[0][l] % 2 == 0;
      bool c_padding = handle.dofs[1][l] % 2 == 0;
      bool r_padding = handle.dofs[0][l] % 2 == 0;

      if (debug_print) { // debug
        printf("recomposition: before corection\n");
        for (int i = 0; i < handle.dofs[3][0]; i++) {
          printf("i = %d\n", i);
          print_matrix_cuda(handle.dofs[2][0], handle.dofs[1][0],
                            handle.dofs[0][0],
                            dv + i * ldvs_h[0] * ldvs_h[1] * ldvs_h[2],
                            ldvs_h[0], ldvs_h[1], ldvs_h[0]);
        }
      }

      int curr_dim_r, curr_dim_c, curr_dim_f;
      int lddv1, lddv2;
      int lddw1, lddw2;
      int lddb1, lddb2;
      // un-apply correction
      std::vector<SIZE> shape(handle.D_padded);
      for (DIM d = 0; d < handle.D_padded; d++)
        shape[d] = handle.shapes_h[l][d];

      SubArray<D, T> dcoeff(shape, dv, ldvs_h, ldvs_d);
      SubArray<D, T> dcorrection(shape, handle.dw, handle.ldws_h,
                                 handle.ldws_d);

      if (debug_print) { // debug
        printf("before subtract correction [%d]\n", l);
        for (int k = 0; k < dcoeff.shape[4]; k++) {
          for (int j = 0; j < dcoeff.shape[3]; j++) {
            printf("i,j = %d,%d\n", k, j);
            print_matrix_cuda(
                dcoeff.shape[2], dcoeff.shape[1], dcoeff.shape[0],
                dcoeff.dv +
                    k * dcoeff.ldvs_h[0] * dcoeff.ldvs_h[1] * dcoeff.ldvs_h[2] *
                        dcoeff.ldvs_h[3] +
                    j * dcoeff.ldvs_h[0] * dcoeff.ldvs_h[1] * dcoeff.ldvs_h[2],
                dcoeff.ldvs_h[0], dcoeff.ldvs_h[1], dcoeff.ldvs_h[0]);
          }
        }
      } // deb

      calc_correction_nd(handle, dcoeff, dcorrection, l, 0);

      lwpk<D, T, SUBTRACT>(handle, handle.shapes_h[l + 1],
                           handle.shapes_d[l + 1], dcorrection.dv,
                           dcorrection.ldvs_d, dv, ldvs_d, queue_idx);

      if (debug_print) { // debug
        printf("after subtract correction [%d]\n", l);
        for (int k = 0; k < dcoeff.shape[4]; k++) {
          for (int j = 0; j < dcoeff.shape[3]; j++) {
            printf("i,j = %d,%d\n", k, j);
            print_matrix_cuda(
                dcoeff.shape[2], dcoeff.shape[1], dcoeff.shape[0],
                dcoeff.dv +
                    k * dcoeff.ldvs_h[0] * dcoeff.ldvs_h[1] * dcoeff.ldvs_h[2] *
                        dcoeff.ldvs_h[3] +
                    j * dcoeff.ldvs_h[0] * dcoeff.ldvs_h[1] * dcoeff.ldvs_h[2],
                dcoeff.ldvs_h[0], dcoeff.ldvs_h[1], dcoeff.ldvs_h[0]);
          }
        }
      } // deb

      lwpk<D, T, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l], dv,
                       ldvs_d, handle.db, handle.ldbs_d, queue_idx);
      lwpk<D, T, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l], dv,
                       ldvs_d, handle.dw, handle.ldws_d, queue_idx);

      SubArray<D, T> dinput1(shape, handle.dw, handle.ldws_h, handle.ldws_d);
      SubArray<D, T> dinput2(shape, handle.db, handle.ldbs_h, handle.ldbs_d);
      SubArray<D, T> doutput(shape, dv, ldvs_h, ldvs_d);

      coefficients_restore_nd(handle, dinput1, dinput2, doutput, l, queue_idx);

    } // loop levels

    if (debug_print) { // debug
      std::vector<SIZE> shape(handle.D_padded);
      for (DIM d = 0; d < handle.D_padded; d++)
        shape[d] = handle.shapes_h[0][d];
      SubArray<D, T> dcoeff(shape, dv, ldvs_h, ldvs_d);
      printf("final output\n");
      for (int k = 0; k < dcoeff.shape[4]; k++) {
        for (int j = 0; j < dcoeff.shape[3]; j++) {
          printf("i,j = %d,%d\n", k, j);
          print_matrix_cuda(
              dcoeff.shape[2], dcoeff.shape[1], dcoeff.shape[0],
              dcoeff.dv +
                  k * dcoeff.ldvs_h[0] * dcoeff.ldvs_h[1] * dcoeff.ldvs_h[2] *
                      dcoeff.ldvs_h[3] +
                  j * dcoeff.ldvs_h[0] * dcoeff.ldvs_h[1] * dcoeff.ldvs_h[2],
              dcoeff.ldvs_h[0], dcoeff.ldvs_h[1], dcoeff.ldvs_h[0]);
        }
      }
    } // deb

    // { // debug
    //     lwpk<D, T, COPY>(handle, handle.shapes_h[0], handle.shapes_d[0], dv,
    //                        ldvs_d, handle.db, handle.ldbs_d, queue_idx);
    //     std::vector<SIZE> shape(D);
    //     for (DIM d = 0; d < D; d++) shape[d] = handle.shapes_h[0][d];
    //     SubArray<D, T> dcoeff(shape, handle.db, handle.ldbs_h,
    //     handle.ldbs_d); SubArray<D, T> doutput(shape, handle.dw,
    //     handle.ldws_h, handle.ldws_d); ReverseReorderGPU(handle, dcoeff,
    //     doutput, 0);

    //     printf("recomposition: done\n");
    //     for (int i = 0; i < handle.dofs[3][0]; i++) {
    //       printf("i = %d\n", i);
    //       print_matrix_cuda(handle.dofs[2][0], handle.dofs[1][0],
    //       handle.dofs[0][0],
    //                         dv+i*ldvs_h[0]*ldvs_h[1]*ldvs_h[2], ldvs_h[0],
    //                         ldvs_h[1], ldvs_h[0]);
    //     }

    //     // printf("after coeff reverse\n");
    //     // for (int i = 0; i < handle.dofs[3][0]; i++) {
    //     //   printf("i = %d\n", i);
    //     //   print_matrix_cuda(handle.dofs[2][0], handle.dofs[1][0],
    //     //   handle.dofs[0][0],
    //     //
    //     doutput.dv+i*doutput.ldvs_h[0]*doutput.ldvs_h[1]*doutput.ldvs_h[2],
    //     doutput.ldvs_h[0], doutput.ldvs_h[1],
    //     //                     doutput.ldvs_h[0]);
    //   }

  } // D > 3
}

} // namespace mgard_cuda