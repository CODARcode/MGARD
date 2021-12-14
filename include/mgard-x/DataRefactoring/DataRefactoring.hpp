/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include "../Handle.hpp"
#include "../RuntimeX/RuntimeX.h"
// #include "SubArray.hpp"
// #include "DeviceAdapters/DeviceAdapterCuda.h"

// #include "DataRefactoring/Coefficient/GridProcessingKernel.h"
#include "Coefficient/GridProcessingKernel.hpp"
// #include "cuda/DataRefactoring/Coefficient/GridProcessingKernel2.hpp"

// #include "DataRefactoring/Coefficient/GridProcessingKernel3D.h"
#include "Coefficient/GridProcessingKernel3D.hpp"
// #include "cuda/DataRefactoring/Coefficient/GridProcessingKernel2.hpp"
// #include "DataRefactoring/Correction/IterativeProcessingKernel.h"
// #include "DataRefactoring/Correction/IterativeProcessingKernel3D.h"
#include "Correction/IterativeProcessingKernel3D.hpp"
#include "Correction/IterativeProcessingKernel.hpp"
// #include "LevelwiseProcessingKernel.h"
#include "Correction/LevelwiseProcessingKernel.hpp"
// #include "DataRefactoring/Correction/LinearProcessingKernel.h"
#include "Correction/LinearProcessingKernel.hpp"
// #include "DataRefactoring/Correction/LinearProcessingKernel3D.h"
#include "Correction/LinearProcessingKernel3D.hpp"

#include "DataRefactoring.h"

// #include "cuda/Testing/ReorderToolsGPU.hpp"

#include <iostream>

#include <chrono>
namespace mgard_x {

static bool store = false;
static bool verify = false;
static bool debug_print = false;

template <typename SubArrayType> 
void CompareSubarray4D(SubArrayType subArray1, SubArrayType subArray2) {
  if (SubArrayType::NumDims != 4) {std::cout << log::log_err << "CompareSubarray4D expects 4D subarray type.\n"; exit(-1); }
  if (subArray1.getShape(3) != subArray2.getShape(3)) {std::cout << log::log_err << "CompareSubarray4D mismatch 4D size.\n"; exit(-1); }

  using T = typename SubArrayType::DataType;
  SIZE idx[4] = {0, 0, 0, 0};
  for (SIZE i = 0; i < subArray1.getShape(3); i++) {
    idx[3] = i;
    SubArrayType temp1 = subArray1;
    SubArrayType temp2 = subArray2;
    temp1.offset(3, i);
    temp2.offset(3, i);
    CompareSubarray("4D = " + std::to_string(i), temp1.Slice3D(0, 1, 2), temp2.Slice3D(0, 1, 2));
  } 
}

template <typename SubArrayType> 
void PrintSubarray4D(std::string name, SubArrayType subArray1) {
  if (SubArrayType::NumDims != 4) {std::cout << log::log_err << "PrintSubarray4D expects 4D subarray type.\n"; exit(-1); }
  std::cout << name << "\n";
  using T = typename SubArrayType::DataType;
  SIZE idx[4] = {0, 0, 0, 0};
  for (SIZE i = 0; i < subArray1.getShape(3); i++) {
    idx[3] = i;
    SubArrayType temp1 = subArray1;
    temp1.offset(3, i);
    PrintSubarray("i = " + std::to_string(i), temp1.Slice3D(0, 1, 2));
  } 
}





template <DIM D, typename T, typename DeviceType>
void calc_coeff_pointers(Handle<D, T, DeviceType> &handle, DIM curr_dims[3], DIM l, SubArray<D, T, DeviceType> doutput,
                         SubArray<D, T, DeviceType> &dcoarse,
                         SubArray<D, T, DeviceType> &dcoeff_f,
                         SubArray<D, T, DeviceType> &dcoeff_c,
                         SubArray<D, T, DeviceType> &dcoeff_r,
                         SubArray<D, T, DeviceType> &dcoeff_cf,
                         SubArray<D, T, DeviceType> &dcoeff_rf,
                         SubArray<D, T, DeviceType> &dcoeff_rc,
                         SubArray<D, T, DeviceType> &dcoeff_rcf) {
 
  SIZE n[3];
  SIZE nn[3];
  for (DIM d = 0; d < 3; d++) {
    n[d] = handle.dofs[curr_dims[d]][l];
    nn[d] = handle.dofs[curr_dims[d]][l+1];
  }

  dcoarse = doutput;
  dcoarse.resize(curr_dims[0], nn[0]);
  dcoarse.resize(curr_dims[1], nn[1]);
  dcoarse.resize(curr_dims[2], nn[2]);

  dcoeff_f = doutput;
  dcoeff_f.offset(curr_dims[0], nn[0]);
  dcoeff_f.resize(curr_dims[0], n[0]-nn[0]);
  dcoeff_f.resize(curr_dims[1], nn[1]);
  dcoeff_f.resize(curr_dims[2], nn[2]);

  dcoeff_c = doutput;
  dcoeff_c.offset(curr_dims[1], nn[1]);
  dcoeff_c.resize(curr_dims[0], nn[0]);
  dcoeff_c.resize(curr_dims[1], n[1]-nn[1]);
  dcoeff_c.resize(curr_dims[2], nn[2]);

  dcoeff_r = doutput;
  dcoeff_r.offset(curr_dims[2], nn[2]);
  dcoeff_r.resize(curr_dims[0], nn[0]);
  dcoeff_r.resize(curr_dims[1], nn[1]);
  dcoeff_r.resize(curr_dims[2], n[2]-nn[2]);

  dcoeff_cf = doutput;
  dcoeff_cf.offset(curr_dims[0], nn[0]);
  dcoeff_cf.offset(curr_dims[1], nn[1]);
  dcoeff_cf.resize(curr_dims[0], n[0]-nn[0]);
  dcoeff_cf.resize(curr_dims[1], n[1]-nn[1]);
  dcoeff_cf.resize(curr_dims[2], nn[2]);

  dcoeff_rf = doutput;
  dcoeff_rf.offset(curr_dims[0], nn[0]);
  dcoeff_rf.offset(curr_dims[2], nn[2]);
  dcoeff_rf.resize(curr_dims[0], n[0]-nn[0]);
  dcoeff_rf.resize(curr_dims[1], nn[1]);
  dcoeff_rf.resize(curr_dims[2], n[2]-nn[2]);

  dcoeff_rc = doutput;
  dcoeff_rc.offset(curr_dims[1], nn[1]);
  dcoeff_rc.offset(curr_dims[2], nn[2]);
  dcoeff_rc.resize(curr_dims[0], nn[0]);
  dcoeff_rc.resize(curr_dims[1], n[1]-nn[1]);
  dcoeff_rc.resize(curr_dims[2], n[2]-nn[2]);

  dcoeff_rcf = doutput;
  dcoeff_rcf.offset(curr_dims[0], nn[0]);
  dcoeff_rcf.offset(curr_dims[1], nn[1]);
  dcoeff_rcf.offset(curr_dims[2], nn[2]);
  dcoeff_rcf.resize(curr_dims[0], n[0]-nn[0]);
  dcoeff_rcf.resize(curr_dims[1], n[1]-nn[1]);
  dcoeff_rcf.resize(curr_dims[2], n[2]-nn[2]);
}

template <DIM D, typename T, typename DeviceType>
void calc_coefficients_3d(Handle<D, T, DeviceType> &handle, SubArray<D, T, DeviceType> dinput, 
                        SubArray<D, T, DeviceType> &doutput, SIZE l, int queue_idx) {

  int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
  int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(handle.shape[d]) + "_";

  dinput.project(0, 1, 2);
  doutput.project(0, 1, 2);
  
  SIZE f = handle.dofs[0][l];
  SIZE c = handle.dofs[1][l];
  SIZE r = handle.dofs[2][l];
  SIZE ff = handle.dofs[0][l+1];
  SIZE cc = handle.dofs[1][l+1];
  SIZE rr = handle.dofs[2][l+1];

  SubArray<D, T, DeviceType> dcoarse = doutput;
  dcoarse.resize({ff, cc, rr});
  SubArray<D, T, DeviceType> dcoeff_f = doutput;
  dcoeff_f.offset({ff, 0, 0});
  dcoeff_f.resize({f-ff, cc, rr});
  SubArray<D, T, DeviceType> dcoeff_c = doutput;
  dcoeff_c.offset({0, cc, 0});
  dcoeff_c.resize({ff, c-cc, rr});
  SubArray<D, T, DeviceType> dcoeff_r = doutput;
  dcoeff_r.offset({0, 0, rr});
  dcoeff_r.resize({ff, cc, r-rr});
  SubArray<D, T, DeviceType> dcoeff_cf = doutput;
  dcoeff_cf.offset({ff, cc, 0});
  dcoeff_cf.resize({f-ff, c-cc, rr});
  SubArray<D, T, DeviceType> dcoeff_rf = doutput;
  dcoeff_rf.offset({ff, 0, rr});
  dcoeff_rf.resize({f-ff, cc, r-rr});
  SubArray<D, T, DeviceType> dcoeff_rc = doutput;
  dcoeff_rc.offset({0, cc, rr});
  dcoeff_rc.resize({ff, c-cc, r-rr});
  SubArray<D, T, DeviceType> dcoeff_rcf = doutput;
  dcoeff_rcf.offset({ff, cc, rr});
  dcoeff_rcf.resize({f-ff, c-cc, r-rr});

  // SubArray<1, T, DeviceType> ratio_r({handle.dofs[2][l]}, handle.ratio[2][l]);
  // SubArray<1, T, DeviceType> ratio_c({handle.dofs[1][l]}, handle.ratio[1][l]);
  // SubArray<1, T, DeviceType> ratio_f({handle.dofs[0][l]}, handle.ratio[0][l]);

  T *null = NULL;
  GpkReo3D<D, T, DeviceType>().Execute(
      handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
      handle.dofs[2][l+1], handle.dofs[1][l+1], handle.dofs[0][l+1],
      SubArray(handle.ratio_array[2][l]),
      SubArray(handle.ratio_array[1][l]),
      SubArray(handle.ratio_array[0][l]),
      // ratio_r, ratio_c, ratio_f,
      dinput, dcoarse,
      dcoeff_f, dcoeff_c, dcoeff_r,
      dcoeff_cf, dcoeff_rf, dcoeff_rc,
      dcoeff_rcf,
      queue_idx);
  // handle.sync_all();
  //  if (debug_print) {  
  //   PrintSubarray("after pi_Ql_reo", doutput);
  // }


  // {
  //   std::vector<SIZE> shape2_rev(D);
  //   std::vector<SIZE> shape2_pad_rev(D);
  //   for (int i = 0; i < D; i++) {
  //     shape2_rev[i] = handle.dofs[D-1-i][0];
  //     shape2_pad_rev[i] = handle.dofs[D-1-i][0] + 2;
  //   }
  //   mgard_cuda::Array<D, T> input2(shape2_rev);
  //   mgard_cuda::Array<D, T> work2(shape2_pad_rev);

  //   MemoryManager<DeviceType>::CopyND(input2.get_dv(), in_array2.get_ldvs_h()[0],
  //                                   dinput.data(), in_array.getLd(0),
  //                                   handle.dofs[0][0], handle.dofs[1][0] * handle.linearized_depth, 0);


  //   gpk_reo_3d(
  //       handle, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
  //       handle.ratio[2][l], handle.ratio[1][l], handle.ratio[0][l], 
  //       dinput.data(), dinput.getLddv1(), dinput.getLddv2(), 
  //       dcoarse.data(), dcoarse.getLddv1(), dcoarse.getLddv2(), 
  //       dcoeff_f.data(), dcoeff_f.getLddv1(), dcoeff_f.getLddv2(), 
  //       dcoeff_c.data(), dcoeff_c.getLddv1(), dcoeff_c.getLddv2(), 
  //       dcoeff_r.data(), dcoeff_r.getLddv1(), dcoeff_r.getLddv2(), 
  //       dcoeff_cf.data(), dcoeff_cf.getLddv1(), dcoeff_cf.getLddv2(), 
  //       dcoeff_rf.data(), dcoeff_rf.getLddv1(), dcoeff_rf.getLddv2(), 
  //       dcoeff_rc.data(), dcoeff_rc.getLddv1(), dcoeff_rc.getLddv2(),
  //       dcoeff_rcf.data(), dcoeff_rcf.getLddv1(), dcoeff_rcf.getLddv2(),
  //       queue_idx, handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

  // }

  verify_matrix_cuda(handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l], 
                     doutput.data(), doutput.getLd(0), doutput.getLd(1), doutput.getLd(0),
                     prefix + "gpk_reo_3d" + "_level_" + std::to_string(l),
                     store, verify);

  if (debug_print) {  
    PrintSubarray("after pi_Ql_reo", doutput);
  }
}

template <DIM D, typename T, typename DeviceType>
void coefficients_restore_3d(Handle<D, T, DeviceType> &handle, SubArray<D, T, DeviceType> dinput, 
                        SubArray<D, T, DeviceType> &doutput, SIZE l, int queue_idx) {

  int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
  int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(handle.shape[d]) + "_";

  dinput.project(0, 1, 2);
  doutput.project(0, 1, 2);
  
  SIZE f = handle.dofs[0][l];
  SIZE c = handle.dofs[1][l];
  SIZE r = handle.dofs[2][l];
  SIZE ff = handle.dofs[0][l+1];
  SIZE cc = handle.dofs[1][l+1];
  SIZE rr = handle.dofs[2][l+1];

  SubArray<D, T, DeviceType> dcoarse = dinput;
  dcoarse.resize({ff, cc, rr});
  SubArray<D, T, DeviceType> dcoeff_f = dinput;
  dcoeff_f.offset({ff, 0, 0});
  dcoeff_f.resize({f-ff, cc, rr});
  SubArray<D, T, DeviceType> dcoeff_c = dinput;
  dcoeff_c.offset({0, cc, 0});
  dcoeff_c.resize({ff, c-cc, rr});
  SubArray<D, T, DeviceType> dcoeff_r = dinput;
  dcoeff_r.offset({0, 0, rr});
  dcoeff_r.resize({ff, cc, r-rr});
  SubArray<D, T, DeviceType> dcoeff_cf = dinput;
  dcoeff_cf.offset({ff, cc, 0});
  dcoeff_cf.resize({f-ff, c-cc, rr});
  SubArray<D, T, DeviceType> dcoeff_rf = dinput;
  dcoeff_rf.offset({ff, 0, rr});
  dcoeff_rf.resize({f-ff, cc, r-rr});
  SubArray<D, T, DeviceType> dcoeff_rc = dinput;
  dcoeff_rc.offset({0, cc, rr});
  dcoeff_rc.resize({ff, c-cc, r-rr});
  SubArray<D, T, DeviceType> dcoeff_rcf = dinput;
  dcoeff_rcf.offset({ff, cc, rr});
  dcoeff_rcf.resize({f-ff, c-cc, r-rr});



  // SubArray<1, T, DeviceType> ratio_r({handle.dofs[2][l]}, handle.ratio[2][l]);
  // SubArray<1, T, DeviceType> ratio_c({handle.dofs[1][l]}, handle.ratio[1][l]);
  // SubArray<1, T, DeviceType> ratio_f({handle.dofs[0][l]}, handle.ratio[0][l]);

  GpkRev3D<D, T, DeviceType>().Execute(
      handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
      handle.dofs[2][l+1], handle.dofs[1][l+1], handle.dofs[0][l+1],
      SubArray(handle.ratio_array[2][l]),
      SubArray(handle.ratio_array[1][l]),
      SubArray(handle.ratio_array[0][l]),
      // ratio_r, ratio_c, ratio_f,
      doutput, dcoarse,
      dcoeff_f, dcoeff_c, dcoeff_r,
      dcoeff_cf, dcoeff_rf, dcoeff_rc,
      dcoeff_rcf,
      0, 0, 0, 
      handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
      queue_idx);



  T *null = NULL;
  // gpk_rev_3d(
  //     handle, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l],
  //     handle.ratio[2][l], handle.ratio[1][l], handle.ratio[0][l], 
  //     doutput.data(), doutput.getLddv1(), doutput.getLddv2(), 
  //     dcoarse.data(), dcoarse.getLddv1(), dcoarse.getLddv2(), 
  //     // null, ldvs_h[0], ldvs_h[1],
  //     dcoeff_f.data(), dcoeff_f.getLddv1(), dcoeff_f.getLddv2(), 
  //     // null, ldvs_h[0], ldvs_h[1],
  //     dcoeff_c.data(), dcoeff_c.getLddv1(), dcoeff_c.getLddv2(), 
  //     // null, ldvs_h[0], ldvs_h[1],
  //     dcoeff_r.data(), dcoeff_r.getLddv1(), dcoeff_r.getLddv2(), 
  //     // null, ldvs_h[0], ldvs_h[1],
  //     dcoeff_cf.data(), dcoeff_cf.getLddv1(), dcoeff_cf.getLddv2(), 
  //     // null, ldvs_h[0], ldvs_h[1],
  //     dcoeff_rf.data(), dcoeff_rf.getLddv1(), dcoeff_rf.getLddv2(), 
  //     // null, ldvs_h[0], ldvs_h[1],
  //     dcoeff_rc.data(), dcoeff_rc.getLddv1(), dcoeff_rc.getLddv2(),
  //     // null, ldvs_h[0], ldvs_h[1],
  //     dcoeff_rcf.data(), dcoeff_rcf.getLddv1(), dcoeff_rcf.getLddv2(),
  //     // null, ldvs_h[0], ldvs_h[1],
  //     0, 0, 0, handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l], queue_idx,
  //     handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

  // handle.sync(0);
  verify_matrix_cuda(
      handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l], 
      doutput.data(), doutput.getLd(0), doutput.getLd(1), doutput.getLd(0),
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
  // handle.dofs[0][l], doutput.data(), doutput.ldvs_h[0], doutput.ldvs_h[1], doutput.ldvs_h[0],);

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
    PrintSubarray("after coeff-restore", doutput);
  }
}

template <DIM D, typename T, typename DeviceType>
void calc_correction_3d(Handle<D, T, DeviceType> &handle, SubArray<D, T, DeviceType> dcoeff, 
                        SubArray<D, T, DeviceType> &dcorrection, SIZE l, int queue_idx) {

  int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
  int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(handle.shape[d]) + "_";

  SubArray<D, T, DeviceType> dw_in1, dw_in2, dw_out;

  if (D >= 1) {
    dw_in1 = dcoeff;
    dw_in1.resize({handle.dofs[0][l+1], handle.dofs[1][l], handle.dofs[2][l]});
    dw_in2 = dcoeff;
    dw_in2.offset({handle.dofs[0][l+1], 0, 0});
    dw_in2.resize({handle.dofs[0][l]-handle.dofs[0][l+1], handle.dofs[1][l], handle.dofs[2][l]});
    dw_out = dcorrection;
    dw_out.resize({handle.dofs[0][l+1], handle.dofs[1][l], handle.dofs[2][l]});

    Lpk1Reo3D<D, T, DeviceType>().Execute(
        handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l], handle.dofs[0][l + 1], 
        handle.dofs[2][l + 1], handle.dofs[1][l + 1], handle.dofs[0][l + 1], 
        SubArray(handle.dist_array[0][l]),
        SubArray(handle.ratio_array[0][l]),
        dw_in1, dw_in2, dw_out, queue_idx);


    verify_matrix_cuda(
        handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l + 1],
        dw_out.data(), dw_out.getLd(0), dw_out.getLd(1), dw_out.getLd(0),
        prefix + "lpk_reo_1_3d" + "_level_" + std::to_string(l), store,
        verify);

    if (debug_print) {
      PrintSubarray("after mass_trans_multiply_1_cpt", dw_out);
    }
  }

  if (D >= 2) {
    dw_in1 = dw_out;
    dw_in1.resize({handle.dofs[0][l+1], handle.dofs[1][l+1], handle.dofs[2][l]});
    dw_in2 = dw_out;
    dw_in2.offset({0, handle.dofs[1][l+1], 0});
    dw_in2.resize({handle.dofs[0][l+1], handle.dofs[1][l]-handle.dofs[1][l+1], handle.dofs[2][l]});
    dw_out.offset({handle.dofs[0][l+1], 0, 0});
    dw_out.resize({handle.dofs[0][l+1], handle.dofs[1][l+1], handle.dofs[2][l]});

    Lpk2Reo3D<D, T, DeviceType>().Execute(
        handle.dofs[2][l], handle.dofs[1][l], handle.dofs[0][l + 1],
        handle.dofs[1][l + 1], 
        SubArray(handle.dist_array[1][l]),
        SubArray(handle.ratio_array[1][l]),
        dw_in1, dw_in2, dw_out, queue_idx);

    verify_matrix_cuda(
        handle.dofs[2][l], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
        dw_out.data(), dw_out.getLd(0), dw_out.getLd(1), dw_out.getLd(0),
        prefix + "lpk_reo_2_3d" + "_level_" + std::to_string(l), store,
        verify);

    if (debug_print) {
      PrintSubarray("after mass_trans_multiply_2_cpt", dw_out);
    }
  }

  if (D == 3) {
    dw_in1 = dw_out;
    dw_in1.resize({handle.dofs[0][l+1], handle.dofs[1][l+1], handle.dofs[2][l+1]});
    dw_in2 = dw_out;
    dw_in2.offset({0, 0, handle.dofs[2][l+1]});
    dw_in2.resize({handle.dofs[0][l+1], handle.dofs[1][l+1], handle.dofs[2][l]-handle.dofs[2][l+1]});
    dw_out.offset({handle.dofs[0][l+1], handle.dofs[1][l+1], 0});
    dw_out.resize({handle.dofs[0][l+1], handle.dofs[1][l+1], handle.dofs[2][l+1]});

    Lpk3Reo3D<D, T, DeviceType>().Execute(
    handle.dofs[2][l], handle.dofs[1][l+1], handle.dofs[0][l+1], 
    handle.dofs[2][l+1], 
    SubArray(handle.dist_array[2][l]),
    SubArray(handle.ratio_array[2][l]),
    dw_in1, dw_in2, dw_out, queue_idx);

    verify_matrix_cuda(
        handle.dofs[2][l + 1], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
        dw_out.data(), dw_out.getLd(0), dw_out.getLd(1), dw_out.getLd(0),
        prefix + "lpk_reo_3_3d" + "_level_" + std::to_string(l), store,
        verify);

    if (debug_print) {
      PrintSubarray("after mass_trans_multiply_3_cpt", dw_out);
    }
  }

  if (D >= 1) {
    Ipk1Reo3D<D, T, DeviceType>().Execute(
            handle.dofs[2][l+1], handle.dofs[1][l+1], handle.dofs[0][l+1],
            SubArray(handle.am_array[0][l+1]),
            SubArray(handle.bm_array[0][l+1]),
            SubArray(handle.dist_array[0][l+1]),
            dw_out, queue_idx);
    verify_matrix_cuda(
        handle.dofs[2][l+1], handle.dofs[1][l+1], handle.dofs[0][l + 1],
        dw_out.data(), dw_out.getLd(0), dw_out.getLd(1), dw_out.getLd(0),
        prefix + "ipk_1_3d" + "_level_" + std::to_string(l), store, verify);

    if (debug_print) {
      PrintSubarray("after solve_tridiag_1_cpt", dw_out);
    }
  }
  if (D >= 2) {
    Ipk2Reo3D<D, T, DeviceType>().Execute(
            handle.dofs[2][l+1], handle.dofs[1][l+1], handle.dofs[0][l+1],
            SubArray(handle.am_array[1][l+1]),
            SubArray(handle.bm_array[1][l+1]),
            SubArray(handle.dist_array[1][l+1]),
            dw_out, queue_idx);

    verify_matrix_cuda(
        handle.dofs[2][l+1], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
        dw_out.data(), dw_out.getLd(0), dw_out.getLd(1), dw_out.getLd(0),
        prefix + "ipk_2_3d" + "_level_" + std::to_string(l), store, verify);

    if (debug_print) {
      PrintSubarray("after solve_tridiag_2_cpt", dw_out);
    }
  }
  if (D == 3) {
    Ipk3Reo3D<D, T, DeviceType>().Execute(
            handle.dofs[2][l+1], handle.dofs[1][l+1], handle.dofs[0][l+1],
            SubArray(handle.am_array[2][l+1]),
            SubArray(handle.bm_array[2][l+1]),
            SubArray(handle.dist_array[2][l+1]),
            dw_out, queue_idx);

    verify_matrix_cuda(
        handle.dofs[2][l + 1], handle.dofs[1][l + 1], handle.dofs[0][l + 1],
        dw_out.data(), dw_out.getLd(0), dw_out.getLd(1), dw_out.getLd(0),
        prefix + "ipk_3_3d" + "_level_" + std::to_string(l), store, verify);

    if (debug_print) {
      PrintSubarray("after solve_tridiag_3_cpt", dw_out);
    }
  }
  // final correction output
  dcorrection = dw_out;
}

template <DIM D, typename T, typename DeviceType>
void calc_coefficients_nd(Handle<D, T, DeviceType> &handle, SubArray<D, T, DeviceType> dinput1, 
                          SubArray<D, T, DeviceType> dinput2, 
                        SubArray<D, T, DeviceType> &doutput, SIZE l, int queue_idx) {

  int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
  int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(handle.shape[d]) + "_";
  // printf("interpolate 1-3D\n");

  SubArray<D, T, DeviceType> dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, 
                 dcoeff_cf, dcoeff_rf, dcoeff_rc,
                 dcoeff_rcf;

  DIM curr_dims[3];

  int unprocessed_idx = 0;
  curr_dims[0] = 0; curr_dims[1] = 1; curr_dims[2] = 2;
  dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]);

  calc_coeff_pointers(handle, curr_dims, l, doutput, 
                      dcoarse, 
                      dcoeff_f, dcoeff_c, dcoeff_r, 
                      dcoeff_cf, dcoeff_rf, dcoeff_rc,
                      dcoeff_rcf);

  // gpuErrchk(cudaDeviceSynchronize());
  GpkReo<D, 3, T, true, false, 1, DeviceType>().Execute(
      SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
      SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
      handle.unprocessed_n[unprocessed_idx],
      // unprocessed_dims_subarray,
      SubArray(handle.unprocessed_dims[unprocessed_idx]),
      curr_dims[2], curr_dims[1], curr_dims[0],
      SubArray(handle.ratio_array[curr_dims[2]][l]),
      SubArray(handle.ratio_array[curr_dims[1]][l]),
      SubArray(handle.ratio_array[curr_dims[0]][l]),
      // ratio_r, ratio_c, ratio_f, 
      dinput1, dcoarse, dcoeff_f, dcoeff_c, dcoeff_r,
      dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf, queue_idx);
  // gpuErrchk(cudaDeviceSynchronize());

  for (DIM d = 3; d < D; d += 2) {
    //copy back to input1 for interpolation again
    LwpkReo<D, T, COPY, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(handle.shapes[l], true), doutput, dinput1, queue_idx);

    // printf("interpolate %u-%uD\n", d+1, d+2);
    curr_dims[0] = 0; curr_dims[1] = d; curr_dims[2] = d+1;
    dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    calc_coeff_pointers(handle, curr_dims, l, doutput, 
                        dcoarse, 
                        dcoeff_f, dcoeff_c, dcoeff_r, 
                        dcoeff_cf, dcoeff_rf, dcoeff_rc,
                        dcoeff_rcf);

    if (D - d == 1) {
      unprocessed_idx+=1;

      GpkReo<D, 2, T, true, false, 2, DeviceType>().Execute(
          SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
          SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
          handle.unprocessed_n[unprocessed_idx],
          SubArray(handle.unprocessed_dims[unprocessed_idx]),
          curr_dims[2], curr_dims[1], curr_dims[0],
          SubArray(handle.ratio_array[curr_dims[2]][l]),
          SubArray(handle.ratio_array[curr_dims[1]][l]),
          SubArray(handle.ratio_array[curr_dims[0]][l]),
          dinput1, dcoarse, dcoeff_f, dcoeff_c, dcoeff_r,
          dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf, queue_idx);

    } else { //D - d >= 2
      unprocessed_idx += 2;
      GpkReo<D, 3, T, true, false, 2, DeviceType>().Execute(
          SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
          SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
          handle.unprocessed_n[unprocessed_idx],
          SubArray(handle.unprocessed_dims[unprocessed_idx]),
          // unprocessed_dims_subarray,
          curr_dims[2], curr_dims[1], curr_dims[0],
          // ratio_r, ratio_c, ratio_f, 
          SubArray(handle.ratio_array[curr_dims[2]][l]),
          SubArray(handle.ratio_array[curr_dims[1]][l]),
          SubArray(handle.ratio_array[curr_dims[0]][l]),
          dinput1, dcoarse, dcoeff_f, dcoeff_c, dcoeff_r,
          dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf, queue_idx);
    }
  }


  if (debug_print){ // debug
    PrintSubarray4D("after interpolation", doutput);
  } //debug

  unprocessed_idx = 0;
  // printf("reorder 1-3D\n");
  curr_dims[0] = 0; curr_dims[1] = 1; curr_dims[2] = 2;
  dinput2.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]); //reuse input1 as temp output

  calc_coeff_pointers(handle, curr_dims, l, dinput1, 
                      dcoarse, 
                      dcoeff_f, dcoeff_c, dcoeff_r, 
                      dcoeff_cf, dcoeff_rf, dcoeff_rc,
                      dcoeff_rcf);

  GpkReo<D, 3, T, false, false, 1, DeviceType>().Execute(
      SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
      SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
      handle.unprocessed_n[unprocessed_idx],
      SubArray(handle.unprocessed_dims[unprocessed_idx]),
      // unprocessed_dims_subarray,
      curr_dims[2], curr_dims[1], curr_dims[0],
      // ratio_r, ratio_c, ratio_f,
      SubArray(handle.ratio_array[curr_dims[2]][l]),
      SubArray(handle.ratio_array[curr_dims[1]][l]),
      SubArray(handle.ratio_array[curr_dims[0]][l]), 
      dinput2, dcoarse, dcoeff_f, dcoeff_c, dcoeff_r,
      dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf, queue_idx);

  DIM D_reduced = D % 2 == 0 ? D-1 : D-2;
  for (DIM d = 3; d < D_reduced; d += 2) {
    //copy back to input2 for reordering again

    LwpkReo<D, T, COPY, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(handle.shapes[l], true), dinput1, dinput2, queue_idx);

    unprocessed_idx += 2;
    // printf("reorder %u-%uD\n", d+1, d+2);
    curr_dims[0] = 0; curr_dims[1] = d; curr_dims[2] = d+1;
    dinput2.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]); //reuse input1 as temp output

    calc_coeff_pointers(handle, curr_dims, l, dinput1, 
                        dcoarse, 
                        dcoeff_f, dcoeff_c, dcoeff_r, 
                        dcoeff_cf, dcoeff_rf, dcoeff_rc,
                        dcoeff_rcf);

    GpkReo<D, 3, T, false, false, 2, DeviceType>().Execute(
        SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
        SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
        handle.unprocessed_n[unprocessed_idx],
        SubArray(handle.unprocessed_dims[unprocessed_idx]),
        // unprocessed_dims_subarray,
        curr_dims[2], curr_dims[1], curr_dims[0],
        // ratio_r, ratio_c, ratio_f, 
        SubArray(handle.ratio_array[curr_dims[2]][l]),
        SubArray(handle.ratio_array[curr_dims[1]][l]),
        SubArray(handle.ratio_array[curr_dims[0]][l]),
        dinput2, dcoarse, dcoeff_f, dcoeff_c, dcoeff_r,
        dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf, queue_idx);
    // gpuErrchk(cudaDeviceSynchronize());


  }
  
  // printf("calc coeff %u-%dD\n", D_reduced+1, D_reduced+2);
  curr_dims[0] = 0; curr_dims[1] = D_reduced; curr_dims[2] = D_reduced+1;
  dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]); //reuse input1 as temp output
  calc_coeff_pointers(handle, curr_dims, l, doutput, 
                      dcoarse, 
                      dcoeff_f, dcoeff_c, dcoeff_r, 
                      dcoeff_cf, dcoeff_rf, dcoeff_rc,
                      dcoeff_rcf);
  if (D-D_reduced == 1) {
    unprocessed_idx += 1;
    GpkReo<D, 2, T, false, true, 2, DeviceType>().Execute(
        SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
        SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
        handle.unprocessed_n[unprocessed_idx],
        SubArray(handle.unprocessed_dims[unprocessed_idx]),
        // unprocessed_dims_subarray,
        curr_dims[2], curr_dims[1], curr_dims[0],
        // ratio_r, ratio_c, ratio_f, 
        SubArray(handle.ratio_array[curr_dims[2]][l]),
        SubArray(handle.ratio_array[curr_dims[1]][l]),
        SubArray(handle.ratio_array[curr_dims[0]][l]),
        dinput1, dcoarse, dcoeff_f, dcoeff_c, dcoeff_r,
        dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf, queue_idx);
    // gpuErrchk(cudaDeviceSynchronize());

  } else { //D-D_reduced == 2
    unprocessed_idx += 2;

    GpkReo<D, 3, T, false, true, 2, DeviceType>().Execute(
        SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
        SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
        handle.unprocessed_n[unprocessed_idx],
        SubArray(handle.unprocessed_dims[unprocessed_idx]),
        curr_dims[2], curr_dims[1], curr_dims[0],
        SubArray(handle.ratio_array[curr_dims[2]][l]),
        SubArray(handle.ratio_array[curr_dims[1]][l]),
        SubArray(handle.ratio_array[curr_dims[0]][l]),
        dinput1, dcoarse, dcoeff_f, dcoeff_c, dcoeff_r,
        dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf, queue_idx);
  }

  if (debug_print) { // debug
    PrintSubarray4D("after calc coeff", doutput);
  } //debug

}

template <DIM D, typename T, typename DeviceType>
void coefficients_restore_nd(Handle<D, T, DeviceType> &handle, SubArray<D, T, DeviceType> dinput1, 
                             SubArray<D, T, DeviceType> dinput2, 
                             SubArray<D, T, DeviceType> &doutput, SIZE l, int queue_idx) {

  int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
  int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(handle.shape[d]) + "_";
  

  SubArray<D, T, DeviceType> dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, 
                 dcoeff_cf, dcoeff_rf, dcoeff_rc,
                 dcoeff_rcf;

  DIM curr_dims[3];
  int unprocessed_idx = 0;

  // printf("interpolate-restore 1-3D\n");
  curr_dims[0] = 0; curr_dims[1] = 1; curr_dims[2] = 2;
  dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]);




  calc_coeff_pointers(handle, curr_dims, l, dinput1, 
                      dcoarse, 
                      dcoeff_f, dcoeff_c, dcoeff_r, 
                      dcoeff_cf, dcoeff_rf, dcoeff_rc,
                      dcoeff_rcf);

  // gpk_rev<D, 3, T, true, false, 1>(
  //     handle, handle.shapes_h[l], handle.shapes_d[l],
  //     handle.shapes_d[l + 1], doutput.getLdd(), dinput1.getLdd(),
  //     handle.unprocessed_n[unprocessed_idx],
  //     handle.unprocessed_dims_d[unprocessed_idx],
  //     curr_dims[2], curr_dims[1], curr_dims[0],
  //     handle.ratio[curr_dims[2]][l], handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], 
  //     doutput.data(), doutput.getLddv1(), doutput.getLddv2(), 
  //     dcoarse.data(), dcoarse.getLddv1(), dcoarse.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     dcoeff_f.data(), dcoeff_f.getLddv1(), dcoeff_f.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     dcoeff_c.data(), dcoeff_c.getLddv1(), dcoeff_c.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     dcoeff_r.data(), dcoeff_r.getLddv1(), dcoeff_r.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     dcoeff_cf.data(), dcoeff_cf.getLddv1(), dcoeff_cf.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     dcoeff_rf.data(), dcoeff_rf.getLddv1(), dcoeff_rf.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     dcoeff_rc.data(), dcoeff_rc.getLddv1(), dcoeff_rc.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     dcoeff_rcf.data(), dcoeff_rcf.getLddv1(), dcoeff_rcf.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     0, 0, 0, 
  //     handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
  //     queue_idx,
  //     handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

  // gpuErrchk(cudaDeviceSynchronize());
  GpkRev<D, 3, T, true, false, 1, DeviceType>().Execute(
      SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
      SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
      handle.unprocessed_n[unprocessed_idx],
      SubArray(handle.unprocessed_dims[unprocessed_idx]),
      // unprocessed_dims_subarray,
      curr_dims[2], curr_dims[1], curr_dims[0],
      // ratio_r, ratio_c, ratio_f, 
      SubArray(handle.ratio_array[curr_dims[2]][l]),
      SubArray(handle.ratio_array[curr_dims[1]][l]),
      SubArray(handle.ratio_array[curr_dims[0]][l]),
      doutput, dcoarse, dcoeff_f, dcoeff_c, dcoeff_r,
      dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf, 
      0, 0, 0, 
      handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
      queue_idx);
  // gpuErrchk(cudaDeviceSynchronize());
  

  for (DIM d = 3; d < D; d += 2) { 
    // lwpk<D, T, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l],
    //                  doutput.data(), doutput.getLdd(), 
    //                  dinput1.data(), dinput1.getLdd(), queue_idx);

    // gpuErrchk(cudaDeviceSynchronize());
    LwpkReo<D, T, COPY, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(handle.shapes[l], true), doutput, dinput1, queue_idx);
    // gpuErrchk(cudaDeviceSynchronize());

    // printf("interpolate-restore %u-%uD\n", d+1, d+2);
    curr_dims[0] = 0; curr_dims[1] = d; curr_dims[2] = d+1;
    dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    calc_coeff_pointers(handle, curr_dims, l, dinput1, 
                        dcoarse, 
                        dcoeff_f, dcoeff_c, dcoeff_r, 
                        dcoeff_cf, dcoeff_rf, dcoeff_rc,
                        dcoeff_rcf);

    if (D - d == 1) {
      unprocessed_idx += 1;
      // unprocessed_dims_subarray = SubArray<1, DIM, DeviceType>({(SIZE)handle.unprocessed_n[unprocessed_idx]}, 
      //                                                     handle.unprocessed_dims_d[unprocessed_idx]);
      // ratio_r = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[2]][l]}, handle.ratio[curr_dims[2]][l]);
      // ratio_c = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[1]][l]}, handle.ratio[curr_dims[1]][l]);
      // ratio_f = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[0]][l]}, handle.ratio[curr_dims[0]][l]);

      // gpk_rev<D, 2, T, true, false, 2>(
      //     handle, handle.shapes_h[l], handle.shapes_d[l],
      //     handle.shapes_d[l + 1], doutput.getLdd(), dinput1.getLdd(),
      //     handle.unprocessed_n[unprocessed_idx],
      //     handle.unprocessed_dims_d[unprocessed_idx],
      //     curr_dims[2], curr_dims[1], curr_dims[0], 
      //     handle.ratio[curr_dims[2]][l], handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], 
      //     doutput.data(), doutput.getLddv1(), doutput.getLddv2(), 
      //     dcoarse.data(), dcoarse.getLddv1(), dcoarse.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     dcoeff_f.data(), dcoeff_f.getLddv1(), dcoeff_f.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     dcoeff_c.data(), dcoeff_c.getLddv1(), dcoeff_c.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     dcoeff_r.data(), dcoeff_r.getLddv1(), dcoeff_r.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     dcoeff_cf.data(), dcoeff_cf.getLddv1(), dcoeff_cf.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     dcoeff_rf.data(), dcoeff_rf.getLddv1(), dcoeff_rf.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     dcoeff_rc.data(), dcoeff_rc.getLddv1(), dcoeff_rc.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     dcoeff_rcf.data(), dcoeff_rcf.getLddv1(), dcoeff_rcf.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     0, 0, 0, 
      //     handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
      //     queue_idx,
      //     handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

      // gpuErrchk(cudaDeviceSynchronize());
      GpkRev<D, 2, T, true, false, 2, DeviceType>().Execute(
          SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
          SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
          handle.unprocessed_n[unprocessed_idx],
          SubArray(handle.unprocessed_dims[unprocessed_idx]),
          // unprocessed_dims_subarray,
          curr_dims[2], curr_dims[1], curr_dims[0],
          // ratio_r, ratio_c, ratio_f, 
          SubArray(handle.ratio_array[curr_dims[2]][l]),
          SubArray(handle.ratio_array[curr_dims[1]][l]),
          SubArray(handle.ratio_array[curr_dims[0]][l]),
          doutput, dcoarse, dcoeff_f, dcoeff_c, dcoeff_r,
          dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf, 
          0, 0, 0, 
          handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
          queue_idx);
      // gpuErrchk(cudaDeviceSynchronize());

    } else { // D - d >= 2
      unprocessed_idx += 2;
      // unprocessed_dims_subarray = SubArray<1, DIM, DeviceType>({(SIZE)handle.unprocessed_n[unprocessed_idx]}, 
      //                                                     handle.unprocessed_dims_d[unprocessed_idx]);
      // ratio_r = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[2]][l]}, handle.ratio[curr_dims[2]][l]);
      // ratio_c = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[1]][l]}, handle.ratio[curr_dims[1]][l]);
      // ratio_f = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[0]][l]}, handle.ratio[curr_dims[0]][l]);

      // gpk_rev<D, 3, T, true, false, 2>(
      //     handle, handle.shapes_h[l], handle.shapes_d[l],
      //     handle.shapes_d[l + 1], doutput.getLdd(), dinput1.getLdd(),
      //     handle.unprocessed_n[unprocessed_idx],
      //     handle.unprocessed_dims_d[unprocessed_idx],
      //     curr_dims[2], curr_dims[1], curr_dims[0], 
      //     handle.ratio[curr_dims[2]][l], handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], 
      //     doutput.data(), doutput.getLddv1(), doutput.getLddv2(), 
      //     dcoarse.data(), dcoarse.getLddv1(), dcoarse.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     dcoeff_f.data(), dcoeff_f.getLddv1(), dcoeff_f.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     dcoeff_c.data(), dcoeff_c.getLddv1(), dcoeff_c.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     dcoeff_r.data(), dcoeff_r.getLddv1(), dcoeff_r.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     dcoeff_cf.data(), dcoeff_cf.getLddv1(), dcoeff_cf.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     dcoeff_rf.data(), dcoeff_rf.getLddv1(), dcoeff_rf.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     dcoeff_rc.data(), dcoeff_rc.getLddv1(), dcoeff_rc.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     dcoeff_rcf.data(), dcoeff_rcf.getLddv1(), dcoeff_rcf.getLddv2(), 
      //     // null, lddv1, lddv2,
      //     0, 0, 0, 
      //     handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
      //     queue_idx,
      //     handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

      // gpuErrchk(cudaDeviceSynchronize());
      GpkRev<D, 3, T, true, false, 2, DeviceType>().Execute(
          SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
          SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
          handle.unprocessed_n[unprocessed_idx],
          SubArray(handle.unprocessed_dims[unprocessed_idx]),
          // unprocessed_dims_subarray,
          curr_dims[2], curr_dims[1], curr_dims[0],
          // ratio_r, ratio_c, ratio_f, 
          SubArray(handle.ratio_array[curr_dims[2]][l]),
          SubArray(handle.ratio_array[curr_dims[1]][l]),
          SubArray(handle.ratio_array[curr_dims[0]][l]),
          doutput, dcoarse, dcoeff_f, dcoeff_c, dcoeff_r,
          dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf, 
          0, 0, 0, 
          handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
          queue_idx);
      // gpuErrchk(cudaDeviceSynchronize());

    }
  }
  // Done interpolation-restore on doutput


  if (debug_print){ // debug
    PrintSubarray4D("After interpolation reverse-reorder", doutput);
  } //debug


  unprocessed_idx = 0;

  // printf("reorder-restore 1-3D\n");
  curr_dims[0] = 0; curr_dims[1] = 1; curr_dims[2] = 2;
  dinput2.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]); //reuse input1 as temp space

  // unprocessed_dims_subarray = SubArray<1, DIM, DeviceType>({(SIZE)handle.unprocessed_n[unprocessed_idx]}, 
  //                                                         handle.unprocessed_dims_d[unprocessed_idx]);
  // ratio_r = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[2]][l]}, handle.ratio[curr_dims[2]][l]);
  // ratio_c = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[1]][l]}, handle.ratio[curr_dims[1]][l]);
  // ratio_f = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[0]][l]}, handle.ratio[curr_dims[0]][l]);


  calc_coeff_pointers(handle, curr_dims, l, dinput2, 
                      dcoarse, 
                      dcoeff_f, dcoeff_c, dcoeff_r, 
                      dcoeff_cf, dcoeff_rf, dcoeff_rc,
                      dcoeff_rcf);

  // gpk_rev<D, 3, T, false, false, 1>(
  //     handle, handle.shapes_h[l], handle.shapes_d[l],
  //     handle.shapes_d[l + 1], dinput1.getLdd(), dinput2.getLdd(),
  //     handle.unprocessed_n[unprocessed_idx],
  //     handle.unprocessed_dims_d[unprocessed_idx],
  //     curr_dims[2], curr_dims[1], curr_dims[0], 
  //     handle.ratio[curr_dims[2]][l], handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], 
  //     dinput1.data(), dinput1.getLddv1(), dinput1.getLddv2(), 
  //     dcoarse.data(), dcoarse.getLddv1(), dcoarse.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     dcoeff_f.data(), dcoeff_f.getLddv1(), dcoeff_f.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     dcoeff_c.data(), dcoeff_c.getLddv1(), dcoeff_c.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     dcoeff_r.data(), dcoeff_r.getLddv1(), dcoeff_r.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     dcoeff_cf.data(), dcoeff_cf.getLddv1(), dcoeff_cf.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     dcoeff_rf.data(), dcoeff_rf.getLddv1(), dcoeff_rf.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     dcoeff_rc.data(), dcoeff_rc.getLddv1(), dcoeff_rc.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     dcoeff_rcf.data(), dcoeff_rcf.getLddv1(), dcoeff_rcf.getLddv2(), 
  //     // null, lddv1, lddv2,
  //     0, 0, 0, 
  //     handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
  //     queue_idx,
  //     handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

  // gpuErrchk(cudaDeviceSynchronize());
  GpkRev<D, 3, T, false, false, 1, DeviceType>().Execute(
      SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
      SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
      handle.unprocessed_n[unprocessed_idx],
      SubArray(handle.unprocessed_dims[unprocessed_idx]),
      // unprocessed_dims_subarray,
      curr_dims[2], curr_dims[1], curr_dims[0],
      // ratio_r, ratio_c, ratio_f, 
      SubArray(handle.ratio_array[curr_dims[2]][l]),
      SubArray(handle.ratio_array[curr_dims[1]][l]),
      SubArray(handle.ratio_array[curr_dims[0]][l]),
      dinput1, dcoarse, dcoeff_f, dcoeff_c, dcoeff_r,
      dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf, 
      0, 0, 0, 
      handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
      queue_idx);
  // gpuErrchk(cudaDeviceSynchronize());

  DIM D_reduced = D % 2 == 0 ? D-1 : D-2;
  for (DIM d = 3; d < D_reduced; d += 2) {
    // printf("reorder-reverse\n");
    //copy back to input2 for reordering again
    // lwpk<D, T, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l],
    //                dinput1.data(), dinput1.getLdd(), dinput2.data(), dinput2.getLdd(), queue_idx);

    // gpuErrchk(cudaDeviceSynchronize());
    LwpkReo<D, T, COPY, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(handle.shapes[l], true), dinput1, dinput2, queue_idx);
    // gpuErrchk(cudaDeviceSynchronize());

    
    // printf("reorder-restore %u-%uD\n", d+1, d+2);
    curr_dims[0] = 0; curr_dims[1] = d; curr_dims[2] = d+1;
    dinput2.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]); //reuse input1 as temp output

    



    calc_coeff_pointers(handle, curr_dims, l, dinput2, 
                        dcoarse, 
                        dcoeff_f, dcoeff_c, dcoeff_r, 
                        dcoeff_cf, dcoeff_rf, dcoeff_rc,
                        dcoeff_rcf);

    unprocessed_idx += 2;

    // unprocessed_dims_subarray = SubArray<1, DIM, DeviceType>({(SIZE)handle.unprocessed_n[unprocessed_idx]}, 
    //                                                       handle.unprocessed_dims_d[unprocessed_idx]);
    // ratio_r = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[2]][l]}, handle.ratio[curr_dims[2]][l]);
    // ratio_c = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[1]][l]}, handle.ratio[curr_dims[1]][l]);
    // ratio_f = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[0]][l]}, handle.ratio[curr_dims[0]][l]);


    // gpk_rev<D, 3, T, false, false, 2>(
    //   handle, handle.shapes_h[l], handle.shapes_d[l],
    //   handle.shapes_d[l + 1], dinput1.getLdd(), dinput2.getLdd(),
    //   handle.unprocessed_n[unprocessed_idx],
    //   handle.unprocessed_dims_d[unprocessed_idx],
    //   curr_dims[2], curr_dims[1], curr_dims[0], 
    //   handle.ratio[curr_dims[2]][l], handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], 
    //   dinput1.data(), dinput1.getLddv1(), dinput1.getLddv2(), 
    //   dcoarse.data(), dcoarse.getLddv1(), dcoarse.getLddv2(), 
    //   // null, lddv1, lddv2,
    //   dcoeff_f.data(), dcoeff_f.getLddv1(), dcoeff_f.getLddv2(), 
    //   // null, lddv1, lddv2,
    //   dcoeff_c.data(), dcoeff_c.getLddv1(), dcoeff_c.getLddv2(), 
    //   // null, lddv1, lddv2,
    //   dcoeff_r.data(), dcoeff_r.getLddv1(), dcoeff_r.getLddv2(), 
    //   // null, lddv1, lddv2,
    //   dcoeff_cf.data(), dcoeff_cf.getLddv1(), dcoeff_cf.getLddv2(), 
    //   // null, lddv1, lddv2,
    //   dcoeff_rf.data(), dcoeff_rf.getLddv1(), dcoeff_rf.getLddv2(), 
    //   // null, lddv1, lddv2,
    //   dcoeff_rc.data(), dcoeff_rc.getLddv1(), dcoeff_rc.getLddv2(), 
    //   // null, lddv1, lddv2,
    //   dcoeff_rcf.data(), dcoeff_rcf.getLddv1(), dcoeff_rcf.getLddv2(), 
    //   // null, lddv1, lddv2,
    //   0, 0, 0, 
    //   handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
    //   queue_idx,
    //   handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

    // gpuErrchk(cudaDeviceSynchronize());
    GpkRev<D, 3, T, false, false, 2, DeviceType>().Execute(
        SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
        SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
        handle.unprocessed_n[unprocessed_idx],
        SubArray(handle.unprocessed_dims[unprocessed_idx]),
        // unprocessed_dims_subarray,
        curr_dims[2], curr_dims[1], curr_dims[0],
        // ratio_r, ratio_c, ratio_f, 
        SubArray(handle.ratio_array[curr_dims[2]][l]),
        SubArray(handle.ratio_array[curr_dims[1]][l]),
        SubArray(handle.ratio_array[curr_dims[0]][l]),
        dinput1, dcoarse, dcoeff_f, dcoeff_c, dcoeff_r,
        dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf, 
        0, 0, 0, 
        handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
        queue_idx);
    // gpuErrchk(cudaDeviceSynchronize());
  }

  // printf("coeff-restore %u-%dD\n", D_reduced+1, D_reduced+2);
  curr_dims[0] = 0; curr_dims[1] = D_reduced; curr_dims[2] = D_reduced+1;
  dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  calc_coeff_pointers(handle, curr_dims, l, dinput1, 
                      dcoarse, 
                      dcoeff_f, dcoeff_c, dcoeff_r, 
                      dcoeff_cf, dcoeff_rf, dcoeff_rc,
                      dcoeff_rcf);

  if (D - D_reduced == 1) {
    // printf("coeff-restore %u-%dD\n", D_reduced+1, D_reduced+1);
    unprocessed_idx += 1;

    // unprocessed_dims_subarray = SubArray<1, DIM, DeviceType>({(SIZE)handle.unprocessed_n[unprocessed_idx]}, 
    //                                                       handle.unprocessed_dims_d[unprocessed_idx]);
    // ratio_r = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[2]][l]}, handle.ratio[curr_dims[2]][l]);
    // ratio_c = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[1]][l]}, handle.ratio[curr_dims[1]][l]);
    // ratio_f = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[0]][l]}, handle.ratio[curr_dims[0]][l]);


    // gpk_rev<D, 2, T, false, true, 2>(
    //     handle, handle.shapes_h[l], handle.shapes_d[l],
    //     handle.shapes_d[l + 1], doutput.getLdd(), dinput1.getLdd(),
    //     handle.unprocessed_n[unprocessed_idx],
    //     handle.unprocessed_dims_d[unprocessed_idx],
    //     curr_dims[2], curr_dims[1], curr_dims[0], 
    //     handle.ratio[curr_dims[2]][l], handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], 
    //     doutput.data(), doutput.getLddv1(), doutput.getLddv2(), 
    //     dcoarse.data(), dcoarse.getLddv1(), dcoarse.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     dcoeff_f.data(), dcoeff_f.getLddv1(), dcoeff_f.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     dcoeff_c.data(), dcoeff_c.getLddv1(), dcoeff_c.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     dcoeff_r.data(), dcoeff_r.getLddv1(), dcoeff_r.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     dcoeff_cf.data(), dcoeff_cf.getLddv1(), dcoeff_cf.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     dcoeff_rf.data(), dcoeff_rf.getLddv1(), dcoeff_rf.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     dcoeff_rc.data(), dcoeff_rc.getLddv1(), dcoeff_rc.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     dcoeff_rcf.data(), dcoeff_rcf.getLddv1(), dcoeff_rcf.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     0, 0, 0, 
    //     handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
    //     queue_idx,
    //     handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

    // gpuErrchk(cudaDeviceSynchronize());
    GpkRev<D, 2, T, false, true, 2, DeviceType>().Execute(
        SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
        SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
        handle.unprocessed_n[unprocessed_idx],
        SubArray(handle.unprocessed_dims[unprocessed_idx]),
        // unprocessed_dims_subarray,
        curr_dims[2], curr_dims[1], curr_dims[0],
        // ratio_r, ratio_c, ratio_f, 
        SubArray(handle.ratio_array[curr_dims[2]][l]),
        SubArray(handle.ratio_array[curr_dims[1]][l]),
        SubArray(handle.ratio_array[curr_dims[0]][l]), 
        doutput, dcoarse, dcoeff_f, dcoeff_c, dcoeff_r,
        dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf, 
        0, 0, 0, 
        handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
        queue_idx);
    // gpuErrchk(cudaDeviceSynchronize());
  } else { //D - D_reduced >= 2
    // printf("coeff-restore %u-%dD\n", D_reduced+1, D_reduced+2);
    unprocessed_idx += 2;

    // unprocessed_dims_subarray = SubArray<1, DIM, DeviceType>({(SIZE)handle.unprocessed_n[unprocessed_idx]}, 
    //                                                       handle.unprocessed_dims_d[unprocessed_idx]);
    // ratio_r = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[2]][l]}, handle.ratio[curr_dims[2]][l]);
    // ratio_c = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[1]][l]}, handle.ratio[curr_dims[1]][l]);
    // ratio_f = SubArray<1, T, DeviceType>({handle.dofs[curr_dims[0]][l]}, handle.ratio[curr_dims[0]][l]);


    // gpk_rev<D, 3, T, false, true, 2>(
    //     handle, handle.shapes_h[l], handle.shapes_d[l],
    //     handle.shapes_d[l + 1], doutput.getLdd(), dinput1.getLdd(),
    //     handle.unprocessed_n[unprocessed_idx],
    //     handle.unprocessed_dims_d[unprocessed_idx],
    //     curr_dims[2], curr_dims[1], curr_dims[0], 
    //     handle.ratio[curr_dims[2]][l], handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], 
    //     doutput.data(), doutput.getLddv1(), doutput.getLddv2(), 
    //     dcoarse.data(), dcoarse.getLddv1(), dcoarse.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     dcoeff_f.data(), dcoeff_f.getLddv1(), dcoeff_f.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     dcoeff_c.data(), dcoeff_c.getLddv1(), dcoeff_c.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     dcoeff_r.data(), dcoeff_r.getLddv1(), dcoeff_r.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     dcoeff_cf.data(), dcoeff_cf.getLddv1(), dcoeff_cf.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     dcoeff_rf.data(), dcoeff_rf.getLddv1(), dcoeff_rf.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     dcoeff_rc.data(), dcoeff_rc.getLddv1(), dcoeff_rc.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     dcoeff_rcf.data(), dcoeff_rcf.getLddv1(), dcoeff_rcf.getLddv2(), 
    //     // null, lddv1, lddv2,
    //     0, 0, 0, 
    //     handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
    //     queue_idx,
    //     handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

    // gpuErrchk(cudaDeviceSynchronize());
    GpkRev<D, 3, T, false, true, 2, DeviceType>().Execute(
        SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
        SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
        handle.unprocessed_n[unprocessed_idx],
        SubArray(handle.unprocessed_dims[unprocessed_idx]),
        // unprocessed_dims_subarray,
        curr_dims[2], curr_dims[1], curr_dims[0],
        // ratio_r, ratio_c, ratio_f, 
        SubArray(handle.ratio_array[curr_dims[2]][l]),
        SubArray(handle.ratio_array[curr_dims[1]][l]),
        SubArray(handle.ratio_array[curr_dims[0]][l]),
        doutput, dcoarse, dcoeff_f, dcoeff_c, dcoeff_r,
        dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf, 
        0, 0, 0, 
        handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
        queue_idx);
    // gpuErrchk(cudaDeviceSynchronize());
  }

  if (debug_print){ // debug
    PrintSubarray4D("After coeff restore", doutput);
  } //debug

}

template <DIM D, typename T, typename DeviceType>
void calc_correction_nd(Handle<D, T, DeviceType> &handle, SubArray<D, T, DeviceType> dcoeff, 
                        SubArray<D, T, DeviceType> &dcorrection, SIZE l, int queue_idx) {
  int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
  int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(handle.shape[d]) + "_";

  SubArray<D, T, DeviceType> dw_in1 = dcoeff;
  SubArray<D, T, DeviceType> dw_in2 = dcoeff;
  SubArray<D, T, DeviceType> dw_out = dcorrection;

  // start correction calculation
  int prev_dim_r, prev_dim_c, prev_dim_f;
  int curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;

  dw_in1.resize(curr_dim_f, handle.dofs[curr_dim_f][l+1]);
  dw_in2.offset(curr_dim_f, handle.dofs[curr_dim_f][l+1]);
  dw_in2.resize(curr_dim_f, handle.dofs[curr_dim_f][l]-handle.dofs[curr_dim_f][l+1]);
  dw_out.resize(curr_dim_f, handle.dofs[curr_dim_f][l+1]);

  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

  // printf("mass trans 1D\n");
  // lpk_reo_1<D, T>(
  //     handle, handle.shapes_h[l], handle.shapes_h[l + 1],
  //     handle.shapes_d[l], handle.shapes_d[l + 1], dw_in1.getLdd(), dw_out.getLdd(),
  //     handle.processed_n[0], handle.processed_dims_h[0],
  //     handle.processed_dims_d[0], curr_dim_r, curr_dim_c, curr_dim_f,
  //     handle.dist[curr_dim_f][l], handle.ratio[curr_dim_f][l], 
  //     dw_in1.data(), dw_in1.getLddv1(), dw_in1.getLddv2(),
  //     dw_in2.data(), dw_in2.getLddv1(), dw_in2.getLddv2(),
  //     dw_out.data(), dw_out.getLddv1(), dw_out.getLddv2(), queue_idx,
  //     handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);


  // SubArray<1, T, DeviceType> dist_f = SubArray<1, T, DeviceType>({handle.dofs[curr_dim_f][l]}, handle.dist[curr_dim_f][l]);
  // SubArray<1, T, DeviceType> ratio_f = SubArray<1, T, DeviceType>({handle.dofs[curr_dim_f][l]}, handle.ratio[curr_dim_f][l]);
  // gpuErrchk(cudaDeviceSynchronize());
  Lpk1Reo<D, T, DeviceType>().Execute(SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
                                    SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
                                    handle.processed_n[0], 
                                    SubArray<1, SIZE, DeviceType>(handle.processed_dims[0], true), 
                                    curr_dim_r, curr_dim_c, curr_dim_f,
                                    //dist_f, ratio_f,
                                    SubArray(handle.dist_array[curr_dim_f][l]), 
                                    SubArray(handle.ratio_array[curr_dim_f][l]), 
                                    dw_in1, dw_in2, dw_out, 0);
  // gpuErrchk(cudaDeviceSynchronize());

  if (debug_print){ // debug
    PrintSubarray4D(format("decomposition: after MR-1D[{}]", l), dw_out);
  }

  // mass trans 2D
  prev_dim_f = curr_dim_f;
  prev_dim_c = curr_dim_c;
  prev_dim_r = curr_dim_r;
  curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;

  dw_in1 = dw_out;
  dw_in2 = dw_out;
  dw_in1.resize(curr_dim_c, handle.dofs[curr_dim_c][l+1]);
  dw_in2.offset(curr_dim_c, handle.dofs[curr_dim_c][l+1]);
  dw_in2.resize(curr_dim_c, handle.dofs[curr_dim_c][l]-handle.dofs[curr_dim_c][l+1]);
  dw_out.offset(prev_dim_f, handle.dofs[curr_dim_f][l+1]);
  dw_out.resize(curr_dim_c, handle.dofs[curr_dim_c][l+1]);

  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

  // printf("mass trans 2D\n");
  // lpk_reo_2<D, T>(
  //     handle, handle.shapes_h[l], handle.shapes_h[l + 1],
  //     handle.shapes_d[l], handle.shapes_d[l + 1], 
  //     dw_in1.getLdd(), dw_out.getLdd(),
  //     handle.processed_n[1], handle.processed_dims_h[1],
  //     handle.processed_dims_d[1], 
  //     curr_dim_r, curr_dim_c, curr_dim_f,
  //     handle.dist[curr_dim_c][l], handle.ratio[curr_dim_c][l],
  //     dw_in1.data(), dw_in1.getLddv1(), dw_in1.getLddv2(),
  //     dw_in2.data(), dw_in2.getLddv1(), dw_in2.getLddv2(),
  //     dw_out.data(), dw_out.getLddv1(), dw_out.getLddv2(), queue_idx,
  //     handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

  // gpuErrchk(cudaDeviceSynchronize());
  Lpk2Reo<D, T, DeviceType>().Execute(SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
                                    SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
                                    handle.processed_n[1], 
                                    SubArray<1, SIZE, DeviceType>(handle.processed_dims[1], true), 
                                    curr_dim_r, curr_dim_c, curr_dim_f,
                                    //dist_f, ratio_f,
                                    SubArray(handle.dist_array[curr_dim_c][l]), 
                                    SubArray(handle.ratio_array[curr_dim_c][l]), 
                                    dw_in1, dw_in2, dw_out, 0);
  // gpuErrchk(cudaDeviceSynchronize());

  if (debug_print){ // debug
    PrintSubarray4D(format("decomposition: after MR-2D[{}]", l), dw_out);
  }

  // mass trans 3D

  prev_dim_f = curr_dim_f;
  prev_dim_c = curr_dim_c;
  prev_dim_r = curr_dim_r;
  curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;

  dw_in1 = dw_out;
  dw_in2 = dw_out;
  dw_in1.resize(curr_dim_r, handle.dofs[curr_dim_r][l+1]);
  dw_in2.offset(curr_dim_r, handle.dofs[curr_dim_r][l+1]);
  dw_in2.resize(curr_dim_r, handle.dofs[curr_dim_r][l]-handle.dofs[curr_dim_r][l+1]);
  dw_out.offset(prev_dim_c, handle.dofs[curr_dim_c][l+1]);
  dw_out.resize(curr_dim_r, handle.dofs[curr_dim_r][l+1]);

  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

  // printf("mass trans 3D\n");
  // lpk_reo_3<D, T>(
  //     handle, handle.shapes_h[l], handle.shapes_h[l + 1],
  //     handle.shapes_d[l], handle.shapes_d[l + 1], 
  //     dw_in1.getLdd(), dw_out.getLdd(),
  //     handle.processed_n[2], handle.processed_dims_h[2],
  //     handle.processed_dims_d[2], 
  //     curr_dim_r, curr_dim_c, curr_dim_f,
  //     handle.dist[curr_dim_r][l], handle.ratio[curr_dim_r][l],
  //     dw_in1.data(), dw_in1.getLddv1(), dw_in1.getLddv2(),
  //     dw_in2.data(), dw_in2.getLddv1(), dw_in2.getLddv2(),
  //     dw_out.data(), dw_out.getLddv1(), dw_out.getLddv2(), queue_idx,
  //     handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

  // gpuErrchk(cudaDeviceSynchronize());
  Lpk3Reo<D, T, DeviceType>().Execute(SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
                                    SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
                                    handle.processed_n[2], 
                                    SubArray<1, SIZE, DeviceType>(handle.processed_dims[2], true), 
                                    curr_dim_r, curr_dim_c, curr_dim_f,
                                    //dist_f, ratio_f,
                                    SubArray(handle.dist_array[curr_dim_r][l]), 
                                    SubArray(handle.ratio_array[curr_dim_r][l]), 
                                    dw_in1, dw_in2, dw_out, 0);
  // gpuErrchk(cudaDeviceSynchronize());

  if (debug_print){ // debug
    PrintSubarray4D(format("decomposition: after MR-3D[{}]", l), dw_out);
  }

  // mass trans 4D+
  for (int i = 3; i < D; i++) {
    prev_dim_f = curr_dim_f;
    prev_dim_c = curr_dim_c;
    prev_dim_r = curr_dim_r;
    curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = i;
    dw_in1 = dw_out;
    dw_in2 = dw_out;
    dw_in1.resize(curr_dim_r, handle.dofs[curr_dim_r][l+1]);
    dw_in2.offset(curr_dim_r, handle.dofs[curr_dim_r][l+1]);
    dw_in2.resize(curr_dim_r, handle.dofs[curr_dim_r][l]-handle.dofs[curr_dim_r][l+1]);
    dw_out.offset(prev_dim_r, handle.dofs[prev_dim_r][l+1]);
    dw_out.resize(curr_dim_r, handle.dofs[curr_dim_r][l+1]);

    dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
    dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
    dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);
    
    // printf("mass trans %dD\n", i+1);
    // lpk_reo_3<D, T>(
    //     handle, handle.shapes_h[l], handle.shapes_h[l + 1],
    //     handle.shapes_d[l], handle.shapes_d[l + 1], 
    //     dw_in1.getLdd(), dw_out.getLdd(),
    //     handle.processed_n[i], handle.processed_dims_h[i],
    //     handle.processed_dims_d[i], 
    //     curr_dim_r, curr_dim_c, curr_dim_f,
    //     handle.dist[curr_dim_r][l], handle.ratio[curr_dim_r][l],
    //     dw_in1.data(), dw_in1.getLddv1(), dw_in1.getLddv2(),
    //     dw_in2.data(), dw_in2.getLddv1(), dw_in2.getLddv2(),
    //     dw_out.data(), dw_out.getLddv1(), dw_out.getLddv2(), queue_idx,
    //     handle.auto_tuning_mr1[handle.arch][handle.precision][range_lp1]);

    // gpuErrchk(cudaDeviceSynchronize());
    Lpk3Reo<D, T, DeviceType>().Execute(SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
                                      SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
                                      handle.processed_n[i], 
                                      SubArray<1, SIZE, DeviceType>(handle.processed_dims[i], true), 
                                      curr_dim_r, curr_dim_c, curr_dim_f,
                                      //dist_f, ratio_f,
                                      SubArray(handle.dist_array[curr_dim_r][l]), 
                                      SubArray(handle.ratio_array[curr_dim_r][l]), 
                                      dw_in1, dw_in2, dw_out, 0);
    // gpuErrchk(cudaDeviceSynchronize());

    if (debug_print){ // debug
      PrintSubarray4D(format("decomposition: after MR-{}D[{}]", i+1, l), dw_out);
    }
  }

  curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

  // printf("solve tridiag 1D\n");
  // ipk_1<D, T>(
  //     handle, handle.shapes_h[l], handle.shapes_h[l + 1],
  //     handle.shapes_d[l], handle.shapes_d[l + 1], 
  //     dw_out.getLdd(), dw_out.getLdd(), 
  //     handle.processed_n[0], handle.processed_dims_h[0],
  //     handle.processed_dims_d[0], 
  //     curr_dim_r, curr_dim_c, curr_dim_f,
  //     handle.am[curr_dim_f][l + 1], handle.bm[curr_dim_f][l + 1],
  //     handle.dist[curr_dim_f][l + 1], 
  //     dw_out.data(), dw_out.getLddv1(), dw_out.getLddv2(), queue_idx,
  //     handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);

  // gpuErrchk(cudaDeviceSynchronize());
  Ipk1Reo<D, T, DeviceType>().Execute(SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
                                      SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
                                      handle.processed_n[0], 
                                      SubArray<1, SIZE, DeviceType>(handle.processed_dims[0], true), 
                                      curr_dim_r, curr_dim_c, curr_dim_f,
                                      SubArray(handle.am_array[curr_dim_f][l+1]), 
                                      SubArray(handle.bm_array[curr_dim_f][l+1]), 
                                      dw_out, 0);
  // gpuErrchk(cudaDeviceSynchronize());

  if (debug_print){ // debug
    PrintSubarray4D(format("decomposition: after TR-1D[{}]", l), dw_out);
  } //debug

  curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

  // printf("solve tridiag 2D\n");
  // ipk_2<D, T>(
  //     handle, handle.shapes_h[l], handle.shapes_h[l + 1],
  //     handle.shapes_d[l], handle.shapes_d[l + 1], 
  //     dw_out.getLdd(), dw_out.getLdd(),
  //     handle.processed_n[1], handle.processed_dims_h[1],
  //     handle.processed_dims_d[1], 
  //     curr_dim_r, curr_dim_c, curr_dim_f,
  //     handle.am[curr_dim_c][l + 1], handle.bm[curr_dim_c][l + 1],
  //     handle.dist[curr_dim_c][l + 1], 
  //     dw_out.data(), dw_out.getLddv1(), dw_out.getLddv2(), queue_idx,
  //     handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);

  // gpuErrchk(cudaDeviceSynchronize());
  Ipk2Reo<D, T, DeviceType>().Execute(SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
                                      SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
                                      handle.processed_n[1], 
                                      SubArray<1, SIZE, DeviceType>(handle.processed_dims[1], true), 
                                      curr_dim_r, curr_dim_c, curr_dim_f,
                                      SubArray(handle.am_array[curr_dim_c][l+1]), 
                                      SubArray(handle.bm_array[curr_dim_c][l+1]), 
                                      // SubArray(handle.dist_array[curr_dim_f][l+1]), 
                                      dw_out, 0);
  // gpuErrchk(cudaDeviceSynchronize());

  if (debug_print){ // debug
    PrintSubarray4D(format("decomposition: after TR-2D[{}]", l), dw_out);
  } //debug

  curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

  // printf("solve tridiag 3D\n");
  // ipk_3<D, T>(
  //     handle, handle.shapes_h[l], handle.shapes_h[l + 1],
  //     handle.shapes_d[l], handle.shapes_d[l + 1], 
  //     dw_out.getLdd(), dw_out.getLdd(),
  //     handle.processed_n[2], handle.processed_dims_h[2],
  //     handle.processed_dims_d[2], curr_dim_r, curr_dim_c, curr_dim_f,
  //     handle.am[curr_dim_r][l + 1], handle.bm[curr_dim_r][l + 1],
  //     handle.dist[curr_dim_r][l + 1], 
  //     dw_out.data(), dw_out.getLddv1(), dw_out.getLddv2(), queue_idx,
  //     handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);

  // gpuErrchk(cudaDeviceSynchronize());
  Ipk3Reo<D, T, DeviceType>().Execute(SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
                                      SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
                                      handle.processed_n[2], 
                                      SubArray<1, SIZE, DeviceType>(handle.processed_dims[2], true), 
                                      curr_dim_r, curr_dim_c, curr_dim_f,
                                      SubArray(handle.am_array[curr_dim_r][l+1]), 
                                      SubArray(handle.bm_array[curr_dim_r][l+1]), 
                                      // SubArray(handle.dist_array[curr_dim_f][l+1]), 
                                      dw_out, 0);
  // gpuErrchk(cudaDeviceSynchronize());


  if (debug_print){ // debug
    PrintSubarray4D(format("decomposition: after TR-3D[{}]", l), dw_out);
  } //debug

  // mass trans 4D+
  for (int i = 3; i < D; i++) {
    curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = i;
    dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
    dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
    dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);
    // printf("solve tridiag %dD\n", i+1);
    // ipk_3<D, T>(
    //     handle, handle.shapes_h[l], handle.shapes_h[l + 1],
    //     handle.shapes_d[l], handle.shapes_d[l + 1], 
    //     dw_out.getLdd(), dw_out.getLdd(),
    //     handle.processed_n[i], handle.processed_dims_h[i],
    //     handle.processed_dims_d[i], curr_dim_r, curr_dim_c, curr_dim_f,
    //     handle.am[curr_dim_r][l + 1], handle.bm[curr_dim_r][l + 1],
    //     handle.dist[curr_dim_r][l + 1], 
    //     dw_out.data(), dw_out.getLddv1(), dw_out.getLddv2(), queue_idx,
    //     handle.auto_tuning_ts1[handle.arch][handle.precision][range_lp1]);

    // gpuErrchk(cudaDeviceSynchronize());
    Ipk3Reo<D, T, DeviceType>().Execute(SubArray<1, SIZE, DeviceType>(handle.shapes[l], true),
                                      SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true),
                                      handle.processed_n[i], 
                                      SubArray<1, SIZE, DeviceType>(handle.processed_dims[i], true), 
                                      curr_dim_r, curr_dim_c, curr_dim_f,
                                      SubArray(handle.am_array[curr_dim_r][l+1]), 
                                      SubArray(handle.bm_array[curr_dim_r][l+1]), 
                                      // SubArray(handle.dist_array[curr_dim_f][l+1]), 
                                      dw_out, 0);
    // gpuErrchk(cudaDeviceSynchronize());
    if (debug_print){ // debug
      PrintSubarray4D(format("decomposition: after TR-{}D[{}]", i+1, l), dw_out);
    } //debug
  }

  dcorrection = dw_out;
}

template <DIM D, typename T, typename DeviceType>
void decompose(Handle<D, T, DeviceType> &handle, SubArray<D, T, DeviceType>& v, SIZE l_target, int queue_idx) {

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(handle.shape[d]) + "_";
  // std::cout << prefix << std::endl;

  std::vector<SIZE> workspace_shape(D);
  for (DIM d = 0; d < D; d++) workspace_shape[d] = handle.dofs[d][0] + 2;
  std::reverse(workspace_shape.begin(), workspace_shape.end());
  Array<D, T, DeviceType> workspace(workspace_shape);
  // workspace.memset(0); can cause large overhead in HIP
  SubArray w(workspace);

  if (D <= 3) {
    for (int l = 0; l < l_target; ++l) {
      if (debug_print) {
        PrintSubarray("input v", v);
      }

      // DeviceRuntime<DeviceType>::SyncDevice();
      LwpkReo<D, T, COPY, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(handle.shapes[l], true), v, w, queue_idx);
      // DeviceRuntime<DeviceType>::SyncDevice();
      calc_coefficients_3d(handle, w, v, l, queue_idx);
      // DeviceRuntime<DeviceType>::SyncDevice();
      calc_correction_3d(handle, v, w, l, queue_idx);
      // DeviceRuntime<DeviceType>::SyncDevice();

      LwpkReo<D, T, ADD, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true), w, v, queue_idx);
      // DeviceRuntime<DeviceType>::SyncDevice();
      if (debug_print) {
        PrintSubarray("after add", v);
      }
    } // end of loop

    if (debug_print) {
      PrintSubarray("output of decomposition", v);
    }
  }

  if (D > 3) {
    Array<D, T, DeviceType> workspace2(workspace_shape);
    SubArray b(workspace2);
    for (int l = 0; l < l_target; ++l) {
      if (debug_print){ // debug
        PrintSubarray4D("before coeff", v);
      }

      // std::vector<SIZE> shape(handle.D_padded);
      // for (DIM d = 0; d < handle.D_padded; d++) shape[d] = handle.shapes_h[l][d];

      // gpuErrchk(cudaDeviceSynchronize());
      LwpkReo<D, T, COPY, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(handle.shapes[l], true), v, w, queue_idx);
      LwpkReo<D, T, COPY, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(handle.shapes[l], true), v, b, queue_idx);
      // gpuErrchk(cudaDeviceSynchronize());

      calc_coefficients_nd(handle, w, b, v, l, queue_idx);

      if (debug_print){ // debug
        PrintSubarray4D(format("after coeff[%d]", l), v);
      } //debug

      // gpuErrchk(cudaDeviceSynchronize());
      calc_correction_nd(handle, v, w, l, 0);
      // gpuErrchk(cudaDeviceSynchronize());

      LwpkReo<D, T, ADD, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true), w, v, queue_idx);
      // gpuErrchk(cudaDeviceSynchronize());

      if (debug_print){ // debug
        PrintSubarray4D(format("after apply correction[%d]", l), v);
      } //debug
    }
  }
}

template <DIM D, typename T, typename DeviceType>
void recompose(Handle<D, T, DeviceType> &handle, SubArray<D, T, DeviceType>& v,
               SIZE l_target, int queue_idx) {


  std::vector<SIZE> workspace_shape(D);
  for (DIM d = 0; d < D; d++) workspace_shape[d] = handle.dofs[d][0] + 2;
  std::reverse(workspace_shape.begin(), workspace_shape.end());
  Array<D, T, DeviceType> workspace(workspace_shape);
  // workspace.memset(0); // can cause large overhead in HIP
  SubArray w(workspace);
  if (D <= 3) {
    if (debug_print) {
      PrintSubarray("input of recomposition", v);
    }
    std::string prefix = "recomp_";
    if (sizeof(T) == sizeof(double))
      prefix += "d_";
    if (sizeof(T) == sizeof(float))
      prefix += "f_";
    for (int d = 0; d < D; d++)
      prefix += std::to_string(handle.shape[d]) + "_";
    // std::cout << prefix << std::endl;

    for (int l = l_target - 1; l >= 0; l--) {

      calc_correction_3d(handle, v, w, l, 0);


      LwpkReo<D, T, SUBTRACT, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true), w, v, queue_idx);

      
      coefficients_restore_3d(handle, v, w, l, 0);

      LwpkReo<D, T, COPY, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(handle.shapes[l], true), w, v, queue_idx);
      // gpuErrchk(cudaDeviceSynchronize());

      if (debug_print) {
        PrintSubarray("output of recomposition", v);
      }
    }
  }
  if (D > 3) {
    Array<D, T, DeviceType> workspace2(workspace_shape);
    SubArray b(workspace2);
    for (int l = l_target - 1; l >= 0; l--) {

      if (debug_print){ // debug
        PrintSubarray4D(format("before corection[%d]", l), v);
      }

      int curr_dim_r, curr_dim_c, curr_dim_f;
      int lddv1, lddv2;
      int lddw1, lddw2;
      int lddb1, lddb2;
      // un-apply correction
      // std::vector<SIZE> shape(handle.D_padded);
      // for (DIM d = 0; d < handle.D_padded; d++) shape[d] = handle.shapes_h[l][d];

      if (debug_print){ // debug
        PrintSubarray4D(format("before subtract correction[%d]", l), v);
      } //deb

      // gpuErrchk(cudaDeviceSynchronize());

      calc_correction_nd(handle, v, w, l, 0);

      // gpuErrchk(cudaDeviceSynchronize());
      LwpkReo<D, T, SUBTRACT, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(handle.shapes[l+1], true), w, v, queue_idx);
      // gpuErrchk(cudaDeviceSynchronize());

      if (debug_print){ // debug
        PrintSubarray4D(format("after subtract correction[%d]", l), v);
      } //deb

      // gpuErrchk(cudaDeviceSynchronize());
      LwpkReo<D, T, COPY, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(handle.shapes[l], true), v, b, queue_idx);
      LwpkReo<D, T, COPY, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(handle.shapes[l], true), v, w, queue_idx);
      // gpuErrchk(cudaDeviceSynchronize());

      coefficients_restore_nd(handle, w, b, v, l, queue_idx);

    } // loop levels


    if (debug_print){ // debug
      std::vector<SIZE> shape(handle.D_padded);
      // for (DIM d = 0; d < handle.D_padded; d++) shape[d] = handle.shapes_h[0][d];
      PrintSubarray4D(format("final output"), v);
    } //deb
  } // D > 3
}

}