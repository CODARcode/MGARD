#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda_kernels.h"
#include "cuda/mgard_cuda_refactor_3d.h"

#include <iostream>

#include <chrono>
namespace mgard_cuda {

template <typename T>
void refactor_3D_cuda_cpt(mgard_cuda_handle<T> &handle, T *dv, int lddv1,
                          int lddv2) {

  T *dcv;
  size_t dcv_pitch;
  cudaMalloc3DHelper((void **)&dcv, &dcv_pitch, handle.nf * sizeof(T),
                     handle.nc, handle.nr);
  int lddcv1 = dcv_pitch / sizeof(T);
  int lddcv2 = handle.nc;

  org_to_pow2p1(handle, handle.nrow, handle.ncol, handle.nfib, handle.nr,
                handle.nc, handle.nf, handle.dirow, handle.dicol, handle.difib,
                dv, lddv1, lddv2, dcv, lddcv1, lddcv2, 0);
  handle.sync_all();
  for (int l = 0; l < handle.l_target; ++l) {
    int stride = std::pow(2, l);
    int Cstride = stride * 2;

    pow2p1_to_cpt(handle, handle.nr, handle.nc, handle.nf, stride, stride,
                  stride, dcv, lddcv1, lddcv2, handle.dwork, handle.lddwork1,
                  handle.lddwork2, 0);

    pi_Ql_cpt(handle, handle.nr_l[l], handle.nc_l[l], handle.nf_l[l], 1, 1, 1,
              handle.ddist_r_l[l], handle.ddist_c_l[l], handle.ddist_f_l[l],
              handle.dwork, handle.lddwork1, handle.lddwork2, 0);

    cpt_to_pow2p1(handle, handle.nr, handle.nc, handle.nf, stride, stride,
                  stride, handle.dwork, handle.lddwork1, handle.lddwork2, dcv,
                  lddcv1, lddcv2, 0);

    copy_level_cpt(handle, handle.nr, handle.nc, handle.nf, stride, stride,
                   stride, dcv, lddcv1, lddcv2, handle.dwork, handle.lddwork1,
                   handle.lddwork2, 0);

    assign_num_level_cpt(handle, handle.nr_l[0], handle.nc_l[0], handle.nf_l[0],
                         Cstride, Cstride, Cstride, (T)0.0, handle.dwork,
                         handle.lddwork1, handle.lddwork2, 0);

    handle.sync_all();

    for (int f = 0; f < handle.nf; f += stride) {
      int queue_idx = (f / stride) % handle.num_of_queues;

      T *slice = handle.dwork + f;
      int ldslice = handle.lddwork2;

      pow2p1_to_cpt(handle, handle.nr * handle.lddwork1,
                    handle.nc * handle.lddwork1, stride * handle.lddwork1,
                    stride * handle.lddwork1, slice, ldslice,
                    handle.dcwork_2d_rc[queue_idx],
                    handle.lddcwork_2d_rc[queue_idx], queue_idx);

      mass_multiply_1_cpt(handle, handle.nr_l[l], handle.nc_l[l], 1, 1,
                          handle.ddist_c_l[l], handle.dcwork_2d_rc[queue_idx],
                          handle.lddcwork_2d_rc[queue_idx], queue_idx);

      restriction_1_cpt(handle, handle.nr_l[l], handle.nc_l[l], 1, 1,
                        handle.ddist_c_l[l], handle.dcwork_2d_rc[queue_idx],
                        handle.lddcwork_2d_rc[queue_idx], queue_idx);

      solve_tridiag_1_cpt(handle, handle.nr_l[l], handle.nc_l[l], 1, 2,
                          handle.ddist_c_l[l + 1], handle.am_col[queue_idx],
                          handle.bm_col[queue_idx],
                          handle.dcwork_2d_rc[queue_idx],
                          handle.lddcwork_2d_rc[queue_idx], queue_idx);

      mass_multiply_2_cpt(handle, handle.nr_l[l], handle.nc_l[l], 1, 2,
                          handle.ddist_r_l[l], handle.dcwork_2d_rc[queue_idx],
                          handle.lddcwork_2d_rc[queue_idx], queue_idx);

      restriction_2_cpt(handle, handle.nr_l[l], handle.nc_l[l], 1, 2,
                        handle.ddist_r_l[l], handle.dcwork_2d_rc[queue_idx],
                        handle.lddcwork_2d_rc[queue_idx], queue_idx);

      solve_tridiag_2_cpt(handle, handle.nr_l[l], handle.nc_l[l], 2, 2,
                          handle.ddist_r_l[l + 1], handle.am_row[queue_idx],
                          handle.bm_row[queue_idx],
                          handle.dcwork_2d_rc[queue_idx],
                          handle.lddcwork_2d_rc[queue_idx], queue_idx);

      cpt_to_pow2p1(handle, handle.nr * handle.lddwork1,
                    handle.nc * handle.lddwork1, stride * handle.lddwork1,
                    stride * handle.lddwork1, handle.dcwork_2d_rc[queue_idx],
                    handle.lddcwork_2d_rc[queue_idx], slice, ldslice,
                    queue_idx);
    }

    handle.sync_all();

    for (int r = 0; r < handle.nr; r += Cstride) {
      int queue_idx = (r / Cstride) % handle.num_of_queues;
      T *slice = handle.dwork + r * handle.lddwork1 * handle.lddwork2;
      int ldslice = handle.lddwork1;

      pow2p1_to_cpt(handle, handle.nc, handle.nf, Cstride, stride, slice,
                    ldslice, handle.dcwork_2d_cf[queue_idx],
                    handle.lddcwork_2d_cf[queue_idx], queue_idx);

      mass_multiply_1_cpt(handle, handle.nc_l[l + 1], handle.nf_l[l], 1, 1,
                          handle.ddist_f_l[l], handle.dcwork_2d_cf[queue_idx],
                          handle.lddcwork_2d_cf[queue_idx], queue_idx);

      restriction_1_cpt(handle, handle.nc_l[l + 1], handle.nf_l[l], 1, 1,
                        handle.ddist_f_l[l], handle.dcwork_2d_cf[queue_idx],
                        handle.lddcwork_2d_cf[queue_idx], queue_idx);

      solve_tridiag_1_cpt(handle, handle.nc_l[l + 1], handle.nf_l[l], 1, 2,
                          handle.ddist_f_l[l + 1], handle.am_fib[queue_idx],
                          handle.bm_fib[queue_idx],
                          handle.dcwork_2d_cf[queue_idx],
                          handle.lddcwork_2d_cf[queue_idx], queue_idx);

      cpt_to_pow2p1(handle, handle.nc, handle.nf, Cstride, stride,
                    handle.dcwork_2d_cf[queue_idx],
                    handle.lddcwork_2d_cf[queue_idx], slice, ldslice,
                    queue_idx);
    }

    handle.sync_all();

    add_level_cpt(handle, handle.nr, handle.nc, handle.nf, Cstride, Cstride,
                  Cstride, dcv, lddcv1, lddcv2, handle.dwork, handle.lddwork1,
                  handle.lddwork2, 0);

  } // end of loop

  pow2p1_to_org(handle, handle.nrow, handle.ncol, handle.nfib, handle.nr,
                handle.nc, handle.nf, handle.dirow, handle.dicol, handle.difib,
                dcv, lddcv1, lddcv2, dv, lddv1, lddv2, 0);

  cudaFreeHelper(dcv);
}

template void refactor_3D_cuda_cpt<double>(mgard_cuda_handle<double> &handle,
                                           double *dv, int lddv1, int lddv2);
template void refactor_3D_cuda_cpt<float>(mgard_cuda_handle<float> &handle,
                                          float *dv, int lddv1, int lddv2);
} // namespace mgard_cuda