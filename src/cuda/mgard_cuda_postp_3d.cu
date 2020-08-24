#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda_kernels.h"
#include "cuda/mgard_cuda_postp_3d.h"

#include <fstream>

#include <chrono>

namespace mgard_cuda {

template <typename T>
void postp_3D_cuda_cpt(mgard_cuda_handle<T> &handle, T *dv, int lddv1,
                       int lddv2) {

  copy_level_cpt(handle, handle.nrow, handle.ncol, handle.nfib, 1, 1, 1, dv,
                 lddv1, lddv2, handle.dwork, handle.lddwork1, handle.lddwork2,
                 0);

  assign_num_level(handle, handle.nrow, handle.ncol, handle.nfib, handle.nr,
                   handle.nc, handle.nf, 1, 1, 1, handle.dirow, handle.dicol,
                   handle.difib, (T)0.0, handle.dwork, handle.lddwork1,
                   handle.lddwork2, 0);

  handle.sync_all();
  for (int r = 0; r < handle.nrow; r += 1) {
    T *dwork_local = handle.dwork + r * handle.lddwork1 * handle.lddwork2;
    mass_multiply_2_cpt(handle, handle.ncol, handle.nfib, 1, 1, handle.ddist_c,
                        dwork_local, handle.lddwork1, r % handle.num_of_queues);

    restriction_first_2(handle, handle.ncol, handle.nfib, handle.nc,
                        handle.nfib, 1, 1, handle.dicol_p, handle.difib_a,
                        handle.ddist_c, dwork_local, handle.lddwork1,
                        r % handle.num_of_queues);
  }

  handle.sync_all();

  for (int r = 0; r < handle.nr; r += 1) {
    int ir = get_lindex_cuda(handle.nr, handle.nrow, r);
    T *dwork_local = handle.dwork + ir * handle.lddwork1 * handle.lddwork2;
    solve_tridiag_2(handle, handle.ncol, handle.nfib, handle.nc, handle.nfib, 1,
                    1, handle.dicol, handle.difib_a, handle.dcoords_c,
                    dwork_local, handle.lddwork1, r % handle.num_of_queues);
  }
  handle.sync_all();
  for (int c = 0; c < handle.ncol; c += 1) {
    T *dwork_local = handle.dwork + c * handle.lddwork1;
    mass_multiply_2_cpt(handle, handle.nrow, handle.nfib, 1, 1, handle.ddist_r,
                        dwork_local, handle.lddwork1 * handle.lddwork2,
                        c % handle.num_of_queues);

    restriction_first_2(
        handle, handle.nrow, handle.nfib, handle.nr, handle.nfib, 1, 1,
        handle.dirow_p, handle.difib_a, handle.ddist_r, dwork_local,
        handle.lddwork1 * handle.lddwork2, c % handle.num_of_queues);
  }
  handle.sync_all();

  for (int c = 0; c < handle.nc; c += 1) {
    int ic = get_lindex_cuda(handle.nc, handle.ncol, c);
    T *dwork_local = handle.dwork + ic * handle.lddwork1;
    solve_tridiag_2(handle, handle.nrow, handle.nfib, handle.nr, handle.nfib, 1,
                    1, handle.dirow, handle.difib_a, handle.dcoords_r,
                    dwork_local, handle.lddwork1 * handle.lddwork2,
                    c % handle.num_of_queues);
  }
  handle.sync_all();

  for (int r = 0; r < handle.nr; r += 1) {
    int ir = get_lindex_cuda(handle.nr, handle.nrow, r);
    T *dwork_local = handle.dwork + ir * handle.lddwork1 * handle.lddwork2;
    mass_multiply_1_cpt(handle, handle.ncol, handle.nfib, 1, 1, handle.ddist_f,
                        dwork_local, handle.lddwork1, r % handle.num_of_queues);
  }
  handle.sync_all();
  for (int r = 0; r < handle.nr; r += 1) {
    int ir = get_lindex_cuda(handle.nr, handle.nrow, r);
    T *dwork_local = handle.dwork + ir * handle.lddwork1 * handle.lddwork2;
    restriction_first_1(handle, handle.ncol, handle.nfib, handle.nc, handle.nf,
                        1, 1, handle.dicol, handle.difib_p, handle.ddist_f,
                        dwork_local, handle.lddwork1, r % handle.num_of_queues);
    solve_tridiag_1(handle, handle.ncol, handle.nfib, handle.nc, handle.nf, 1,
                    1, handle.dicol, handle.difib, handle.dcoords_f,
                    dwork_local, handle.lddwork1, r % handle.num_of_queues);
  }
  handle.sync_all();
  subtract_level(handle, handle.nrow, handle.ncol, handle.nfib, handle.nr,
                 handle.nc, handle.nf, 1, 1, 1, handle.dirow, handle.dicol,
                 handle.difib, dv, lddv1, lddv2, handle.dwork, handle.lddwork1,
                 handle.lddwork2, 0);

  handle.sync_all();

  prolongate_last_1(handle, handle.nrow, handle.ncol, handle.nfib, handle.nr,
                    handle.nc, handle.nf, handle.dirow, handle.dicol,
                    handle.difib_p, handle.ddist_r, handle.ddist_c,
                    handle.ddist_f, dv, lddv1, lddv2, 0 % handle.num_of_queues);

  prolongate_last_2(handle, handle.nrow, handle.ncol, handle.nfib, handle.nr,
                    handle.nc, handle.nf, handle.dirow, handle.dicol_p,
                    handle.difib, handle.ddist_r, handle.ddist_c,
                    handle.ddist_f, dv, lddv1, lddv2, 1 % handle.num_of_queues);

  prolongate_last_3(handle, handle.nrow, handle.ncol, handle.nfib, handle.nr,
                    handle.nc, handle.nf, handle.dirow_p, handle.dicol,
                    handle.difib, handle.ddist_r, handle.ddist_c,
                    handle.ddist_f, dv, lddv1, lddv2, 2 % handle.num_of_queues);

  prolongate_last_12(handle, handle.nrow, handle.ncol, handle.nfib, handle.nr,
                     handle.nc, handle.nf, handle.dirow, handle.dicol_p,
                     handle.difib_p, handle.ddist_r, handle.ddist_c,
                     handle.ddist_f, dv, lddv1, lddv2,
                     3 % handle.num_of_queues);

  prolongate_last_13(handle, handle.nrow, handle.ncol, handle.nfib, handle.nr,
                     handle.nc, handle.nf, handle.dirow_p, handle.dicol,
                     handle.difib_p, handle.ddist_r, handle.ddist_c,
                     handle.ddist_f, dv, lddv1, lddv2,
                     4 % handle.num_of_queues);

  prolongate_last_23(handle, handle.nrow, handle.ncol, handle.nfib, handle.nr,
                     handle.nc, handle.nf, handle.dirow_p, handle.dicol_p,
                     handle.difib, handle.ddist_r, handle.ddist_c,
                     handle.ddist_f, dv, lddv1, lddv2,
                     5 % handle.num_of_queues);

  prolongate_last_123(handle, handle.nrow, handle.ncol, handle.nfib, handle.nr,
                      handle.nc, handle.nf, handle.dirow_p, handle.dicol_p,
                      handle.difib_p, handle.ddist_r, handle.ddist_c,
                      handle.ddist_f, dv, lddv1, lddv2,
                      6 % handle.num_of_queues);
  handle.sync_all();
}

template void postp_3D_cuda_cpt<double>(mgard_cuda_handle<double> &handle,
                                        double *dv, int lddv1, int lddv2);
template void postp_3D_cuda_cpt<float>(mgard_cuda_handle<float> &handle,
                                       float *dv, int lddv1, int lddv2);

} // namespace mgard_cuda