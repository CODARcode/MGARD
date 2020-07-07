#include "cuda/mgard_cuda_prep_2d.h"
#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda_kernels.h"
#include <fstream>

namespace mgard_cuda {

template <typename T> 
void 
prep_2D_cuda(mgard_cuda_handle<T> & handle, T * dv, int lddv) {

  pi_Ql_first_1(handle,
                handle.nrow, handle.ncol,
                handle.nr, handle.nc, 
                handle.dirow, handle.dicol_p,
                handle.ddist_r, handle.ddist_c,
                dv,        lddv,
                0);
  pi_Ql_first_2(handle,
                handle.nrow, handle.ncol,
                handle.nr, handle.nc, 
                handle.dirow_p, handle.dicol,
                handle.ddist_r, handle.ddist_c,
                dv,        lddv,
                0);
  pi_Ql_first_12(handle,
                handle.nrow, handle.ncol,
                handle.nr, handle.nc, 
                handle.dirow_p, handle.dicol_p,
                handle.ddist_r, handle.ddist_c,
                dv, lddv,
                0);

  copy_level_cpt(handle,
                 handle.nrow, handle.ncol, 
                 1, 1,
                 dv, lddv,
                 handle.dwork, handle.lddwork,
                 0);

  assign_num_level(handle,
                   handle.nrow, handle.ncol,
                   handle.nr, handle.nc,
                   1, 1,
                   handle.dirow, handle.dicol,
                   (T)0.0,
                   handle.dwork, handle.lddwork,
                   0);

  mass_multiply_1(handle,
                  handle.nrow, handle.ncol,
                  handle.nrow, handle.ncol,
                  1, 1,
                  handle.dirow_a, handle.dicol_a,
                  handle.dcoords_c,
                  handle.dwork, handle.lddwork,
                  0);

  restriction_first_1(handle,
                      handle.nrow, handle.ncol,
                      handle.nrow, handle.nc,
                      1, 1,
                      handle.dirow_a, handle.dicol_p, 
                      handle.ddist_c,
                      handle.dwork, handle.lddwork,
                      0);

  solve_tridiag_1(handle,
                handle.nrow, handle.ncol,
                handle.nr, handle.nc,
                1, 1,
                handle.dirow, handle.dicol, 
                handle.dcoords_c,
                handle.dwork, handle.lddwork, 
                0);

  mass_multiply_2(handle,
                  handle.nrow, handle.ncol,
                  handle.nrow, handle.ncol,
                  1, 1,
                  handle.dirow_a, handle.dicol_a,
                  handle.dcoords_r,
                  handle.dwork, handle.lddwork,
                  0);

  restriction_first_2(handle,
                      handle.nrow, handle.ncol,
                      handle.nr, handle.ncol,
                      1, 1,
                       handle.dirow_p, handle.dicol_a, 
                       handle.ddist_r,
                       handle.dwork, handle.lddwork,
                       0);

  solve_tridiag_2(handle, 
                handle.nrow, handle.ncol,
                handle.nr, handle.nc,
                1, 1,
                handle.dirow, handle.dicol,
                handle.dcoords_r,
                handle.dwork, handle.lddwork, 
                0);

  add_level(handle,
            handle.nrow, handle.ncol, 
            handle.nr, handle.nc, 
            1, 1, 
            handle.dirow, handle.dicol, 
            dv, lddv, 
            handle.dwork, handle.lddwork, 
            0);
}


template void 
prep_2D_cuda<double>(mgard_cuda_handle<double> & handle, double * dv, int lddv);
template void 
prep_2D_cuda<float>(mgard_cuda_handle<float> & handle, float * dv, int lddv);

template <typename T>
void 
prep_2D_cuda_cpt(mgard_cuda_handle<T> & handle, T * dv, int lddv){

  T * dcwork;
  size_t dcwork_pitch;
  cudaMallocPitchHelper((void**)&dcwork, &dcwork_pitch, handle.nc * sizeof(T), handle.nr);
  int lddcwork = dcwork_pitch / sizeof(T);

  pi_Ql_first_1(handle,
                handle.nrow, handle.ncol,
                handle.nr, handle.nc, 
                handle.dirow, handle.dicol_p,
                handle.ddist_r, handle.ddist_c,
                dv, lddv,
                0);
  pi_Ql_first_2(handle,
                handle.nrow, handle.ncol,
                handle.nr, handle.nc, 
                handle.dirow_p, handle.dicol,
                handle.ddist_r, handle.ddist_c,
                dv,        lddv,
                0);
  pi_Ql_first_12(handle,
                handle.nrow, handle.ncol,
                handle.nr, handle.nc, 
                handle.dirow_p, handle.dicol_p,
                handle.ddist_r, handle.ddist_c,
                dv, lddv,
                0);

  copy_level_cpt(handle,
                 handle.nrow, handle.ncol, 
                 1, 1,
                 dv, lddv,
                 handle.dwork, handle.lddwork,
                 0);

  assign_num_level(handle,
                   handle.nrow, handle.ncol,
                   handle.nr, handle.nc,
                   1, 1,
                   handle.dirow, handle.dicol,
                   (T)0.0,
                   handle.dwork, handle.lddwork, 
                   0);


  mass_multiply_1_cpt(handle,
                      handle.nrow, handle.ncol,
                      1, 1,
                      handle.ddist_c,
                      handle.dwork, handle.lddwork,
                      0);

  restriction_first_1(handle,
                      handle.nrow, handle.ncol,
                      handle.nrow, handle.nc,
                      1, 1,
                      handle.dirow_a, handle.dicol_p, 
                      handle.ddist_c,
                      handle.dwork, handle.lddwork,
                      0);

  org_to_pow2p1(handle,
                handle.nrow, handle.ncol,
                handle.nr, handle.nc,
                handle.dirow, handle.dicol,
                handle.dwork, handle.lddwork,
                dcwork, lddcwork, 
                0);

  solve_tridiag_1_cpt(handle, 
                    handle.nr, handle.nc,
                    1, 1,
                    handle.ddist_c_l[0],
                    handle.am_col[0], handle.bm_col[0],
                    dcwork, lddcwork,
                    0);

  pow2p1_to_org(handle,
                handle.nrow, handle.ncol,
                handle.nr, handle.nc,
                handle.dirow, handle.dicol,
                dcwork, lddcwork,
                handle.dwork, handle.lddwork, 
                0);

  mass_multiply_2_cpt(handle,
                      handle.nrow, handle.ncol,
                      1, 1,
                      handle.ddist_r,
                      handle.dwork, handle.lddwork,
                      0);  

  restriction_first_2(handle,
                      handle.nrow, handle.ncol,
                      handle.nr, handle.ncol,
                      1, 1,
                       handle.dirow_p, handle.dicol_a, 
                       handle.ddist_r,
                       handle.dwork, handle.lddwork,
                       0);

  org_to_pow2p1(handle,
                handle.nrow, handle.ncol,
                handle.nr, handle.nc,
                handle.dirow, handle.dicol,
                handle.dwork, handle.lddwork,
                dcwork, lddcwork, 
                0);

  solve_tridiag_2_cpt(handle,
                    handle.nr, handle.nc,
                    1, 1,
                    handle.ddist_r_l[0],
                    handle.am_row[0], handle.bm_row[0],
                    dcwork, lddcwork,
                    0);

  pow2p1_to_org(handle,
                handle.nrow, handle.ncol,
                handle.nr, handle.nc,
                handle.dirow, handle.dicol,
                dcwork, lddcwork,
                handle.dwork, handle.lddwork, 
                0);

  add_level(handle, 
            handle.nrow, handle.ncol, 
            handle.nr, handle.nc, 
            1, 1,
            handle.dirow, handle.dicol, 
            dv, lddv, 
            handle.dwork, handle.lddwork,
            0);
}

template void 
prep_2D_cuda_cpt<double>(mgard_cuda_handle<double> & handle, double * dv, int lddv); 
template void 
prep_2D_cuda_cpt<float>(mgard_cuda_handle<float> & handle, float * dv, int lddv); 

}