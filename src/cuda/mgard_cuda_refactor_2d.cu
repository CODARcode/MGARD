#include "cuda/mgard_cuda_refactor_2d.h"
#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda_kernels.h"
#include <fstream>



namespace mgard_cuda {

template <typename T>
void 
refactor_2D_cuda(mgard_cuda_handle<T> & handle, T * dv, int lddv){

  for (int l = 0; l < handle.l_target; ++l) {
    int stride = std::pow(2, l); // current stride
    int Cstride = stride * 2;    // coarser stride

    pi_Ql(handle,
          handle.nrow, handle.ncol,
          handle.nr, handle.nc,
          stride, stride,
          handle.dirow, handle.dicol,
          handle.dcoords_r, handle.dcoords_c,
          dv, lddv, 
          0);

    copy_level(handle,
               handle.nrow,  handle.ncol,
               handle.nr,    handle.nc,
               stride,       stride,
               handle.dirow, handle.dicol,
               dv,           lddv, 
               handle.dwork, handle.lddwork, 
               0);

    assign_num_level(handle,
                     handle.nrow,  handle.ncol,
                     handle.nr,    handle.nc,
                     Cstride,      Cstride,
                     handle.dirow, handle.dicol,
                     (T)0.0,
                     handle.dwork, handle.lddwork,
                     0);

    mass_multiply_1(handle,
                    handle.nrow,      handle.ncol,
                    handle.nr,        handle.nc,
                    stride,           stride,
                    handle.dirow,     handle.dicol,
                    handle.dcoords_c,
                    handle.dwork,     handle.lddwork,
                    0);

    restriction_1(handle,
              handle.nrow,       handle.ncol,
              handle.nr,         handle.nc,
              stride, stride,
              handle.dirow,      handle.dicol,
              handle.dcoords_c,
              handle.dwork,      handle.lddwork,
              0);

    solve_tridiag_1(handle,
                handle.nrow,       handle.ncol,
                handle.nr,         handle.nc,
                stride, Cstride,
                handle.dirow,      handle.dicol,
                handle.dcoords_c,
                handle.dwork,      handle.lddwork,
                0);

    mass_multiply_2(handle,
                    handle.nrow,       handle.ncol,
                    handle.nr,         handle.nc,
                    stride, Cstride,
                    handle.dirow,      handle.dicol,
                    handle.dcoords_r,
                    handle.dwork,      handle.lddwork,
                    0);

    restriction_2(handle,
              handle.nrow,       handle.ncol,
              handle.nr,         handle.nc,
              stride, Cstride,
              handle.dirow,      handle.dicol,
              handle.dcoords_r,
              handle.dwork,      handle.lddwork,
              0);

    solve_tridiag_2(handle,
                handle.nrow,       handle.ncol,
                handle.nr,         handle.nc,
                Cstride, Cstride,
                handle.dirow,      handle.dicol,
                handle.dcoords_r,
                handle.dwork,      handle.lddwork,
                0);
    
    add_level(handle,
              handle.nrow,  handle.ncol, 
              handle.nr,    handle.nc,
              Cstride,      Cstride,
              handle.dirow, handle.dicol,
              dv,           lddv, 
              handle.dwork, handle.lddwork, 
              0);

  }// end of loop
}

template void 
refactor_2D_cuda<double>(mgard_cuda_handle<double> & handle, double * dv, int lddv);
template void 
refactor_2D_cuda<float>(mgard_cuda_handle<float> & handle, float * dv, int lddv);



template <typename T>
void
refactor_2D_cuda_cpt(mgard_cuda_handle<T> & handle, T * dv, int lddv) {


  T * dcv;
  size_t dcv_pitch;
  cudaMallocPitchHelper((void**)&dcv, &dcv_pitch, handle.nc * sizeof(T), handle.nr);
  int lddcv = dcv_pitch / sizeof(T);

  org_to_pow2p1(handle,
                handle.nrow,  handle.ncol,
                handle.nr,    handle.nc,
                handle.dirow, handle.dicol,
                dv,    lddv,
                dcv,   lddcv, 
                0);

  for (int l = 0; l < handle.l_target; ++l) {
    int stride = std::pow(2, l); // current stride
    int Cstride = stride * 2;    // coarser stride

    pi_Ql_cpt(handle,
              handle.nr, handle.nc,
              stride, stride,
              handle.ddist_r_l[l], handle.ddist_c_l[l],
              dcv, lddcv, 
              0);
    
    pow2p1_to_cpt_num_assign(handle,
                             handle.nr,    handle.nc,
                             stride,   stride,
                             (T)0.0,
                             dcv,        lddcv,  
                             handle.dwork,     handle.lddwork,
                             0);

    mass_multiply_1_cpt(handle,
                        handle.nr_l[l],      handle.nc_l[l],
                        1,                   1,
                        handle.ddist_c_l[l],
                        handle.dwork,        handle.lddwork,
                        0);

    restriction_1_cpt(handle,
                      handle.nr_l[l],  handle.nc_l[l],
                      1, 1,
                      handle.ddist_c_l[l],
                      handle.dwork, handle.lddwork,
                      0);
 
    solve_tridiag_1_cpt(handle, 
                    handle.nr_l[l],    handle.nc_l[l],
                    1, 2,
                    handle.ddist_c_l[l+1],
                    handle.am_col[0], handle.bm_col[0],
                    handle.dwork,     handle.lddwork,
                    0);

    mass_multiply_2_cpt(handle,
                        handle.nr_l[l],     handle.nc_l[l],
                        1,  2,
                        handle.ddist_r_l[l],
                        handle.dwork,     handle.lddwork,
                        0);

    restriction_2_cpt(handle,
                      handle.nr_l[l],  handle.nc_l[l],
                      1, 2,
                      handle.ddist_r_l[l],
                      handle.dwork, handle.lddwork,
                      0);

    solve_tridiag_2_cpt(handle, 
                    handle.nr_l[l],    handle.nc_l[l],
                    2, 2,
                    handle.ddist_r_l[l+1],
                    handle.am_row[0], handle.bm_row[0],
                    handle.dwork,     handle.lddwork,
                    0);

    cpt_to_pow2p1_add(handle,
                      handle.nr,         handle.nc,
                      2, 2, 
                      Cstride, Cstride,
                      handle.dwork,     handle.lddwork,
                      dcv,        lddcv,
                      0);
  }

  pow2p1_to_org(handle,
                handle.nrow,  handle.ncol,
                handle.nr,    handle.nc,
                handle.dirow, handle.dicol,
                dcv,   lddcv,
                dv,    lddv, 
                0);
  cudaFreeHelper(dcv);

}

template void
refactor_2D_cuda_cpt<double>(mgard_cuda_handle<double> & handle, double * dv, int lddv);
template void
refactor_2D_cuda_cpt<float>(mgard_cuda_handle<float> & handle, float * dv, int lddv);

}