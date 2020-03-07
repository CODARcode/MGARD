#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_nuni_3d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include "mgard_nuni_3d_cuda_pi_Ql.h"
#include "mgard_nuni_3d_cuda_copy_level_l.h"
#include "mgard_nuni_3d_cuda_assign_num_level_l.h"
#include "mgard_nuni_2d_cuda_cpt_l2_sm.h"
#include <fstream>
#include <cmath>

namespace mgard_common {

}


namespace mgard_cannon {

}

namespace mgard_gen {

void 
refactor_3D_cuda_cpt_l2_sm(int l_target,
                           int nrow,           int ncol,           int nfib,  
                           int nr,             int nc,             int nf, 
                           int * dirow,        int * dicol,        int * difib,
                           double * dv,        int lddv1,          int lddv2,
                           double * dwork,     int lddwork1,       int lddwork2,
                           double * dcoords_x, double * dcoords_y, double * dcoords_z) {

  double * dcv;
  size_t dcv_pitch;
  cudaMalloc3DHelper((void**)&dcv, &dcv_pitch, nc * sizeof(double), nr, nf);
  int lddcv1 = dcv_pitch / sizeof(double);
  int lddcv2 = nr;

  double * dcwork;
  size_t dcwork_pitch;
  cudaMalloc3DHelper((void**)&dcwork, &dcwork_pitch, nc * sizeof(double), nr, nf);
  int lddcwork1 = dcwork_pitch / sizeof(double);
  int lddcwork2 = nr;

  double * dccoords_x;
  cudaMallocHelper((void**)&dccoords_x, nc * sizeof(double));
  double * dccoords_y;
  cudaMallocHelper((void**)&dccoords_y, nr * sizeof(double));
  double * dccoords_z;
  cudaMallocHelper((void**)&dccoords_z, nf * sizeof(double));

  org_to_pow2p1(ncol, nc, dicol, 
                dcoords_x, dccoords_x);
  org_to_pow2p1(nrow, nr, dirow, 
                dcoords_y, dccoords_y);
  org_to_pow2p1(nfib, nf, difib, 
                dcoords_z, dccoords_z);

  int * nf_l = new int[l_target+1];
  int * nr_l = new int[l_target+1];
  int * nc_l = new int[l_target+1];

  double ** ddist_z_l = new double*[l_target+1];
  double ** ddist_y_l = new double*[l_target+1];
  double ** ddist_x_l = new double*[l_target+1];
  
  for (int l = 0; l < l_target+1; l++) {
    int stride = std::pow(2, l);

    nf_l[l] = ceil((float)nf/std::pow(2, l));
    nr_l[l] = ceil((float)nr/std::pow(2, l));
    nc_l[l] = ceil((float)nc/std::pow(2, l));

    cudaMallocHelper((void**)&ddist_x_l[l], nc_l[l] * sizeof(double));
    calc_cpt_dist(nc, stride, dccoords_x, ddist_x_l[l]);

    cudaMallocHelper((void**)&ddist_y_l[l], nr_l[l] * sizeof(double));
    calc_cpt_dist(nr, stride, dccoords_y, ddist_y_l[l]);

    cudaMallocHelper((void**)&ddist_z_l[l], nf_l[l] * sizeof(double));
    calc_cpt_dist(nf, stride, dccoords_z, ddist_z_l[l]);
  }

  int fib_stride;
  int row_stride;
  int col_stride;

  mgard_cuda_ret ret;

  double org_to_pow2p1_time = 0.0;
  double pow2p1_to_org_time = 0.0;

  double pow2p1_to_cpt_time = 0.0;
  double cpt_to_pow2p1_time = 0.0;

  double pi_Ql_cuda_time = 0.0;
  double copy_level_l_cuda_time = 0.0;
  double assign_num_level_l_cuda_time = 0.0;

  ret = org_to_pow2p1(nfib,  nrow,   ncol,
                      nf,    nr,     nc,
                      difib, dirow,  dicol,
                      dv,    lddv1,  lddv2,
                      dcv,   lddcv1, lddcv2);
  org_to_pow2p1_time += ret.time;

  for (int l = 0; l < l_target; ++l) {
    int stride = std::pow(2, l);
    int Cstride = stride * 2;
    
    fib_stride = stride;
    row_stride = stride;
    col_stride = stride;
    ret = pow2p1_to_cpt(nf,         nr,         nc,
                        fib_stride, row_stride, col_stride,
                        dcv,        lddcv1,     lddcv2,
                        dcwork,     lddcwork1,  lddcwork2);
    pow2p1_to_cpt_time += ret.time;

    fib_stride = 1;
    row_stride = 1;
    col_stride = 1;
    ret = pi_Ql_cuda_cpt_sm(nf,           nr,           nc,
                            fib_stride,   row_stride,   col_stride,
                            dcwork,       lddcwork1,    lddcwork2
                            ddist_x_l[l], ddist_y_l[l], ddist_z_l[l], 16);
    pi_Ql_cuda_time += ret;

    ret = cpt_to_pow2p1(nf,         nr,         nc,
                        fib_stride, row_stride, col_stride,
                        dcwork,     lddcwork1,  lddcwork2,
                        dcv,        lddcv1,     lddcv2);
    cpt_to_pow2p1_time += ret.time;



    fib_stride = stride;
    row_stride = stride;
    col_stride = stride;
    copy_level_l_cuda_cpt(nf,         nr,         nc,
                          fib_stride, row_stride, col_stride,
                          dcv,        lddcv1,     lddcv2,
                          dwork,      lddwork1,   lddwork2);

    fib_stride = Cstride;
    row_stride = Cstride;
    col_stride = Cstride;
    ret = pow2p1_to_cpt(nf,         nr,         nc,
                        fib_stride, row_stride, col_stride,
                        dwork,      lddwork1,   lddwork2
                        dcwork,     lddcwork1,  lddcwork2);
    pow2p1_to_cpt_time += ret.time;

    fib_stride = 1;
    row_stride = 1;
    col_stride = 1;
    assign_num_level_l_cuda_cpt(nf,         nr,         nc,
                                fib_stride, row_stride, col_stride,
                                dcwork,     lddcwork1,  lddcwork2,
                                0.0);
    fib_stride = Cstride;
    row_stride = Cstride;
    col_stride = Cstride;
    ret = cpt_to_pow2p1(nf,         nr,         nc,
                        fib_stride, row_stride, col_stride,
                        dcwork,     lddcwork1,  lddcwork2,
                        dwork,      lddwork1,   lddwork2);
    cpt_to_pow2p1_time += ret.time;

    fib_stride = stride;
    for (int f = 0; f < nf; f += fib_stride) {
      double * slice = dwork + lddwork1 * lddwork2 * f;
      int ldslice = lddwork1;
      row_stride = stride;//1;
      col_stride = stride;
      ret = pow2p1_to_cpt(nr,    nc,
                          row_stride, col_stride,
                          slice,      ldslice,
                          dcwork,     lddcwork1);
      pow2p1_to_cpt_time += ret.time;

      row_stride = 1;
      col_stride = 1;
      ret = mass_mult_l_row_cuda_sm(nr_l[0],    nc_l[l],
                                    row_stride, col_stride,
                                    dcwork,     lddcwork1,
                                    ddist_x_l[l],
                                    16, 16);
      mass_mult_l_row_cuda_time += ret.time;

      row_stride = 1;
      col_stride = 1;
      ret = restriction_l_row_cuda_sm(nr_l[0],     nc_l[l],
                                      row_stride,  col_stride,
                                      dcwork,      lddcwork1,
                                      ddist_x_l[l],
                                      16, 16);
      restriction_l_row_cuda_time += ret.time;

      row_stride = 1;
      col_stride = 2;//1;
      ret = solve_tridiag_M_l_row_cuda_sm(nr_l[0],    nc_l[l],
                                          row_stride, col_stride,
                                          dcwork,     lddcwork1,
                                          ddist_x_l[l+1],
                                          16, 16);
      solve_tridiag_M_l_row_cuda_time += ret.time;

      row_stride = 1;
      col_stride = 2;
      ret = mass_mult_l_col_cuda_sm(nr_l[l],     nc_l[l],
                                   row_stride,  col_stride,
                                   dcwork,     lddcwork1,
                                   ddist_y_l[l],
                                   16, 16);
      mass_mult_l_col_cuda_time += ret.time;

    }
        

  }

  ret = pow2p1_to_org(nfib, nrow,  ncol,
                      nf, nr, nc,
                      difib, dirow, dicol,
                      dcv, lddcv1, lddcv2,
                      dv, lddv1, lddv2,);
  pow2p1_to_org_time += ret.time;

}


}