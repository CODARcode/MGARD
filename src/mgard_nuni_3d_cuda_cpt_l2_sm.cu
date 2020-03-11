#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_nuni_3d_cuda_cpt_l2_sm.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include "mgard_nuni_3d_cuda_pi_Ql.h"
#include "mgard_nuni_3d_cuda_copy_level_l.h"
#include "mgard_nuni_3d_cuda_assign_num_level_l.h"
#include "mgard_nuni_3d_cuda_add_level_l.h"
#include "mgard_nuni_2d_cuda_mass_mult_l.h"
#include "mgard_nuni_2d_cuda_restriction_l.h"
#include "mgard_nuni_2d_cuda_solve_tridiag_M_l.h"



#include <fstream>
#include <cmath>

namespace mgard_common {
inline int get_index3(const int ncol, const int nfib, const int i, const int j,
                      const int k) {
  return (ncol * i + j) * nfib + k;
}
 
}


namespace mgard_cannon {

}

namespace mgard_gen {

inline int get_lindex(const int n, const int no, const int i) {
  // no: original number of points
  // n : number of points at next coarser level (L-1) with  2^k+1 nodes
  int lindex;
  //    return floor((no-2)/(n-2)*i);
  if (i != n - 1) {
    lindex = floor(((double)no - 2.0) / ((double)n - 2.0) * i);
  } else if (i == n - 1) {
    lindex = no - 1;
  }

  return lindex;
}



mgard_cuda_ret 
refactor_3D_cuda_cpt_l2_sm(int l_target,
                           int nrow,           int ncol,           int nfib,
                           int nr,             int nc,             int nf,             
                           int * dirow,        int * dicol,        int * difib,
                           double * dv,        int lddv1,          int lddv2,
                           double * dwork,     int lddwork1,       int lddwork2,
                           double * dcoords_r, double * dcoords_c, double * dcoords_f) {

  //for debug
    
  double * v = new double[nrow*ncol*nfib];
  std::vector<double> work(nrow * ncol * nfib),work2d(nrow * ncol * nfib);
  std::vector<double> v2d(nrow * ncol), fib_vec(nfib);
  std::vector<double> row_vec(ncol);
  std::vector<double> col_vec(nrow);
  cudaMemcpy3DHelper(v, nfib  * sizeof(double), nfib * sizeof(double), ncol,
                     dv, lddv1 * sizeof(double), nfib * sizeof(double), ncol,
                     nfib * sizeof(double), ncol, nrow,
                     D2H);
  std::vector<double> coords_r(nrow), coords_c(ncol), coords_f(nfib);

  cudaMemcpyHelper(coords_r.data(), dcoords_r, nrow * sizeof(double), D2H);
  cudaMemcpyHelper(coords_c.data(), dcoords_c, ncol * sizeof(double), D2H);
  cudaMemcpyHelper(coords_f.data(), dcoords_f, nfib * sizeof(double), D2H);

  double * dv2;
  size_t dv2_pitch;
  cudaMalloc3DHelper((void**)&dv2, &dv2_pitch, nfib * sizeof(double), ncol, nrow);
  int lddv21 = dv2_pitch / sizeof(double);
  int lddv22 = ncol;

  cudaMemcpy3DHelper(dv2, lddv21  * sizeof(double), nfib * sizeof(double), lddv22,
                     dv, lddv1 * sizeof(double), nfib * sizeof(double), ncol,
                     nfib * sizeof(double), ncol, nrow,
                     D2D);



  double * dwork2;
  size_t dwork2_pitch;
  cudaMalloc3DHelper((void**)&dwork2, &dwork2_pitch, nfib * sizeof(double), ncol, nrow);
  int lddwork21 = dwork2_pitch / sizeof(double);
  int lddwork22 = ncol;

  double * dcwork2;
  size_t dcwork2_pitch;
  cudaMalloc3DHelper((void**)&dcwork2, &dcwork2_pitch, nf * sizeof(double), nc, nr);
  int lddcwork21 = dcwork2_pitch / sizeof(double);
  int lddcwork22 = nc;
  //end for debug


  double * dcv;
  size_t dcv_pitch;
  cudaMalloc3DHelper((void**)&dcv, &dcv_pitch, nf * sizeof(double), nc, nr);
  int lddcv1 = dcv_pitch / sizeof(double);
  int lddcv2 = nc;

  double * dcwork;
  size_t dcwork_pitch;
  cudaMalloc3DHelper((void**)&dcwork, &dcwork_pitch, nf * sizeof(double), nc, nr);
  int lddcwork1 = dcwork_pitch / sizeof(double);
  int lddcwork2 = nc;

  double * dccoords_r;
  cudaMallocHelper((void**)&dccoords_r, nr * sizeof(double));
  double * dccoords_c;
  cudaMallocHelper((void**)&dccoords_c, nc * sizeof(double));
  double * dccoords_f;
  cudaMallocHelper((void**)&dccoords_f, nf * sizeof(double));

  org_to_pow2p1(nrow, nr, dirow, dcoords_r, dccoords_r);
  org_to_pow2p1(ncol, nc, dicol, dcoords_c, dccoords_c);
  org_to_pow2p1(nfib, nf, difib, dcoords_f, dccoords_f);
  

  int * nr_l = new int[l_target+1];
  int * nc_l = new int[l_target+1];
  int * nf_l = new int[l_target+1];

  double ** ddist_r_l = new double*[l_target+1];
  double ** ddist_c_l = new double*[l_target+1];
  double ** ddist_f_l = new double*[l_target+1];
  
  for (int l = 0; l < l_target+1; l++) {
    int stride = std::pow(2, l);

    nr_l[l] = ceil((float)nr/std::pow(2, l));
    nc_l[l] = ceil((float)nc/std::pow(2, l));
    nf_l[l] = ceil((float)nf/std::pow(2, l));

    cudaMallocHelper((void**)&ddist_r_l[l], nr_l[l] * sizeof(double));
    calc_cpt_dist(nr, stride, dccoords_r, ddist_r_l[l]);

    cudaMallocHelper((void**)&ddist_c_l[l], nc_l[l] * sizeof(double));
    calc_cpt_dist(nc, stride, dccoords_c, ddist_c_l[l]);

    cudaMallocHelper((void**)&ddist_f_l[l], nf_l[l] * sizeof(double));
    calc_cpt_dist(nf, stride, dccoords_f, ddist_f_l[l]);
  }

  int row_stride;
  int col_stride;
  int fib_stride;
  
  mgard_cuda_ret ret;

  double org_to_pow2p1_time = 0.0;
  double pow2p1_to_org_time = 0.0;

  double pow2p1_to_cpt_time = 0.0;
  double cpt_to_pow2p1_time = 0.0;

  double pi_Ql_cuda_time = 0.0;
  double copy_level_l_cuda_time = 0.0;
  double assign_num_level_l_cuda_time = 0.0;

  double mass_mult_l_row_cuda_time = 0.0;
  double restriction_l_row_cuda_time = 0.0;
  double solve_tridiag_M_l_row_cuda_time = 0.0;

  double mass_mult_l_col_cuda_time = 0.0;
  double restriction_l_col_cuda_time = 0.0;
  double solve_tridiag_M_l_col_cuda_time = 0.0;

  double add_level_cuda_time = 0.0;

  ret = org_to_pow2p1(nrow,   ncol,  nfib,  
                      nr,     nc,    nf,
                      dirow,  dicol, difib,
                      dv,    lddv1,  lddv2,
                      dcv,   lddcv1, lddcv2);
  org_to_pow2p1_time += ret.time;

  // printf("start v:\n");
  // print_matrix_cuda(1,   1,  10,
  //                   dv,    lddv1,  lddv2,
  //                   nfib); 
  // printf("start cv:\n");
  // print_matrix_cuda(1,     1,    10,
  //                   dcv,   lddcv1, lddcv2, 
  //                   nf); 
  std::cout << "l_target = " << l_target << std::endl;
  for (int l = 0; l < l_target; ++l) {
    std::cout << "l = " << l << std::endl;
    int stride = std::pow(2, l);
    int Cstride = stride * 2;

    // print_matrix(nfib,  nrow,   ncol,
    //              v,  ncol, nrow);


    pi_Ql3D(nr, nc, nf, nrow,   ncol, nfib, l, v, coords_c, coords_r, coords_f,
            row_vec, col_vec, fib_vec);

    mgard_gen::copy3_level_l(l, v, work.data(), nr, nc, nf, nrow, ncol, nfib);
    mgard_gen::assign3_level_l(l + 1, work.data(), 0.0, nr, nc, nf, nrow, ncol,
                               nfib);

    for (int kfib = 0; kfib < nf; kfib += stride) {
      //           int kf = kfib;
      int kf = get_lindex(nf, nfib,
                          kfib); // get the real location of logical index irow
      mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kf);
      mgard_gen::refactor_2D(nr, nc, nrow, ncol, l, v2d.data(), work2d,
                             coords_c, coords_r, row_vec, col_vec);
      mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kf);
    }

    for (int irow = 0; irow < nr; irow += Cstride) {
      int ir = get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < nc; jcol += Cstride) {
        int jc = get_lindex(nc, ncol, jcol);
        for (int kfib = 0; kfib < nfib; ++kfib) {
          fib_vec[kfib] =
              work[mgard_common::get_index3(ncol, nfib, ir, jc, kfib)];
        }
        mgard_gen::mass_mult_l(l, fib_vec, coords_f, nf, nfib);
        mgard_gen::restriction_l(l + 1, fib_vec, coords_f, nf, nfib);
        mgard_gen::solve_tridiag_M_l(l + 1, fib_vec, coords_f, nf, nfib);
        for (int kfib = 0; kfib < nfib; ++kfib) {
          work[mgard_common::get_index3(ncol, nfib, ir, jc, kfib)] =
              fib_vec[kfib];
        }
      }
    }

    add3_level_l(l + 1, v, work.data(), nr, nc, nf, nrow, ncol, nfib);


    // print_matrix(nfib,  nrow,   ncol,
    //                     v,  ncol, nrow);





    cudaMemcpy3DHelper(dwork2, lddwork21 * sizeof(double), nfib * sizeof(double), ncol,
                       v, nfib  * sizeof(double), nfib * sizeof(double), ncol,
                       nfib * sizeof(double), ncol, nrow,
                       H2D);

    // cudaMemcpy3DHelper(dwork2, lddwork21 * sizeof(double), nfib * sizeof(double), ncol,
    //                    work.data(), nfib  * sizeof(double), nfib * sizeof(double), ncol,
    //                    nfib * sizeof(double), ncol, nrow,
    //                    H2D);

    org_to_pow2p1(nrow,   ncol,  nfib,  
                  nr,     nc,    nf,    
                  dirow,  dicol, difib, 
                  dwork2, lddwork21,  lddwork22,
                  dcwork2, lddcwork21,  lddcwork22);


    

    // print_matrix(nrow,   ncol, nfib,  
    //                     v,  nfib, ncol);

    
    fib_stride = stride;
    row_stride = stride;
    col_stride = stride;
    ret = pow2p1_to_cpt(nr,         nc,         nf,
                        row_stride, col_stride, fib_stride,
                        dcv,        lddcv1,     lddcv2,
                        dcwork,     lddcwork1,  lddcwork2);
    pow2p1_to_cpt_time += ret.time;

    fib_stride = 1;
    row_stride = 1;
    col_stride = 1;
    ret = pi_Ql_cuda_cpt_sm(nr_l[l],      nc_l[l],      nf_l[l],
                            row_stride,   col_stride,   fib_stride,
                            dcwork,       lddcwork1,    lddcwork2,
                            ddist_r_l[l], ddist_c_l[l], ddist_f_l[l], 4);
    pi_Ql_cuda_time += ret.time;

    fib_stride = stride;
    row_stride = stride;
    col_stride = stride;
    ret = cpt_to_pow2p1(nr,         nc,         nf,
                        row_stride, col_stride, fib_stride,
                        dcwork,     lddcwork1,  lddcwork2,
                        dcv,        lddcv1,     lddcv2);
    cpt_to_pow2p1_time += ret.time;    


  //   printf("pi_Ql cv:\n");
  // print_matrix_cuda(nr,     nc,    nf,
  //                   dcv,   lddcv1, lddcv2, 
  //                   nf); 

    fib_stride = stride;
    row_stride = stride;
    col_stride = stride;
    copy_level_l_cuda_cpt(nr,         nc,         nf,
                          row_stride, col_stride, fib_stride,
                          dcv,        lddcv1,     lddcv2,
                          dwork,      lddwork1,   lddwork2);

    fib_stride = Cstride;
    row_stride = Cstride;
    col_stride = Cstride;
    ret = pow2p1_to_cpt(nr,         nc,         nf,
                        row_stride, col_stride, fib_stride,
                        dwork,      lddwork1,   lddwork2,
                        dcwork,     lddcwork1,  lddcwork2);
    pow2p1_to_cpt_time += ret.time;

    fib_stride = 1;
    row_stride = 1;
    col_stride = 1;
    assign_num_level_l_cuda_cpt(nr_l[l+1],  nc_l[l+1],  nf_l[l+1],
                                row_stride, col_stride, fib_stride,
                                dcwork,     lddcwork1,  lddcwork2,
                                0.0);
    fib_stride = Cstride;
    row_stride = Cstride;
    col_stride = Cstride;
    ret = cpt_to_pow2p1(nr,         nc,         nf,
                        row_stride, col_stride, fib_stride,
                        dcwork,     lddcwork1,  lddcwork2,
                        dwork,      lddwork1,   lddwork2);
    cpt_to_pow2p1_time += ret.time;

    // std::cout << "gpu before:\n";
    // print_matrix_cuda(nr,           nc,    nf,      
    //                   dwork,        lddwork1,     lddwork2,  
    //                   nf); 

    fib_stride = stride;
    for (int f = 0; f < nf; f += fib_stride) {
      double * slice = dwork + f;
      int ldslice = lddwork2;

      row_stride = 1 * lddwork1;
      col_stride = stride * lddwork1;
      ret = pow2p1_to_cpt(nr*lddwork1,    nc*lddwork1,
                          row_stride, col_stride,
                          slice,      ldslice,
                          dcwork,     lddcwork1);
      pow2p1_to_cpt_time += ret.time;

      // std::cout << "f before = " << f << "\n";
      // print_matrix_cuda(nr_l[l],    nc_l[l],
      //                   dcwork,     lddcwork1); 

      row_stride = 1;
      col_stride = 1;
      ret = mgard_2d::mgard_gen::mass_mult_l_row_cuda_sm(nr_l[0],    nc_l[l],
                                                        row_stride, col_stride,
                                                        dcwork,     lddcwork1,
                                                        ddist_c_l[l],
                                                        16, 16);
      mass_mult_l_row_cuda_time += ret.time;

      // // std::cout << "f after= " << f << "\n";
      // // print_matrix_cuda(nr_l[l],    nc_l[l],
      // //                   dcwork,     lddcwork1); 

      row_stride = 1;
      col_stride = 1;
      ret = mgard_2d::mgard_gen::restriction_l_row_cuda_sm(nr_l[0],     nc_l[l],
                                      row_stride,  col_stride,
                                      dcwork,      lddcwork1,
                                      ddist_c_l[l],
                                      16, 16);
      restriction_l_row_cuda_time += ret.time;

      row_stride = 1;
      col_stride = 2;//1;
      ret = mgard_2d::mgard_gen::solve_tridiag_M_l_row_cuda_sm(nr_l[0],    nc_l[l],
                                          row_stride, col_stride,
                                          dcwork,     lddcwork1,
                                          ddist_c_l[l+1],
                                          16, 16);
      solve_tridiag_M_l_row_cuda_time += ret.time;


      // std::cout << "f before = " << f << "\n";
      // print_matrix_cuda(nr_l[0],     nc_l[l],
      //                   dcwork,     lddcwork1); 


      row_stride = stride;
      col_stride = 2;
      ret = mgard_2d::mgard_gen::mass_mult_l_col_cuda_sm(nr_l[0],     nc_l[l],
                                   row_stride,  col_stride,
                                   dcwork,     lddcwork1,
                                   ddist_r_l[l],
                                   2, 2);
      mass_mult_l_col_cuda_time += ret.time;

      // std::cout << "f after= " << f << "\n";
      // print_matrix_cuda(nr_l[0],     nc_l[l],
      //                   dcwork,     lddcwork1); 

      row_stride = stride;
      col_stride = 2;
      ret = mgard_2d::mgard_gen::restriction_l_col_cuda_sm(nr_l[0],         nc_l[l],
                                      row_stride, col_stride,
                                      dcwork, lddcwork1,
                                      ddist_r_l[l],
                                      16, 16);
      restriction_l_col_cuda_time += ret.time;

      row_stride = Cstride;
      col_stride = 2;
      ret = mgard_2d::mgard_gen::solve_tridiag_M_l_col_cuda_sm(nr_l[0],     nc_l[l],
                                          row_stride,    col_stride,
                                          dcwork,        lddcwork1,
                                          ddist_r_l[l+1],
                                          16, 16);
      solve_tridiag_M_l_col_cuda_time += ret.time;

      row_stride = 1 * lddwork1;
      col_stride = stride * lddwork1;
      ret = cpt_to_pow2p1(nr*lddwork1,    nc*lddwork1,
                          row_stride, col_stride,
                          dcwork,     lddcwork1,
                          slice,      ldslice);
      cpt_to_pow2p1_time += ret.time;


    }

    // // // std::cout << "gpu after:\n";
    // // // print_matrix_cuda(nr,           nc,    nf,      
    // // //                   dwork,        lddwork1,     lddwork2,  
    // // //                   nf); 



    row_stride = Cstride;
    for (int r = 0; r < nr; r += row_stride) {
    //for (int f = 0; f < 1; f += fib_stride) {
      double * slice = dwork + r * lddwork1 * lddwork2;
      int ldslice = lddwork1;

      col_stride = Cstride;
      fib_stride = stride;
      ret = pow2p1_to_cpt(nc,         nf,
                          col_stride, fib_stride,
                          slice,      ldslice,
                          dcwork,     lddcwork1);
      pow2p1_to_cpt_time += ret.time;

      col_stride = 1;
      fib_stride = 1;
      ret = mgard_2d::mgard_gen::mass_mult_l_row_cuda_sm(nc_l[l],    nf_l[l],
                                                        col_stride, fib_stride,
                                                        dcwork,     lddcwork1,
                                                        ddist_f_l[l],
                                                        16, 16);
      mass_mult_l_row_cuda_time += ret.time;

      col_stride = 1;
      fib_stride = 1;
      ret = mgard_2d::mgard_gen::restriction_l_row_cuda_sm(nc_l[l],    nf_l[l],
                                                           col_stride, fib_stride,
                                                           dcwork,      lddcwork1,
                                                           ddist_f_l[l],
                                                           16, 16);
      restriction_l_row_cuda_time += ret.time;

      col_stride = 1;
      fib_stride = 2;
      ret = mgard_2d::mgard_gen::solve_tridiag_M_l_row_cuda_sm(nc_l[l],    nf_l[l],
                                                               col_stride, fib_stride,
                                                               dcwork,     lddcwork1,
                                                               ddist_f_l[l+1],
                                                               16, 16);
      solve_tridiag_M_l_row_cuda_time += ret.time;



      col_stride = Cstride;
      fib_stride = stride;
      ret = cpt_to_pow2p1(nc,         nf,
                          col_stride, fib_stride,
                          dcwork,     lddcwork1,
                          slice,      ldslice);
      cpt_to_pow2p1_time += ret.time;


    }


    // // std::cout << "gpu dcv-before:\n";
    // // print_matrix_cuda(nr,           nc,    nf,      
    // //                   dcv,        lddcv1,     lddcv2,  
    // //                   nf); 

    fib_stride = Cstride;
    row_stride = Cstride;
    col_stride = Cstride;
    ret = add_level_l_cuda_cpt(nr,         nc,         nf,
                               row_stride, col_stride, fib_stride,
                               dcv,        lddcv1,     lddcv2,
                               dwork,      lddwork1,   lddwork2);
    add_level_cuda_time += ret.time;

    // std::cout << "cpu:\n";
    // print_matrix_cuda(1,           1, 10,
    //                   dcwork2, lddcwork21,  lddcwork22,  
    //                   nf); 

    // std::cout << "gpu dcv:\n";
    // print_matrix_cuda(1,           1, 10,
    //                   dcv,        lddcv1,     lddcv2,  
    //                   nf); 

    // compare_matrix_cuda(nr,         nc, nf,      
    //                     dcv,        lddcv1,     lddcv2,  nf,
    //                     dcwork2, lddcwork21,  lddcwork22,  nf);



    // std::cout << "gpu dwork:\n";
    // print_matrix_cuda(1,           1, 10,
    //                   dwork,        lddwork1,     lddwork2,  
    //                   nf); 

    // compare_matrix_cuda(nr,           nc,  nf,      
    //                     dwork,        lddwork1,     lddwork2,  nf,
    //                     dcwork2, lddcwork21,  lddcwork22,  nf);

  }

  compare_matrix_cuda(nr,         nc, nf,      
                      dcv,        lddcv1,     lddcv2,  nf,
                      dcwork2, lddcwork21,  lddcwork22,  nf);

  compare_matrix_cuda(nrow,   ncol,   nfib,  
                      dv,    lddv1, lddv2, nfib,
                      dv2,    lddv21, lddv22, nfib);


  ret = pow2p1_to_org(nrow,   ncol,   nfib,  
                      nr,     nc,     nf,
                      dirow,  dicol,  difib,
                      dcwork2, lddcwork21,  lddcwork22,
                      dv2,     lddv21,  lddv22);

  ret = pow2p1_to_org(nrow,   ncol,   nfib,  
                      nr,     nc,     nf,
                      dirow,  dicol,  difib,
                      dcv,    lddcv1, lddcv2,
                      dv,     lddv1,  lddv2);

  compare_matrix_cuda(nrow,   ncol,   nfib,  
                      dv,    lddv1, lddv2, nfib,
                      dv2,    lddv21, lddv22, nfib);



  pow2p1_to_org_time += ret.time;

  return mgard_cuda_ret(0, 1.0);

}


}