#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include "mgard_nuni_2d_cuda_kernels.h"
#include <fstream>
#include <cmath>

namespace mgard_2d {
namespace mgard_common {
__device__ double 
_get_dist_o1(double * coords, int i, int j) {
  return coords[j] - coords[i];
}

}
namespace mgard_gen {

template <typename T> 
mgard_cuda_ret 
refactor_2D_cuda_compact_l1(const int l_target,
                            const int nrow,     const int ncol,
                            const int nr,       const int nc, 
                            int * dirow,        int * dicol,
                            int * dirowP,       int * dicolP,
                            T * dv,        int lddv, 
                            T * dwork,     int lddwork,
                            T * dcoords_x, T * dcoords_y,
                            int B,
                            mgard_cuda_handle & handle, bool profile) {

  T * dcv;
  size_t dcv_pitch;
  cudaMallocPitchHelper((void**)&dcv, &dcv_pitch, nc * sizeof(T), nr);
  int lddcv = dcv_pitch / sizeof(T);

  T * dcwork;
  size_t dcwork_pitch;
  cudaMallocPitchHelper((void**)&dcwork, &dcwork_pitch, nc * sizeof(T), nr);
  int lddcwork = dcwork_pitch / sizeof(T);

  int * cirow = new int[nr];
  int * cicol = new int[nc];

  for (int i = 0; i < nr; i++) {
    cirow[i] = i;
  }

  for (int i = 0; i < nc; i++) {
    cicol[i] = i;
  }



  //std::cout <<"test1\n";
  int * dcirow;
  cudaMallocHelper((void**)&dcirow, nr * sizeof(int));
  cudaMemcpyAsyncHelper(dcirow, cirow, nr * sizeof(int), H2D,
                        handle, 0, profile);

  //std::cout <<"test2\n";
  int * dcicol;
  cudaMallocHelper((void**)&dcicol, nc * sizeof(int));
  cudaMemcpyAsyncHelper(dcicol, cicol, nc * sizeof(int), H2D,
                        handle, 0, profile);

 // std::cout <<"test3\n";
  T * coords_x = new T[ncol];
  T * coords_y = new T[nrow];
  cudaMemcpyAsyncHelper(coords_x, dcoords_x, ncol * sizeof(T), D2H,
                        handle, 0, profile);
  //std::cout <<"test4\n";
  cudaMemcpyAsyncHelper(coords_y, dcoords_y, nrow * sizeof(T), D2H,
                        handle, 0, profile);


  int * irow = new int[nr];
  int * icol = new int[nc];
  //std::cout <<"test5\n";
  cudaMemcpyAsyncHelper(irow, dirow, nr * sizeof(int), D2H,
                        handle, 0, profile);
 // std::cout <<"test6\n";
  cudaMemcpyAsyncHelper(icol, dicol, nc * sizeof(int), D2H,
                        handle, 0, profile);

  T * ccoords_x = new T[nc];
  T * ccoords_y = new T[nr];

  for (int i = 0; i < nc; i++) {
    ccoords_x[i] = coords_x[icol[i]];
  }

  for (int i = 0; i < nr; i++) {
    ccoords_y[i] = coords_y[irow[i]];
  }

  T * dccoords_x;
  //std::cout <<"test7\n";
  cudaMallocHelper((void**)&dccoords_x, nc * sizeof(T));
  cudaMemcpyAsyncHelper(dccoords_x, ccoords_x, nc * sizeof(T), H2D,
                        handle, 0, profile);

  T * dccoords_y;
 // std::cout <<"test8\n"; 
  cudaMallocHelper((void**)&dccoords_y, nr * sizeof(T));
  cudaMemcpyAsyncHelper(dccoords_y, ccoords_y, nr * sizeof(T), H2D,
                        handle, 0, profile);

  


  mgard_cuda_ret ret;
  double total_time = 0.0;
  std::ofstream timing_results;
  if (profile) {
    timing_results.open (handle.csv_prefix + "refactor_2D_cuda_cpt_l1.csv");
  }


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

  // ret = org_to_pow2p1(nrow,  ncol,
  //                      nr,    nc,
  //                      dirow, dicol,
  //                      dv,    lddv,
  //                      dcv,   lddcv);
  // org_to_pow2p1_time = ret.time;


  

  // std::cout << "dv:" << std::endl;
  // print_matrix_cuda(nrow, ncol, dv,  lddv );
  
  // std::cout << "dcv:" << std::endl;
  // print_matrix_cuda(nr,   nc,   dcv, lddcv);

  ret = org_to_pow2p1(nrow,  ncol,
                       nr,    nc,
                       dirow, dicol,
                       dv,    lddv,
                       dcv,   lddcv, B,
                       handle, 0, profile);
  org_to_pow2p1_time = ret.time;



  for (int l = 0; l < l_target; ++l) {
    // std::cout << "l = " << l << std::endl;
    int stride = std::pow(2, l); // current stride
    int Cstride = stride * 2;    // coarser stride

    // -> change funcs in pi_QL to use _l functions, otherwise distances are
    // wrong!!!
    // pi_Ql(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec,
    //       col_vec); // rename!. v@l has I-\Pi_l Q_l+1 u

    int row_stride = stride;
    int col_stride = stride;
    ret = pi_Ql_cuda(nr,              nc,
                     nr,              nc,
                     row_stride,      col_stride,
                     dcirow,          dcicol,
                     dcv,             lddcv, 
                     dccoords_x,      dccoords_y, B,
                     handle, 0, profile);
    pi_Ql_cuda_time = ret.time;

    

    // pi_Ql(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec,
    //       col_vec); // rename!. v@l has I-\Pi_l Q_l+1 u

    // copy_level_l(l, v, work.data(), nr, nc, nrow, ncol);

    row_stride = stride;
    col_stride = stride;
    ret = copy_level_l_cuda(nr,              nc,
                            nr,         nc,
                            row_stride, col_stride,
                            dcirow,     dcicol,
                            dcv,        lddcv, 
                            dcwork,      lddcwork, B,
                            handle, 0, profile);
    copy_level_l_cuda_time = ret.time;

    // assign_num_level_l(l + 1, work.data(), 0.0, nr, nc, nrow, ncol);
    row_stride = Cstride;
    col_stride = Cstride;
    T val = 0.0;
    ret = assign_num_level_l_cuda(nr,         nc,
                                  nr,         nc,
                                  row_stride, col_stride,
                                  dcirow,     dcicol,
                                  dcwork,      lddcwork, 
                                  val, B,
                                  handle, 0, profile);
    assign_num_level_l_cuda_time = ret.time;

    
    row_stride = stride;//1;
    col_stride = stride;
    ret = mass_mult_l_row_cuda(nr,         nc,
                               nr,         nc,
                               row_stride, col_stride,
                               dcirow,     dcicol,
                               dcwork,     lddcwork,
                               dccoords_x, B,
                               handle, 0, profile);
    mass_mult_l_row_cuda_time = ret.time;

    row_stride = stride;//1;
    col_stride = stride;
    ret = restriction_l_row_cuda(nr,         nc,
                                 nr,         nc,
                                 row_stride, col_stride,
                                 dcirow,     dcicol,
                                 dcwork,      lddcwork,
                                 dccoords_x, B,
                                 handle, 0, profile);
    restriction_l_row_cuda_time = ret.time;

    row_stride = stride;//1;
    col_stride = Cstride;
    ret = solve_tridiag_M_l_row_cuda(nr,         nc,
                                     nr,         nc,
                                     row_stride, col_stride,
                                     dcirow,     dcicol,
                                     dcwork,     lddcwork,
                                     dccoords_x, B,
                                     handle, 0, profile);
    solve_tridiag_M_l_row_cuda_time = ret.time;


    if (nrow > 1) // do this if we have an 2-dimensional array
    {

      row_stride = stride;
      col_stride = Cstride;
      ret = mass_mult_l_col_cuda(nr,         nc,
                                 nr,         nc,
                                 row_stride, col_stride,
                                 dcirow,     dcicol,
                                 dcwork,     lddcwork,
                                 dccoords_y, B,
                                 handle, 0, profile);
      mass_mult_l_col_cuda_time = ret.time;

      row_stride = stride;
      col_stride = Cstride;
      ret = restriction_l_col_cuda(nr,         nc,
                                   nr,         nc,
                                   row_stride, col_stride,
                                   dcirow,     dcicol,
                                   dcwork, lddcwork,
                                   dccoords_y, B,
                                   handle, 0, profile);
      restriction_l_col_cuda_time = ret.time;

      row_stride = Cstride;
      col_stride = Cstride;
      ret = solve_tridiag_M_l_col_cuda(nr,         nc,
                                       nr,         nc,
                                       row_stride, col_stride,
                                       dcirow,     dcicol,
                                       dcwork, lddcwork,
                                       dccoords_y, B,
                                       handle, 0, profile);
      solve_tridiag_M_l_col_cuda_time = ret.time;

    }

    // // Solved for (z_l, phi_l) = (c_{l+1}, vl)
    // // add_level_l(l + 1, v, work.data(), nr, nc, nrow, ncol);
    ret = add_level_l_cuda(nr,       nc,
                           nr,         nc,
                           row_stride, col_stride,
                           dcirow,     dcicol,
                           dcv,        lddcv, 
                           dcwork,      lddcwork, B,
                           handle, 0, profile);
    add_level_cuda_time = ret.time;


    if (profile) {
      timing_results << l << ",pow2p1_to_cpt_time," << pow2p1_to_cpt_time << std::endl;
      timing_results << l << ",cpt_to_pow2p1_time," << cpt_to_pow2p1_time << std::endl;

      timing_results << l << ",pi_Ql_cuda_time," << pi_Ql_cuda_time << std::endl;
      timing_results << l << ",copy_level_l_cuda_time," << copy_level_l_cuda_time << std::endl;
      timing_results << l << ",assign_num_level_l_cuda_time," << assign_num_level_l_cuda_time << std::endl;

      timing_results << l << ",mass_mult_l_row_cuda_time," << mass_mult_l_row_cuda_time << std::endl;
      timing_results << l << ",restriction_l_row_cuda_time," << restriction_l_row_cuda_time << std::endl;
      timing_results << l << ",solve_tridiag_M_l_row_cuda_time," << solve_tridiag_M_l_row_cuda_time << std::endl;

      timing_results << l << ",mass_mult_l_col_cuda_time," << mass_mult_l_col_cuda_time << std::endl;
      timing_results << l << ",restriction_l_col_cuda_time," << restriction_l_col_cuda_time << std::endl;
      timing_results << l << ",solve_tridiag_M_l_col_cuda_time," << solve_tridiag_M_l_col_cuda_time << std::endl;
      timing_results << l << ",add_level_cuda_time," << add_level_cuda_time << std::endl;
  
      total_time += pow2p1_to_cpt_time;
      total_time += cpt_to_pow2p1_time;

      total_time += pi_Ql_cuda_time;
      total_time += copy_level_l_cuda_time;
      total_time += assign_num_level_l_cuda_time;

      total_time += mass_mult_l_row_cuda_time;
      total_time += restriction_l_row_cuda_time;
      total_time += solve_tridiag_M_l_row_cuda_time;

      total_time += mass_mult_l_col_cuda_time;
      total_time += solve_tridiag_M_l_col_cuda_time;
      total_time += solve_tridiag_M_l_col_cuda_time;

      total_time += add_level_cuda_time;
    }


  } //out of loop

  ret = pow2p1_to_org(nrow,  ncol,
                      nr,    nc,
                      dirow, dicol,
                      dcv,   lddcv,
                      dv,    lddv, B,
                      handle, 0, profile);
  pow2p1_to_org_time = ret.time;

  if (profile) {
    timing_results << 0 << ",org_to_pow2p1_time," << org_to_pow2p1_time << std::endl;
    timing_results << 0 << ",pow2p1_to_org_time," << pow2p1_to_org_time << std::endl;
    timing_results.close();

    total_time += org_to_pow2p1_time;
    total_time += pow2p1_to_org_time;
  }

  cudaFreeHelper(dcv);
  cudaFreeHelper(dcwork);
  cudaFreeHelper(dcirow);
  cudaFreeHelper(dcicol);
  cudaFreeHelper(dccoords_x);
  cudaFreeHelper(dccoords_y);
  
  return mgard_cuda_ret(0, total_time);
}

template mgard_cuda_ret 
refactor_2D_cuda_compact_l1<double>(const int l_target,
                            const int nrow,     const int ncol,
                            const int nr,       const int nc, 
                            int * dirow,        int * dicol,
                            int * dirowP,       int * dicolP,
                            double * dv,        int lddv, 
                            double * dwork,     int lddwork,
                            double * dcoords_x, double * dcoords_y,
                            int B, 
                            mgard_cuda_handle & handle, bool profile);
template mgard_cuda_ret 
refactor_2D_cuda_compact_l1<float>(const int l_target,
                            const int nrow,     const int ncol,
                            const int nr,       const int nc, 
                            int * dirow,        int * dicol,
                            int * dirowP,       int * dicolP,
                            float * dv,        int lddv, 
                            float * dwork,     int lddwork,
                            float * dcoords_x, float * dcoords_y,
                            int B, 
                            mgard_cuda_handle & handle, bool profile);


}
}

