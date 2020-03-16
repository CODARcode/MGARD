#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"
#include "mgard_nuni_2d_cuda_kernels.h"

#include <fstream>

namespace mgard_2d {
namespace mgard_gen {

template <typename T> 
mgard_cuda_ret 
prep_2D_cuda(const int nrow,     const int ncol,
             const int nr,       const int nc, 
             int * dirow,        int * dicol,
             int * dirowP,       int * dicolP,
             T * dv,        int lddv, 
             T * dwork,     int lddwork,
             T * dcoords_x, T * dcoords_y,
             int B,
             mgard_cuda_handle & handle, bool profile) {

  mgard_cuda_ret ret;
  double total_time = 0.0;
  std::ofstream timing_results;

  if (profile) {
    timing_results.open ("prep_2D_cuda.csv");
  }

  double org_to_pow2p1_time = 0.0;
  double pow2p1_to_org_time = 0.0;

  double pow2p1_to_cpt_time = 0.0;
  double cpt_to_pow2p1_time = 0.0;

  double pi_Ql_first_cuda_time = 0.0;
  double copy_level_cuda_time = 0.0;
  double assign_num_level_l_cuda_time = 0.0;

  double mass_matrix_multiply_row_cuda_time = 0.0;
  double restriction_first_row_cuda_time = 0.0;
  double solve_tridiag_M_l_row_cuda_time = 0.0;
  
  double mass_matrix_multiply_col_cuda_time = 0.0;
  double restriction_first_col_cuda_time = 0.0;
  double solve_tridiag_M_l_col_cuda_time = 0.0;
  double add_level_l_cuda_time = 0.0;

  int l = 0;
  int row_stride = 1;
  int col_stride = 1;

  ret = pi_Ql_first_cuda(nrow,      ncol,
                         nr,        nc, 
                         dirow,     dicol,
                         dirowP,    dicolP,
                         dcoords_x, dcoords_y,
                         dv,        lddv, B,
                         handle, 0, profile); //(I-\Pi)u this is the initial move to 2^k+1 nodes
  pi_Ql_first_cuda_time = ret.time;

  ret = mgard_cannon::copy_level_cuda(nrow,       ncol, 
                                row_stride, col_stride,
                                dv,         lddv,
                                dwork,      lddwork,
                                B,
                                handle, 0, profile);
  copy_level_cuda_time = ret.time;

  T val = 0.0;
  ret = assign_num_level_l_cuda(nrow,       ncol,
                          nr,         nc,
                          row_stride, col_stride,
                          dirow,      dicol,
                          dwork,      lddwork, 
                          val, B,
                          handle, 0, profile);
  assign_num_level_l_cuda_time = ret.time;

  row_stride = 1;
  col_stride = 1;
  ret = mgard_cannon::mass_matrix_multiply_row_cuda(nrow,       ncol,
                                              row_stride, col_stride,
                                              dwork,      lddwork,
                                              dcoords_x,  B,
                                              handle, 0, profile);
  mass_matrix_multiply_row_cuda_time = ret.time;

  ret = restriction_first_row_cuda(nrow,       ncol,
                             row_stride, dicolP, nc,
                             dwork,      lddwork,
                             dcoords_x, B,
                             handle, 0, profile);
  restriction_first_row_cuda_time = ret.time;

 

  //   //   //std::cout  << "recomposing-colsweep" << "\n";

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) // check if we have 1-D array..
  {
    row_stride = 1;
    col_stride = 1;
    ret = mgard_cannon::mass_matrix_multiply_col_cuda(nrow,       ncol,
                                                row_stride, col_stride,
                                                dwork,      lddwork,
                                                dcoords_y, B,
                                                handle, 0, profile);
    mass_matrix_multiply_col_cuda_time = ret.time;


    ret = restriction_first_col_cuda(nrow,   ncol,
                               dirowP, nr, col_stride,
                               dwork,  lddwork,
                               dcoords_y, B,
                               handle, 0, profile);
    restriction_first_col_cuda_time = ret.time;

    
 }
  // add_level_l(0, v, work.data(), nr, nc, nrow, ncol);
  row_stride = 1;
  col_stride = 1;
  ret = add_level_l_cuda(nrow,       ncol, 
                   nr,         nc, 
                   row_stride, col_stride, 
                   dirow,      dicol, 
                   dv,         lddv, 
                   dwork,      lddwork, B,
                   handle, 0, profile);
  add_level_l_cuda_time = ret.time;

  if (profile) {
    timing_results << l << ",org_to_pow2p1_time," << org_to_pow2p1_time << std::endl;
    timing_results << l << ",pow2p1_to_org_time," << pow2p1_to_org_time << std::endl;

    timing_results << l << ",pow2p1_to_cpt_time," << pow2p1_to_cpt_time << std::endl;
    timing_results << l << ",cpt_to_pow2p1_time," << cpt_to_pow2p1_time << std::endl;

    timing_results << l << ",pi_Ql_first_cuda_time," << pi_Ql_first_cuda_time << std::endl;
    timing_results << l << ",copy_level_cuda_time," << copy_level_cuda_time << std::endl;
    timing_results << l << ",assign_num_level_l_cuda_time," << assign_num_level_l_cuda_time << std::endl;

    timing_results << l << ",mass_matrix_multiply_row_cuda_time," << mass_matrix_multiply_row_cuda_time << std::endl;
    timing_results << l << ",restriction_first_row_cuda_time," << restriction_first_row_cuda_time << std::endl;
    timing_results << l << ",solve_tridiag_M_l_row_cuda_time," << solve_tridiag_M_l_row_cuda_time << std::endl;
    
    timing_results << l << ",mass_matrix_multiply_col_cuda_time," << mass_matrix_multiply_col_cuda_time << std::endl;
    timing_results << l << ",restriction_first_col_cuda_time," << restriction_first_col_cuda_time << std::endl;
    timing_results << l << ",solve_tridiag_M_l_col_cuda_time," << solve_tridiag_M_l_col_cuda_time << std::endl;
    timing_results << l << ",add_level_l_cuda_time," << add_level_l_cuda_time << std::endl;
    timing_results.close();
    total_time += org_to_pow2p1_time;
    total_time += pow2p1_to_org_time;

    total_time += pow2p1_to_cpt_time;
    total_time += cpt_to_pow2p1_time;

    total_time += pi_Ql_first_cuda_time;
    total_time += copy_level_cuda_time;
    total_time += assign_num_level_l_cuda_time;

    total_time += mass_matrix_multiply_row_cuda_time;
    total_time += restriction_first_row_cuda_time;
    total_time += solve_tridiag_M_l_row_cuda_time;

    total_time += mass_matrix_multiply_col_cuda_time;
    total_time += restriction_first_col_cuda_time;
    total_time += solve_tridiag_M_l_col_cuda_time;

    total_time += add_level_l_cuda_time;
  }
  return mgard_cuda_ret(0, total_time);
}


template <typename T>
mgard_cuda_ret 
refactor_2D_cuda(const int l_target,
                 const int nrow,     const int ncol,
                 const int nr,       const int nc, 
                 int * dirow,        int * dicol,
                 int * dirowP,       int * dicolP,
                 T * dv,        int lddv, 
                 T * dwork,     int lddwork,
                 T * dcoords_x, T * dcoords_y,
                 int B, 
                 mgard_cuda_handle & handle, bool profile) {
  // refactor
  //    //std::cout  << "I am the general refactorer!" <<"\n";
  
  mgard_cuda_ret ret;
  double total_time = 0.0;
  std::ofstream timing_results;

  if (profile) {
    timing_results.open ("refactor_2D_cuda.csv");
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

  for (int l = 0; l < l_target; ++l) {
    int stride = std::pow(2, l); // current stride
    int Cstride = stride * 2;    // coarser stride

    // -> change funcs in pi_QL to use _l functions, otherwise distances are
    // wrong!!!
    int row_stride = stride;
    int col_stride = stride;
    ret = pi_Ql_cuda(nrow,            ncol,
                     nr,              nc,
                     row_stride,      col_stride,
                     dirow,           dicol,
                     dv,              lddv, 
                     dcoords_x,       dcoords_y, B, 
                     handle, 0, profile);
    pi_Ql_cuda_time = ret.time;

    row_stride = stride;
    col_stride = stride;
    ret = copy_level_l_cuda(nrow,       ncol,
                      nr,         nc,
                      row_stride, col_stride,
                      dirow,      dicol,
                      dv,         lddv, 
                      dwork,      lddwork, B, 
                      handle, 0, profile);
    copy_level_l_cuda_time = ret.time;

    row_stride = Cstride;
    col_stride = Cstride;
    T val = 0.0;
    ret = assign_num_level_l_cuda(nrow,       ncol,
                            nr,         nc,
                            row_stride, col_stride,
                            dirow,      dicol,
                            dwork,      lddwork, 
                            val, B, 
                            handle, 0, profile);
    assign_num_level_l_cuda_time = ret.time;

    row_stride = 1;
    col_stride = stride;
    ret = mass_mult_l_row_cuda(nrow,       ncol,
                         nr,         nc,
                         row_stride, col_stride,
                         dirow,      dicol,
                         dwork,      lddwork,
                         dcoords_x, B, 
                         handle, 0, profile);
    mass_mult_l_row_cuda_time = ret.time;


    row_stride = 1;
    col_stride = stride;
    ret = restriction_l_row_cuda(nrow,       ncol,
                           nr,         nc,
                           row_stride, col_stride,
                           dirow,      dicol,
                           dwork,      lddwork,
                           dcoords_x, B, 
                           handle, 0, profile);
    restriction_l_row_cuda_time = ret.time;

    row_stride = 1;
    col_stride = Cstride;
    ret = solve_tridiag_M_l_row_cuda(nrow,       ncol,
                               nr,         nc,
                               row_stride, col_stride,
                               dirow,      dicol,
                               dwork,      lddwork,
                               dcoords_x, B, 
                               handle, 0, profile);
    solve_tridiag_M_l_row_cuda_time = ret.time;

    

    // column-sweep
    if (nrow > 1) // do this if we have an 2-dimensional array
    {
      // print_matrix(nrow, ncol, work.data(), ldwork);
      row_stride = stride;
      col_stride = Cstride;
      ret = mass_mult_l_col_cuda(nrow,       ncol,
                                 nr,         nc,
                                 row_stride, col_stride,
                                 dirow,      dicol,
                                 dwork,      lddwork,
                                 dcoords_y, B, 
                                 handle, 0, profile);
      mass_mult_l_col_cuda_time = ret.time;


      row_stride = stride;
      col_stride = Cstride;
      ret = restriction_l_col_cuda(nrow,       ncol,
                                   nr,         nc,
                                   row_stride, col_stride,
                                   dirow,      dicol,
                                   dwork, lddwork,
                                   dcoords_y, B, 
                                   handle, 0, profile);
      restriction_l_col_cuda_time = ret.time;

      row_stride = Cstride;
      col_stride = Cstride;
      ret = solve_tridiag_M_l_col_cuda(nrow,       ncol,
                                       nr,         nc,
                                       row_stride, col_stride,
                                       dirow,       dicol,
                                       dwork, lddwork,
                                       dcoords_y, B, 
                                       handle, 0, profile);
      solve_tridiag_M_l_col_cuda_time = ret.time;
    }

    // Solved for (z_l, phi_l) = (c_{l+1}, vl)
    row_stride = Cstride;
    col_stride = Cstride;
    ret = add_level_l_cuda(nrow,       ncol, 
                     nr,         nc,
                     row_stride, col_stride,
                     dirow,      dicol,
                     dv,         lddv, 
                     dwork,      lddwork, B, 
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
  }// end of loop
  if (profile) {
    timing_results << 0 << ",org_to_pow2p1_time," << org_to_pow2p1_time << std::endl;
    timing_results << 0 << ",pow2p1_to_org_time," << pow2p1_to_org_time << std::endl;
    total_time += org_to_pow2p1_time;
    total_time += pow2p1_to_org_time;
    timing_results.close();
  }

  return mgard_cuda_ret(0, total_time);
}

template <typename T> 
mgard_cuda_ret 
recompose_2D_cuda(const int l_target,
                  const int nrow,     const int ncol,
                  const int nr,       const int nc, 
                  int * dirow,        int * dicol,
                  int * dirowP,       int * dicolP,
                  T * dv,        int lddv, 
                  T * dwork,     int lddwork,
                  T * dcoords_x, T * dcoords_y,
                  int B,
                  mgard_cuda_handle & handle, bool profile) {
 
  mgard_cuda_ret ret;
  double total_time = 0.0;
  std::ofstream timing_results;
  if (profile) {
    timing_results.open ("recompose_2D_cuda.csv");
  }

  double org_to_pow2p1_time = 0.0;
  double pow2p1_to_org_time = 0.0;

  double pow2p1_to_cpt_time = 0.0;
  double cpt_to_pow2p1_time = 0.0;

  double copy_level_l_cuda_time = 0.0;
  double assign_num_level_l_cuda_time = 0.0;
  double assign_num_level_l_cuda_time2 = 0.0;

  double mass_mult_l_row_cuda_time = 0.0;
  double restriction_l_row_cuda_time = 0.0;
  double solve_tridiag_M_l_row_cuda_time = 0.0;

  double mass_mult_l_col_cuda_time = 0.0;
  double restriction_l_col_cuda_time = 0.0;
  double solve_tridiag_M_l_col_cuda_time = 0.0;

  double subtract_level_l_cuda_time = 0.0;
  double subtract_level_l_cuda_time2 = 0.0;
  double prolongate_l_row_cuda_time = 0.0;
  double prolongate_l_col_cuda_time = 0.0;

  for (int l = l_target; l > 0; --l) {

    int stride = std::pow(2, l); // current stride
    int Pstride = stride / 2;
    int Cstride = stride * 2;

    int row_stride = Pstride;
    int col_stride = Pstride;
    ret = copy_level_l_cuda(nrow,       ncol,
                      nr,         nc,
                      row_stride, col_stride,
                      dirow,      dicol,
                      dv,         lddv, 
                      dwork,      lddwork, B,
                      handle, 0, profile);
    copy_level_l_cuda_time = ret.time;

    // assign_num_level_l(l, work.data(), 0.0, nr, nc, nrow, ncol);
    row_stride = stride;
    col_stride = stride;
    T val = 0.0;
    ret = assign_num_level_l_cuda(nrow,       ncol,
                            nr,         nc,
                            row_stride, col_stride,
                            dirow,      dicol,
                            dwork,      lddwork, 
                            val, B,
                            handle, 0, profile);
    assign_num_level_l_cuda_time = ret.time;

    //        //std::cout  << "recomposing-rowsweep" << "\n";
    //  l = 0;
    // row-sweep
    row_stride = 1;
    col_stride = Pstride;
    ret = mass_mult_l_row_cuda(nrow,       ncol,
                         nr,         nc,
                         row_stride, col_stride,
                         dirow,      dicol,
                         dwork,      lddwork,
                         dcoords_x, B,
                         handle, 0, profile);
    mass_mult_l_row_cuda_time = ret.time;


    row_stride = 1;
    col_stride = Pstride;
    ret = restriction_l_row_cuda(nrow,       ncol,
                           nr,         nc,
                           row_stride, col_stride,
                           dirow,      dicol,
                           dwork,      lddwork,
                           dcoords_x, B,
                           handle, 0, profile);
    restriction_l_row_cuda_time = ret.time;


    row_stride = 1;
    col_stride = stride;
    ret = solve_tridiag_M_l_row_cuda(nrow,       ncol,
                               nr,         nc,
                               row_stride, col_stride,
                               dirow,      dicol,
                               dwork,      lddwork,
                               dcoords_x, B,
                               handle, 0, profile);
    solve_tridiag_M_l_row_cuda_time = ret.time;

    
    //   //std::cout  << "recomposing-colsweep" << "\n";

    // column-sweep, this is the slow one! Need something like column_copy
    if (nrow > 1) // check if we have 1-D array..
    {
      row_stride = Pstride;
      col_stride = stride;
      ret = mass_mult_l_col_cuda(nrow,       ncol,
                           nr,         nc,
                           row_stride, col_stride,
                           dirow,      dicol,
                           dwork,      lddwork,
                           dcoords_y, B,
                           handle, 0, profile);
      mass_mult_l_col_cuda_time = ret.time;


      row_stride = Pstride;
      col_stride = stride;
      ret = restriction_l_col_cuda(nrow,       ncol,
                                   nr,         nc,
                                   row_stride, col_stride,
                                   dirow,       dicol,
                                   dwork, lddwork,
                                   dcoords_y, B,
                                   handle, 0, profile);
      restriction_l_col_cuda_time = ret.time;

      row_stride = stride;
      col_stride = stride;
      ret = solve_tridiag_M_l_col_cuda(nrow,       ncol,
                                       nr,         nc,
                                       row_stride, col_stride,
                                       dirow,      dicol,
                                       dwork,      lddwork,
                                       dcoords_y, B,
                                       handle, 0, profile);
      solve_tridiag_M_l_col_cuda_time = ret.time;
    }

    // subtract_level_l(l, work.data(), v, nr, nc, nrow, ncol); // do -(Qu - zl)
    row_stride = stride;
    col_stride = stride;
    ret = subtract_level_l_cuda(nrow,       ncol, 
                                nr,         nc,
                                row_stride, col_stride,
                                dirow,      dicol,
                                dwork,      lddwork,
                                dv,         lddv, B,
                                handle, 0, profile);
    subtract_level_l_cuda_time = ret.time;

    //        //std::cout  << "recomposing-rowsweep2" << "\n";

    //   //int Pstride = stride/2; //finer stride

    //   // row-sweep

    row_stride = stride;
    col_stride = stride;
    ret = prolongate_l_row_cuda(nrow,       ncol, 
                                nr,         nc,
                                row_stride, col_stride,
                                dirow,      dicol,
                                dwork,      lddwork,
                                dcoords_x, B,
                                handle, 0, profile);
    prolongate_l_row_cuda_time = ret.time;


    //   //std::cout  << "recomposing-colsweep2" << "\n";
    // column-sweep, this is the slow one! Need something like column_copy
    if (nrow > 1) {
      row_stride = stride;
      col_stride = Pstride;
      ret = prolongate_l_col_cuda(nrow,        ncol, 
                             nr,         nc,
                             row_stride, col_stride,
                             dirow,      dicol,
                             dwork,      lddwork,
                             dcoords_y, B,
                             handle, 0, profile);
      prolongate_l_col_cuda_time = ret.time;
    }

    row_stride = stride;
    col_stride = stride;
    ret = assign_num_level_l_cuda(nrow,       ncol,
                            nr,         nc,
                            row_stride, col_stride,
                            dirow,      dicol,
                            dv,         lddv, 
                            val, B,
                            handle, 0, profile);
    assign_num_level_l_cuda_time2 = ret.time;

    row_stride = Pstride;
    col_stride = Pstride;
    ret = subtract_level_l_cuda(nrow,       ncol, 
                          nr,         nc,
                          row_stride, col_stride,
                          dirow,      dicol,
                          dv,         lddv,
                          dwork,      lddwork, B,
                          handle, 0, profile);
    subtract_level_l_cuda_time2 = ret.time;

    if (profile) {
      

      timing_results << l << ",pow2p1_to_cpt_time," << pow2p1_to_cpt_time << std::endl;
      timing_results << l << ",cpt_to_pow2p1_time," << cpt_to_pow2p1_time << std::endl;

      timing_results << l << ",copy_level_l_cuda_time," << copy_level_l_cuda_time << std::endl;
      timing_results << l << ",assign_num_level_l_cuda_time," << assign_num_level_l_cuda_time << std::endl;
      timing_results << l << ",assign_num_level_l_cuda_time2," << assign_num_level_l_cuda_time2 << std::endl;

      timing_results << l << ",mass_mult_l_row_cuda_time," << mass_mult_l_row_cuda_time << std::endl;
      timing_results << l << ",restriction_l_row_cuda_time," << restriction_l_row_cuda_time << std::endl;
      timing_results << l << ",solve_tridiag_M_l_row_cuda_time," << solve_tridiag_M_l_row_cuda_time << std::endl;

      timing_results << l << ",mass_mult_l_col_cuda_time," << mass_mult_l_col_cuda_time << std::endl;
      timing_results << l << ",restriction_l_col_cuda_time," << restriction_l_col_cuda_time << std::endl;
      timing_results << l << ",solve_tridiag_M_l_col_cuda_time," << solve_tridiag_M_l_col_cuda_time << std::endl;

      timing_results << l << ",subtract_level_l_cuda_time," << subtract_level_l_cuda_time << std::endl;
      timing_results << l << ",subtract_level_l_cuda_time2," << subtract_level_l_cuda_time2 << std::endl;
      timing_results << l << ",prolongate_l_row_cuda_time," << prolongate_l_row_cuda_time << std::endl;
      timing_results << l << ",prolongate_l_col_cuda_time," << prolongate_l_col_cuda_time << std::endl;
      
      

      total_time += pow2p1_to_cpt_time;
      total_time += cpt_to_pow2p1_time;

      total_time += copy_level_l_cuda_time;
      total_time += assign_num_level_l_cuda_time;
      total_time += assign_num_level_l_cuda_time2;

      total_time += mass_mult_l_row_cuda_time;
      total_time += restriction_l_row_cuda_time;
      total_time += solve_tridiag_M_l_row_cuda_time;

      total_time += mass_mult_l_col_cuda_time;
      total_time += restriction_l_col_cuda_time;
      total_time += solve_tridiag_M_l_col_cuda_time;

      total_time += subtract_level_l_cuda_time;
      total_time += subtract_level_l_cuda_time2;
      total_time += prolongate_l_row_cuda_time;
      total_time += prolongate_l_col_cuda_time;
    }
  }
  if (profile) {
    timing_results << 0 << ",org_to_pow2p1_time," << org_to_pow2p1_time << std::endl;
    timing_results << 0 << ",pow2p1_to_org_time," << pow2p1_to_org_time << std::endl;
    total_time += org_to_pow2p1_time;
    total_time += pow2p1_to_org_time;
    timing_results.close();
  }
  return mgard_cuda_ret(0, total_time);
}


template <typename T>
mgard_cuda_ret 
postp_2D_cuda(const int nrow,     const int ncol,
              const int nr,       const int nc, 
              int * dirow,        int * dicol,
              int * dirowP,       int * dicolP,
              T * dv,        int lddv, 
              T * dwork,     int lddwork,
              T * dcoords_x, T * dcoords_y,
              int B,
              mgard_cuda_handle & handle, bool profile) {


  mgard_cuda_ret ret;
  double total_time;
  std::ofstream timing_results;
  if (profile) {
    timing_results.open ("postp_2D_cuda.csv");
  }

  double org_to_pow2p1_time = 0.0;
  double pow2p1_to_org_time = 0.0;

  double pow2p1_to_cpt_time = 0.0;
  double cpt_to_pow2p1_time = 0.0;

  double copy_level_cuda_time = 0.0;
  double assign_num_level_l_cuda_time = 0.0;
  double assign_num_level_l_cuda_time2 = 0.0;

  double mass_matrix_multiply_row_cuda_time = 0.0;
  double restriction_first_row_cuda_time = 0.0;
  double solve_tridiag_M_l_row_cuda_time = 0.0;

  double mass_matrix_multiply_col_cuda_time = 0.0;
  double restriction_first_col_cuda_time = 0.0;
  double solve_tridiag_M_l_col_cuda_time = 0.0;

  double subtract_level_l_cuda_time = 0.0;
  double prolongate_last_row_cuda_time = 0.0;
  double prolongate_last_col_cuda_time = 0.0;

  double subtract_level_cuda_time = 0.0;

  int l = 0;
  int row_stride = 1;
  int col_stride = 1;
  ret = mgard_cannon::copy_level_cuda(nrow,       ncol, 
                                row_stride, col_stride,
                                dv,         lddv,
                                dwork,      lddwork, B,
                                handle, 0, profile);
  copy_level_cuda_time = ret.time;

  row_stride = 1;
  col_stride = 1;
  T val = 0.0;
  ret = assign_num_level_l_cuda(nrow,       ncol,
                          nr,         nc,
                          row_stride, col_stride,
                          dirow,      dicol,
                          dwork,      lddwork,
                          val, B,
                          handle, 0, profile);
  assign_num_level_l_cuda_time = ret.time;

  row_stride = 1;
  col_stride = 1;
  ret = mgard_cannon::mass_matrix_multiply_row_cuda(nrow,       ncol, 
                                              row_stride, col_stride,
                                              dwork,      lddwork,
                                              dcoords_x, B,
                                              handle, 0, profile);
  mass_matrix_multiply_row_cuda_time = ret.time;


  row_stride = 1;
  col_stride = 1;
  ret = restriction_first_row_cuda(nrow,       ncol, 
                             row_stride, dicolP, nc,
                             dwork,      lddwork,
                             dcoords_x, B,
                             handle, 0, profile);
  restriction_first_row_cuda_time = ret.time;

  row_stride = 1;
  col_stride = 1;
  ret = solve_tridiag_M_l_row_cuda(nrow,       ncol,
                             nr,         nc,
                             row_stride, col_stride,
                             dirow,      dicol,
                             dwork,      lddwork,
                             dcoords_x, B,
                             handle, 0, profile);
  solve_tridiag_M_l_row_cuda_time = ret.time;


  //   //   //std::cout  << "recomposing-colsweep" << "\n";

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) // check if we have 1-D array..
  {
    
    row_stride = 1;
    col_stride = 1;
    ret = mgard_cannon::mass_matrix_multiply_col_cuda(nrow,      ncol,
                                               row_stride, col_stride,
                                               dwork,      lddwork,
                                               dcoords_y, B,
                                               handle, 0, profile);
    mass_matrix_multiply_col_cuda_time = ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = restriction_first_col_cuda(nrow,   ncol, 
                               dirowP, nr,   col_stride,
                               dwork,  lddwork,
                               dcoords_y, B,
                               handle, 0, profile);
    restriction_first_col_cuda_time = ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = solve_tridiag_M_l_col_cuda(nrow,       ncol,
                               nr,         nc,
                               row_stride, col_stride,
                               dirow,      dicol,
                               dwork,      lddwork,
                               dcoords_y, B,
                               handle, 0, profile);
    solve_tridiag_M_l_col_cuda_time = ret.time;
  }

  // subtract_level_l(0, work.data(), v, nr, nc, nrow, ncol); // do -(Qu - zl)
  row_stride = 1;
  col_stride = 1;
  ret = subtract_level_l_cuda(nrow,       ncol, 
                        nr,         nc,
                        row_stride, col_stride,
                        dirow,      dicol,
                        dwork,      lddwork,
                        dv,         lddv, B,
                        handle, 0, profile);
  subtract_level_l_cuda_time = ret.time;


  //        //std::cout  << "recomposing-rowsweep2" << "\n";

  //     //   //int Pstride = stride/2; //finer stride
  
  row_stride = 1;
  col_stride = 1;
  ret = prolongate_last_row_cuda(nrow,       ncol, 
                           nr,         nc,
                           row_stride, col_stride,
                           dirow,      dicolP,
                           dwork,      lddwork,
                           dcoords_x, B,
                           handle, 0, profile);
  prolongate_last_row_cuda_time = ret.time;

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) {
    row_stride = 1;
    col_stride = 1;
    ret = prolongate_last_col_cuda(nrow,       ncol, 
                             nr,         nc,
                             row_stride, col_stride,
                             dirowP,     dicol,
                             dwork,      lddwork,
                             dcoords_y, B,
                             handle, 0, profile);
    prolongate_last_col_cuda_time = ret.time;

  }

  ret = assign_num_level_l_cuda(nrow,       ncol,
                          nr,         nc,
                          row_stride, col_stride,
                          dirow,      dicol,
                          dv,         lddv,
                          val, B,
                          handle, 0, profile);
  assign_num_level_l_cuda_time2 = ret.time;

  // mgard_cannon::subtract_level(nrow, ncol, 0, v, work.data());
  row_stride = 1;
  col_stride = 1;
  ret = mgard_cannon::subtract_level_cuda(nrow,       ncol, 
                                    row_stride, col_stride,
                                    dv,         lddv, 
                                    dwork,      lddwork, B,
                                    handle, 0, profile); 
  subtract_level_cuda_time = ret.time;

  if (profile) {
    timing_results << l << ",org_to_pow2p1_time," << org_to_pow2p1_time << std::endl;
    timing_results << "pow2p1_to_org_time," << pow2p1_to_org_time << std::endl;

    timing_results << l << ",pow2p1_to_cpt_time," << pow2p1_to_cpt_time << std::endl;
    timing_results << l << ",cpt_to_pow2p1_time," << cpt_to_pow2p1_time << std::endl;

    timing_results << l << ",copy_level_cuda_time," << copy_level_cuda_time << std::endl;
    timing_results << l << ",assign_num_level_l_cuda_time," << assign_num_level_l_cuda_time << std::endl;
    timing_results << l << ",assign_num_level_l_cuda_time2," << assign_num_level_l_cuda_time2 << std::endl;

    timing_results << l << ",mass_matrix_multiply_row_cuda_time," << mass_matrix_multiply_row_cuda_time << std::endl;
    timing_results << l << ",restriction_first_row_cuda_time," << restriction_first_row_cuda_time << std::endl;
    timing_results << l << ",solve_tridiag_M_l_row_cuda_time," << solve_tridiag_M_l_row_cuda_time << std::endl;

    timing_results << l << ",mass_matrix_multiply_col_cuda_time," << mass_matrix_multiply_col_cuda_time << std::endl;
    timing_results << l << ",restriction_first_col_cuda_time," << restriction_first_col_cuda_time << std::endl;
    timing_results << l << ",solve_tridiag_M_l_col_cuda_time," << solve_tridiag_M_l_col_cuda_time << std::endl;

    timing_results << l << ",subtract_level_l_cuda_time," << subtract_level_l_cuda_time << std::endl;
    timing_results << l << ",prolongate_last_row_cuda_time," << prolongate_last_row_cuda_time << std::endl;
    timing_results << l << ",prolongate_last_col_cuda_time," << prolongate_last_col_cuda_time << std::endl;

    timing_results << l << ",subtract_level_cuda_time," << subtract_level_cuda_time << std::endl;
    timing_results.close();

    total_time += org_to_pow2p1_time;
    total_time += pow2p1_to_org_time;

    total_time += pow2p1_to_cpt_time;
    total_time += cpt_to_pow2p1_time;

    total_time += copy_level_cuda_time;
    total_time += assign_num_level_l_cuda_time;
    total_time += assign_num_level_l_cuda_time2;

    total_time += mass_matrix_multiply_row_cuda_time;
    total_time += restriction_first_row_cuda_time;
    total_time += solve_tridiag_M_l_row_cuda_time;

    total_time += mass_matrix_multiply_col_cuda_time;
    total_time += restriction_first_col_cuda_time;
    total_time += solve_tridiag_M_l_col_cuda_time;

    total_time += subtract_level_l_cuda_time;
    total_time += prolongate_last_row_cuda_time;
    total_time += prolongate_last_col_cuda_time;

    total_time += subtract_level_cuda_time;
  }
  return mgard_cuda_ret(0, total_time);
}


template mgard_cuda_ret 
prep_2D_cuda<double>(const int nrow,     const int ncol,
             const int nr,       const int nc, 
             int * dirow,        int * dicol,
             int * dirowP,       int * dicolP,
             double * dv,        int lddv, 
             double * dwork,     int lddwork,
             double * dcoords_x, double * dcoords_y,
             int B,
             mgard_cuda_handle & handle, bool profile);
template mgard_cuda_ret 
prep_2D_cuda<float>(const int nrow,     const int ncol,
             const int nr,       const int nc, 
             int * dirow,        int * dicol,
             int * dirowP,       int * dicolP,
             float * dv,        int lddv, 
             float * dwork,     int lddwork,
             float * dcoords_x, float * dcoords_y,
             int B,
             mgard_cuda_handle & handle, bool profile);

template mgard_cuda_ret 
refactor_2D_cuda<double>(const int l_target,
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
refactor_2D_cuda<float>(const int l_target,
                 const int nrow,     const int ncol,
                 const int nr,       const int nc, 
                 int * dirow,        int * dicol,
                 int * dirowP,       int * dicolP,
                 float * dv,        int lddv, 
                 float * dwork,     int lddwork,
                 float * dcoords_x, float * dcoords_y,
                 int B,
                 mgard_cuda_handle & handle, bool profile);

template mgard_cuda_ret 
recompose_2D_cuda<double>(const int l_target,
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
recompose_2D_cuda<float>(const int l_target,
                  const int nrow,     const int ncol,
                  const int nr,       const int nc, 
                  int * dirow,        int * dicol,
                  int * dirowP,       int * dicolP,
                  float * dv,        int lddv, 
                  float * dwork,     int lddwork,
                  float * dcoords_x, float * dcoords_y,
                  int B,
                  mgard_cuda_handle & handle, bool profile);

template mgard_cuda_ret 
postp_2D_cuda<double>(const int nrow,     const int ncol,
              const int nr,       const int nc, 
              int * dirow,        int * dicol,
              int * dirowP,       int * dicolP,
              double * dv,        int lddv, 
              double * dwork,     int lddwork,
              double * dcoords_x, double * dcoords_y,
              int B,
              mgard_cuda_handle & handle, bool profile);
template mgard_cuda_ret 
postp_2D_cuda<float>(const int nrow,     const int ncol,
              const int nr,       const int nc, 
              int * dirow,        int * dicol,
              int * dirowP,       int * dicolP,
              float * dv,        int lddv, 
              float * dwork,     int lddwork,
              float * dcoords_x, float * dcoords_y,
              int B,
              mgard_cuda_handle & handle, bool profile);

} //end namespace mgard_gen

} //end namespace mard_2d
