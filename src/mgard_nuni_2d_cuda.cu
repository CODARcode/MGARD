#include "mgard_nuni.h"
#include "mgard.h"

#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"
#include "mgard_nuni_2d_cuda_kernels.h"

#include <fstream>

namespace mgard_2d {
namespace mgard_gen {


void 
prep_2D_cuda(const int nrow,     const int ncol,
             const int nr,       const int nc, 
             int * dirow,        int * dicol,
             int * dirowP,       int * dicolP,
             double * dv,        int lddv, 
             double * dwork,     int lddwork,
             double * dcoords_x, double * dcoords_y) {

  mgard_cuda_ret ret;

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
  //int ldv = ncol;
  //int ldwork = ncol;
  
  // pi_Ql_first(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec, col_vec);
  ret = pi_Ql_first_cuda(nrow,      ncol,
                   nr,        nc, 
                   dirow,     dicol,
                   dirowP,    dicolP,
                   dcoords_x, dcoords_y,
                   dv,        lddv); //(I-\Pi)u this is the initial move to 2^k+1 nodes
  pi_Ql_first_cuda_time = ret.time;
  // mgard_cannon::copy_level(nrow, ncol, 0, v, work);
  ret = mgard_cannon::copy_level_cuda(nrow,       ncol, 
                                row_stride, col_stride,
                                dv,         lddv,
                                dwork,      lddwork);
  copy_level_cuda_time = ret.time;

  // assign_num_level_l(0, work.data(), 0.0, nr, nc, nrow, ncol);
  ret = assign_num_level_l_cuda(nrow,       ncol,
                          nr,         nc,
                          row_stride, col_stride,
                          dirow,      dicol,
                          dwork,      lddwork, 
                          0.0);
  assign_num_level_l_cuda_time = ret.time;

  row_stride = 1;
  col_stride = 1;
  ret = mgard_cannon::mass_matrix_multiply_row_cuda(nrow,       ncol,
                                              row_stride, col_stride,
                                              dwork,      lddwork,
                                              dcoords_x);
  mass_matrix_multiply_row_cuda_time = ret.time;

  ret = restriction_first_row_cuda(nrow,       ncol,
                             row_stride, dicolP, nc,
                             dwork,      lddwork,
                             dcoords_x);
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
                                                dcoords_y);
    mass_matrix_multiply_col_cuda_time = ret.time;


    ret = restriction_first_col_cuda(nrow,   ncol,
                               dirowP, nr, col_stride,
                               dwork,  lddwork,
                               dcoords_y);
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
                   dwork,      lddwork);
  add_level_l_cuda_time = ret.time;


  std::ofstream timing_results;
  timing_results.open ("prep_2D_cuda.csv");
  timing_results << "pi_Ql_first_cuda_time," << pi_Ql_first_cuda_time << std::endl;
  timing_results << "copy_level_cuda_time," << copy_level_cuda_time << std::endl;
  timing_results << "assign_num_level_l_cuda_time," << assign_num_level_l_cuda_time << std::endl;

  timing_results << "mass_matrix_multiply_row_cuda_time," << mass_matrix_multiply_row_cuda_time << std::endl;
  timing_results << "restriction_first_row_cuda_time," << restriction_first_row_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_row_cuda_time," << solve_tridiag_M_l_row_cuda_time << std::endl;
  
  timing_results << "mass_matrix_multiply_col_cuda_time," << mass_matrix_multiply_col_cuda_time << std::endl;
  timing_results << "restriction_first_col_cuda_time," << restriction_first_col_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_col_cuda_time," << solve_tridiag_M_l_col_cuda_time << std::endl;
  timing_results << "add_level_l_cuda_time," << add_level_l_cuda_time << std::endl;
  timing_results.close();
}














void 
refactor_2D_cuda(const int l_target,
                 const int nrow,     const int ncol,
                 const int nr,       const int nc, 
                 int * dirow,        int * dicol,
                 int * dirowP,       int * dicolP,
                 double * dv,        int lddv, 
                 double * dwork,     int lddwork,
                 double * dcoords_x, double * dcoords_y) {
  // refactor
  //    //std::cout  << "I am the general refactorer!" <<"\n";
  
  mgard_cuda_ret ret;

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
    // pi_Ql(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec,
    //       col_vec); // rename!. v@l has I-\Pi_l Q_l+1 u
    // print_matrix(nrow, ncol, v, ldv);
    int row_stride = stride;
    int col_stride = stride;
    ret = pi_Ql_cuda(nrow,            ncol,
               nr,              nc,
               row_stride,      col_stride,
               dirow,           dicol,
               dv,              lddv, 
               dcoords_x,       dcoords_y);
    pi_Ql_cuda_time += ret.time;

    // pi_Ql(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec,
    //       col_vec); // rename!. v@l has I-\Pi_l Q_l+1 u

    // copy_level_l(l, v, work.data(), nr, nc, nrow, ncol);
    row_stride = stride;
    col_stride = stride;
    ret = copy_level_l_cuda(nrow,       ncol,
                      nr,         nc,
                      row_stride, col_stride,
                      dirow,      dicol,
                      dv,         lddv, 
                      dwork,      lddwork);
    copy_level_l_cuda_time += ret.time;

    // assign_num_level_l(l + 1, work.data(), 0.0, nr, nc, nrow, ncol);
    row_stride = Cstride;
    col_stride = Cstride;
    ret = assign_num_level_l_cuda(nrow,       ncol,
                            nr,         nc,
                            row_stride, col_stride,
                            dirow,      dicol,
                            dwork,      lddwork, 
                            0.0);
    assign_num_level_l_cuda_time += ret.time;

    row_stride = 1;
    col_stride = stride;
    ret = mass_mult_l_row_cuda(nrow,       ncol,
                         nr,         nc,
                         row_stride, col_stride,
                         dirow,      dicol,
                         dwork,      lddwork,
                         dcoords_x);
    mass_mult_l_row_cuda_time += ret.time;


    row_stride = 1;
    col_stride = stride;
    ret = restriction_l_row_cuda(nrow,       ncol,
                           nr,         nc,
                           row_stride, col_stride,
                           dirow,      dicol,
                           dwork,      lddwork,
                           dcoords_x);
    restriction_l_row_cuda_time += ret.time;

    row_stride = 1;
    col_stride = Cstride;
    ret = solve_tridiag_M_l_row_cuda(nrow,       ncol,
                               nr,         nc,
                               row_stride, col_stride,
                               dirow,      dicol,
                               dwork,      lddwork,
                               dcoords_x);
    solve_tridiag_M_l_row_cuda_time += ret.time;

    // row-sweep
    // std::cout << "cpu: ";
    for (int i = 0; i < nr; ++i) {
      int ir = get_lindex_cuda(nr, nrow, i);
      // std::cout << ir << " ";
      for (int j = 0; j < ncol; ++j) {
        //row_vec[j] = work[mgard_common::get_index_cuda(ncol, ir, j)];
      }

      // mgard_gen::mass_mult_l(l, row_vec, coords_x, nc, ncol);

      // mgard_gen::restriction_l(l + 1, row_vec, coords_x, nc, ncol);

      // mgard_gen::solve_tridiag_M_l(l + 1, row_vec, coords_x, nc, ncol);

      for (int j = 0; j < ncol; ++j) {
        //work[mgard_common::get_index_cuda(ncol, ir, j)] = row_vec[j];
      }
    }
    // std::cout << std::endl;

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
                                 dcoords_y);
      mass_mult_l_col_cuda_time += ret.time;


      row_stride = stride;
      col_stride = Cstride;
      ret = restriction_l_col_cuda(nrow,       ncol,
                                   nr,         nc,
                                   row_stride, col_stride,
                                   dirow,      dicol,
                                   dwork, lddwork,
                                   dcoords_y);
      restriction_l_col_cuda_time += ret.time;

      row_stride = Cstride;
      col_stride = Cstride;
      ret = solve_tridiag_M_l_col_cuda(nrow,       ncol,
                                       nr,         nc,
                                       row_stride, col_stride,
                                       dirow,       dicol,
                                       dwork, lddwork,
                                       dcoords_y);
      solve_tridiag_M_l_col_cuda_time += ret.time;
      // std::cout << "cpu: ";
      for (int j = 0; j < nc; j += Cstride) {
        int jr = get_lindex_cuda(nc, ncol, j);
        // std::cout << jr << " ";
        for (int i = 0; i < nrow; ++i) {
          //col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, jr)];
        }


        // mgard_gen::mass_mult_l(l, col_vec, coords_y, nr, nrow);
        // mgard_gen::restriction_l(l + 1, col_vec, coords_y, nr, nrow);
        // mgard_gen::solve_tridiag_M_l(l + 1, col_vec, coords_y, nr, nrow);

        for (int i = 0; i < nrow; ++i) {
         // work[mgard_common::get_index_cuda(ncol, i, jr)] = col_vec[i];
        }
      }
      // std::cout<<std::endl;
    }

    // Solved for (z_l, phi_l) = (c_{l+1}, vl)
    // add_level_l(l + 1, v, work.data(), nr, nc, nrow, ncol);
    row_stride = Cstride;
    col_stride = Cstride;
    ret = add_level_l_cuda(nrow,       ncol, 
                     nr,         nc,
                     row_stride, col_stride,
                     dirow,      dicol,
                     dv,         lddv, 
                     dwork,      lddwork);
    add_level_cuda_time += ret.time;
  }

  std::ofstream timing_results;
  timing_results.open ("refactor_2D_cuda.csv");
  timing_results << "pi_Ql_cuda_time," << pi_Ql_cuda_time << std::endl;
  timing_results << "copy_level_l_cuda_time," << copy_level_l_cuda_time << std::endl;
  timing_results << "assign_num_level_l_cuda_time," << assign_num_level_l_cuda_time << std::endl;

  timing_results << "mass_mult_l_row_cuda_time," << mass_mult_l_row_cuda_time << std::endl;
  timing_results << "restriction_l_row_cuda_time," << restriction_l_row_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_row_cuda_time," << solve_tridiag_M_l_row_cuda_time << std::endl;

  timing_results << "mass_mult_l_col_cuda_time," << mass_mult_l_col_cuda_time << std::endl;
  timing_results << "restriction_l_col_cuda_time," << restriction_l_col_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_col_cuda_time," << solve_tridiag_M_l_col_cuda_time << std::endl;
  timing_results << "add_level_cuda_time," << add_level_cuda_time << std::endl;
  timing_results.close();
}

void 
recompose_2D_cuda(const int l_target,
                  const int nrow,     const int ncol,
                  const int nr,       const int nc, 
                  int * dirow,        int * dicol,
                  int * dirowP,       int * dicolP,
                  double * dv,        int lddv, 
                  double * dwork,     int lddwork,
                  double * dcoords_x, double * dcoords_y) {
 
  // recompose
  //    //std::cout  << "recomposing" << "\n";

  mgard_cuda_ret ret;
  double copy_level_l_cuda_time = 0.0;
  double assign_num_level_l_cuda_time = 0.0;

  double mass_mult_l_row_cuda_time = 0.0;
  double restriction_l_row_cuda_time = 0.0;
  double solve_tridiag_M_l_row_cuda_time = 0.0;

  double mass_mult_l_col_cuda_time = 0.0;
  double restriction_l_col_cuda_time = 0.0;
  double solve_tridiag_M_l_col_cuda_time = 0.0;

  double subtract_level_l_cuda_time = 0.0;
  double prolongate_l_row_cuda_time = 0.0;
  double prolongate_l_col_cuda_time = 0.0;

  for (int l = l_target; l > 0; --l) {

    int stride = std::pow(2, l); // current stride
    int Pstride = stride / 2;
    int Cstride = stride * 2;

    // copy_level_l(l - 1, v, work.data(), nr, nc, nrow, ncol);
    int row_stride = Pstride;
    int col_stride = Pstride;
    ret = copy_level_l_cuda(nrow,       ncol,
                      nr,         nc,
                      row_stride, col_stride,
                      dirow,      dicol,
                      dv,         lddv, 
                      dwork,      lddwork);
    copy_level_l_cuda_time = ret.time;

    // assign_num_level_l(l, work.data(), 0.0, nr, nc, nrow, ncol);
    row_stride = stride;
    col_stride = stride;
    ret = assign_num_level_l_cuda(nrow,       ncol,
                            nr,         nc,
                            row_stride, col_stride,
                            dirow,      dicol,
                            dwork,      lddwork, 
                            0.0);
    assign_num_level_l_cuda_time += ret.time;

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
                         dcoords_x);
    mass_mult_l_row_cuda_time += ret.time;


    row_stride = 1;
    col_stride = Pstride;
    ret = restriction_l_row_cuda(nrow,       ncol,
                           nr,         nc,
                           row_stride, col_stride,
                           dirow,      dicol,
                           dwork,      lddwork,
                           dcoords_x);
    restriction_l_row_cuda_time += ret.time;


    row_stride = 1;
    col_stride = stride;
    ret = solve_tridiag_M_l_row_cuda(nrow,       ncol,
                               nr,         nc,
                               row_stride, col_stride,
                               dirow,      dicol,
                               dwork,      lddwork,
                               dcoords_x);
    solve_tridiag_M_l_row_cuda_time += ret.time;

    for (int i = 0; i < nr; ++i) {
      int ir = get_lindex_cuda(nr, nrow, i);
      for (int j = 0; j < ncol; ++j) {
        //row_vec[j] = work[mgard_common::get_index_cuda(ncol, ir, j)];
      }

      // mgard_gen::mass_mult_l(l - 1, row_vec, coords_x, nc, ncol);

      // mgard_gen::restriction_l(l, row_vec, coords_x, nc, ncol);

      // mgard_gen::solve_tridiag_M_l(l, row_vec, coords_x, nc, ncol);

      for (int j = 0; j < ncol; ++j) {
        //work[mgard_common::get_index_cuda(ncol, ir, j)] = row_vec[j];
      }
    }

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
                           dcoords_y);
      mass_mult_l_col_cuda_time += ret.time;


      row_stride = Pstride;
      col_stride = stride;
      ret = restriction_l_col_cuda(nrow,       ncol,
                                   nr,         nc,
                                   row_stride, col_stride,
                                   dirow,       dicol,
                                   dwork, lddwork,
                                   dcoords_y);
      restriction_l_col_cuda_time += ret.time;

      row_stride = stride;
      col_stride = stride;
      ret = solve_tridiag_M_l_col_cuda(nrow,       ncol,
                                       nr,         nc,
                                       row_stride, col_stride,
                                       dirow,      dicol,
                                       dwork,      lddwork,
                                       dcoords_y);
      solve_tridiag_M_l_col_cuda_time += ret.time;

      for (int j = 0; j < nc; j += stride) {
        int jr = get_lindex_cuda(nc, ncol, j);
        for (int i = 0; i < nrow; ++i) {
         // col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, jr)];
        }

        // mgard_gen::mass_mult_l(l - 1, col_vec, coords_y, nr, nrow);

        // mgard_gen::restriction_l(l, col_vec, coords_y, nr, nrow);

        // mgard_gen::solve_tridiag_M_l(l, col_vec, coords_y, nr, nrow);

        for (int i = 0; i < nrow; ++i) {
         // work[mgard_common::get_index_cuda(ncol, i, jr)] = col_vec[i];
        }
      }
    }

    // subtract_level_l(l, work.data(), v, nr, nc, nrow, ncol); // do -(Qu - zl)
    row_stride = stride;
    col_stride = stride;
    ret = subtract_level_l_cuda(nrow,       ncol, 
                                nr,         nc,
                                row_stride, col_stride,
                                dirow,      dicol,
                                dwork,      lddwork,
                                dv,         lddv);
    subtract_level_l_cuda_time += ret.time;

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
                                dcoords_x);
    prolongate_l_row_cuda_time += ret.time;



    for (int i = 0; i < nr; i += stride) {
      int ir = get_lindex_cuda(nr, nrow, i);
      for (int j = 0; j < ncol; ++j) {
        //row_vec[j] = work[mgard_common::get_index_cuda(ncol, ir, j)];
      }

      // mgard_gen::prolongate_l(l, row_vec, coords_x, nc, ncol);

      for (int j = 0; j < ncol; ++j) {
        //work[mgard_common::get_index_cuda(ncol, ir, j)] = row_vec[j];
      }
    }

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
                             dcoords_y);
      prolongate_l_col_cuda_time += ret.time;

      for (int j = 0; j < nc; j += Pstride) {
        int jr = get_lindex_cuda(nc, ncol, j);
        for (int i = 0; i < nrow; ++i) // copy all rows
        {
          //col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, jr)];
        }

        // mgard_gen::prolongate_l(l, col_vec, coords_y, nr, nrow);

        for (int i = 0; i < nrow; ++i) {
          //work[mgard_common::get_index_cuda(ncol, i, jr)] = col_vec[i];
        }
      }
    }

    // assign_num_level_l(l, v, 0.0, nr, nc, nrow, ncol);
    row_stride = stride;
    col_stride = stride;
    ret = assign_num_level_l_cuda(nrow,       ncol,
                            nr,         nc,
                            row_stride, col_stride,
                            dirow,      dicol,
                            dv,         lddv, 
                            0.0);
    assign_num_level_l_cuda_time += ret.time;

    // subtract_level_l(l - 1, v, work.data(), nr, nc, nrow, ncol);

    row_stride = Pstride;
    col_stride = Pstride;
    ret = subtract_level_l_cuda(nrow,       ncol, 
                          nr,         nc,
                          row_stride, col_stride,
                          dirow,      dicol,
                          dv,         lddv,
                          dwork,      lddwork);
    subtract_level_l_cuda_time += ret.time;
  }

  std::ofstream timing_results;
  timing_results.open ("recompose_2D_cuda.csv");
  timing_results << "copy_level_l_cuda_time," << copy_level_l_cuda_time << std::endl;
  timing_results << "assign_num_level_l_cuda_time," << assign_num_level_l_cuda_time << std::endl;

  timing_results << "mass_mult_l_row_cuda_time," << mass_mult_l_row_cuda_time << std::endl;
  timing_results << "restriction_l_row_cuda_time," << restriction_l_row_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_row_cuda_time," << solve_tridiag_M_l_row_cuda_time << std::endl;

  timing_results << "mass_mult_l_col_cuda_time," << mass_mult_l_col_cuda_time << std::endl;
  timing_results << "restriction_l_col_cuda_time," << restriction_l_col_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_col_cuda_time," << solve_tridiag_M_l_col_cuda_time << std::endl;

  timing_results << "subtract_level_l_cuda_time," << subtract_level_l_cuda_time << std::endl;
  timing_results << "prolongate_l_row_cuda_time," << prolongate_l_row_cuda_time << std::endl;
  timing_results << "prolongate_l_col_cuda_time," << prolongate_l_col_cuda_time << std::endl;
  timing_results.close();

}


void 
postp_2D_cuda(const int nrow,     const int ncol,
              const int nr,       const int nc, 
              int * dirow,        int * dicol,
              int * dirowP,       int * dicolP,
              double * dv,        int lddv, 
              double * dwork,     int lddwork,
              double * dcoords_x, double * dcoords_y) {


  mgard_cuda_ret ret;
  double copy_level_cuda_time = 0.0;
  double assign_num_level_l_cuda_time = 0.0;

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

 // mgard_cannon::copy_level(nrow, ncol, 0, v, work);
  int row_stride = 1;
  int col_stride = 1;
  ret = mgard_cannon::copy_level_cuda(nrow,       ncol, 
                                row_stride, col_stride,
                                dv,         lddv,
                                dwork,      lddwork);
  copy_level_cuda_time += ret.time;

  // assign_num_level_l(0, work.data(), 0.0, nr, nc, nrow, ncol);

  row_stride = 1;
  col_stride = 1;
  ret = assign_num_level_l_cuda(nrow,       ncol,
                          nr,         nc,
                          row_stride, col_stride,
                          dirow,      dicol,
                          dwork,      lddwork,
                          0.0);
  assign_num_level_l_cuda_time += ret.time;

  row_stride = 1;
  col_stride = 1;
  ret = mgard_cannon::mass_matrix_multiply_row_cuda(nrow,       ncol, 
                                              row_stride, col_stride,
                                              dwork,      lddwork,
                                              dcoords_x);
  mass_matrix_multiply_row_cuda_time += ret.time;


  row_stride = 1;
  col_stride = 1;
  ret = restriction_first_row_cuda(nrow,       ncol, 
                             row_stride, dicolP, nc,
                             dwork,      lddwork,
                             dcoords_x);
  restriction_first_row_cuda_time += ret.time;

  row_stride = 1;
  col_stride = 1;
  ret = solve_tridiag_M_l_row_cuda(nrow,       ncol,
                             nr,         nc,
                             row_stride, col_stride,
                             dirow,      dicol,
                             dwork,      lddwork,
                             dcoords_x);
  solve_tridiag_M_l_row_cuda_time += ret.time;

  for (int i = 0; i < nrow; ++i) {
    int ir = get_lindex_cuda(nr, nrow, i);
    for (int j = 0; j < ncol; ++j) {
      //row_vec[j] = work[mgard_common::get_index_cuda(ncol, i, j)];
    }

    // mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);

    // restriction_first(row_vec, coords_x, nc, ncol);

    for (int j = 0; j < ncol; ++j) {
     // work[mgard_common::get_index_cuda(ncol, i, j)] = row_vec[j];
    }
  }

  for (int i = 0; i < nr; ++i) {
    int ir = get_lindex_cuda(nr, nrow, i);
    for (int j = 0; j < ncol; ++j) {
      //row_vec[j] = work[mgard_common::get_index_cuda(ncol, ir, j)];
    }

    // mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol);

    for (int j = 0; j < ncol; ++j) {
     // work[mgard_common::get_index_cuda(ncol, ir, j)] = row_vec[j];
    }
  }

  //   //   //std::cout  << "recomposing-colsweep" << "\n";

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) // check if we have 1-D array..
  {
    
    row_stride = 1;
    col_stride = 1;
    ret = mgard_cannon::mass_matrix_multiply_col_cuda(nrow,      ncol,
                                               row_stride, col_stride,
                                               dwork,      lddwork,
                                               dcoords_y);
    mass_matrix_multiply_col_cuda_time += ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = restriction_first_col_cuda(nrow,   ncol, 
                               dirowP, nr,   col_stride,
                               dwork,  lddwork,
                               dcoords_y);
    restriction_first_col_cuda_time += ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = solve_tridiag_M_l_col_cuda(nrow,       ncol,
                               nr,         nc,
                               row_stride, col_stride,
                               dirow,      dicol,
                               dwork,      lddwork,
                               dcoords_y);
    solve_tridiag_M_l_col_cuda_time += ret.time;

    for (int j = 0; j < ncol; ++j) {
      int jr  = get_lindex_cuda(nc,  ncol,  j);
      for (int i = 0; i < nrow; ++i) {
       // col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, j)];
      }

      // mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);

      // mgard_gen::restriction_first(col_vec, coords_y, nr, nrow);

      for (int i = 0; i < nrow; ++i) {
       // work[mgard_common::get_index_cuda(ncol, i, j)] = col_vec[i];
      }
    }

    for (int j = 0; j < nc; ++j) {
      int jr = get_lindex_cuda(nc, ncol, j);
      for (int i = 0; i < nrow; ++i) {
        //col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, jr)];
      }

      // mgard_gen::solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow);
      for (int i = 0; i < nrow; ++i) {
        //work[mgard_common::get_index_cuda(ncol, i, jr)] = col_vec[i];
      }
    }
  }

  // subtract_level_l(0, work.data(), v, nr, nc, nrow, ncol); // do -(Qu - zl)
  row_stride = 1;
  col_stride = 1;
  ret = subtract_level_l_cuda(nrow,       ncol, 
                        nr,         nc,
                        row_stride, col_stride,
                        dirow,      dicol,
                        dwork,      lddwork,
                        dv,         lddv);
  subtract_level_l_cuda_time += ret.time;


  //        //std::cout  << "recomposing-rowsweep2" << "\n";

  //     //   //int Pstride = stride/2; //finer stride
  
  row_stride = 1;
  col_stride = 1;
  ret = prolongate_last_row_cuda(nrow,       ncol, 
                           nr,         nc,
                           row_stride, col_stride,
                           dirow,      dicolP,
                           dwork,      lddwork,
                           dcoords_x);
  prolongate_last_row_cuda_time += ret.time;

  //   //   // row-sweep
  // std::cout << "cpu: ";
  for (int i = 0; i < nr; ++i) {
    int ir = get_lindex_cuda(nr, nrow, i);
    // std::cout << ir << " ";
    for (int j = 0; j < ncol; ++j) {
      //row_vec[j] = work[mgard_common::get_index_cuda(ncol, ir, j)];
    }

    // mgard_gen::prolongate_last(row_vec, coords_x, nc, ncol);

    for (int j = 0; j < ncol; ++j) {
     // work[mgard_common::get_index_cuda(ncol, ir, j)] = row_vec[j];
    }
  }
  // std::cout << std::endl;

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) {
    // print_matrix(nrow, ncol, work.data(), ldwork);
    row_stride = 1;
    col_stride = 1;
    ret = prolongate_last_col_cuda(nrow,       ncol, 
                             nr,         nc,
                             row_stride, col_stride,
                             dirowP,     dicol,
                             dwork,      lddwork,
                             dcoords_y);
    prolongate_last_col_cuda_time += ret.time;
    // print_matrix(nrow, ncol, work.data(), ldwork);


    for (int j = 0; j < ncol; ++j) {
      int jr  = get_lindex_cuda(nc,  ncol,  j);
      for (int i = 0; i < nrow; ++i) // copy all rows
      {
       // col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, j)];
      }

      // mgard_gen::prolongate_last(col_vec, coords_y, nr, nrow);

      for (int i = 0; i < nrow; ++i) {
       // work[mgard_common::get_index_cuda(ncol, i, j)] = col_vec[i];
      }
    }
  }
  // print_matrix(nrow, ncol, work.data(), ldwork);
  


  // assign_num_level_l(0, v, 0.0, nr, nc, nrow, ncol);

  ret = assign_num_level_l_cuda(nrow,       ncol,
                          nr,         nc,
                          row_stride, col_stride,
                          dirow,      dicol,
                          dv,         lddv,
                          0.0);
  assign_num_level_l_cuda_time += ret.time;

  // mgard_cannon::subtract_level(nrow, ncol, 0, v, work.data());
  row_stride = 1;
  col_stride = 1;
  ret = mgard_cannon::subtract_level_cuda(nrow,       ncol, 
                                    row_stride, col_stride,
                                    dv,         lddv, 
                                    dwork,      lddwork); 
  subtract_level_cuda_time += ret.time;

  std::ofstream timing_results;
  timing_results.open ("postp_2D_cuda.csv");
  timing_results << "copy_level_cuda_time," << copy_level_cuda_time << std::endl;
  timing_results << "assign_num_level_l_cuda_time," << assign_num_level_l_cuda_time << std::endl;

  timing_results << "mass_matrix_multiply_row_cuda_time," << mass_matrix_multiply_row_cuda_time << std::endl;
  timing_results << "restriction_first_row_cuda_time," << restriction_first_row_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_row_cuda_time," << solve_tridiag_M_l_row_cuda_time << std::endl;

  timing_results << "mass_matrix_multiply_col_cuda_time," << mass_matrix_multiply_col_cuda_time << std::endl;
  timing_results << "restriction_first_col_cuda_time," << restriction_first_col_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_col_cuda_time," << solve_tridiag_M_l_col_cuda_time << std::endl;

  timing_results << "subtract_level_l_cuda_time," << subtract_level_l_cuda_time << std::endl;
  timing_results << "prolongate_last_row_cuda_time," << prolongate_last_row_cuda_time << std::endl;
  timing_results << "prolongate_last_col_cuda_time," << prolongate_last_col_cuda_time << std::endl;

  timing_results << "subtract_level_cuda_time," << subtract_level_cuda_time << std::endl;
  timing_results.close();
}





} //end namespace mgard_gen

} //end namespace mard_2d
