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
namespace mgard_gen {
template <typename T> 
mgard_cuda_ret 
refactor_2D_cuda_compact_l2(const int l_target,
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

  T * dcwork2;
  size_t dcwork_pitch2;
  cudaMallocPitchHelper((void**)&dcwork2, &dcwork_pitch2, nc * sizeof(T), nr);
  int lddcwork2 = dcwork_pitch2 / sizeof(T);

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



  int * nr_l = new int[l_target+1];
  int * nc_l = new int[l_target+1];

  int ** cirow_l = new int*[l_target+1];
  int ** cicol_l = new int*[l_target+1];

  T ** ccoords_y_l = new T*[l_target+1];
  T ** ccoords_x_l = new T*[l_target+1];

  int ** dcirow_l = new int*[l_target+1];
  int ** dcicol_l = new int*[l_target+1];

  T ** dccoords_y_l = new T*[l_target+1];
  T ** dccoords_x_l = new T*[l_target+1];

  for (int l = 0; l < l_target+1; l++) {
    int stride = std::pow(2, l);
    nr_l[l] = ceil((float)nr/std::pow(2, l));
    nc_l[l] = ceil((float)nc/std::pow(2, l));
    cirow_l[l] = new int[nr_l[l]];
    cicol_l[l] = new int[nc_l[l]];

    ccoords_y_l[l] = new T[nr_l[l]];
    ccoords_x_l[l] = new T[nc_l[l]];

    for (int i = 0; i < nr_l[l]; i++) {
      cirow_l[l][i] = i;
      ccoords_y_l[l][i] = ccoords_y[i * stride];
    }

    for (int i = 0; i < nc_l[l]; i++) {
      cicol_l[l][i] = i;
      ccoords_x_l[l][i] = ccoords_x[i * stride];
    }

    cudaMallocHelper((void**)&(dcirow_l[l]), nr_l[l] * sizeof(int));
    cudaMemcpyAsyncHelper(dcirow_l[l], cirow_l[l], nr_l[l] * sizeof(int), H2D,
                          handle, 0, profile);

    cudaMallocHelper((void**)&(dcicol_l[l]), nc_l[l] * sizeof(int));
    cudaMemcpyAsyncHelper(dcicol_l[l], cicol_l[l], nc_l[l] * sizeof(int), H2D,
                          handle, 0, profile);

    cudaMallocHelper((void**)&(dccoords_y_l[l]), nr_l[l] * sizeof(T));
    cudaMemcpyAsyncHelper(dccoords_y_l[l], ccoords_y_l[l], nr_l[l] * sizeof(T), H2D,
                          handle, 0, profile);

    cudaMallocHelper((void**)&(dccoords_x_l[l]), nc_l[l] * sizeof(T));
    cudaMemcpyAsyncHelper(dccoords_x_l[l], ccoords_x_l[l], nc_l[l] * sizeof(T), H2D,
                          handle, 0, profile);
  }




  mgard_cuda_ret ret;
  double total_time = 0.0;
  std::ofstream timing_results;
  if (profile) {
    timing_results.open (handle.csv_prefix + "refactor_2D_cuda_cpt_l2.csv");
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

    pow2p1_to_cpt_time = 0.0;
    cpt_to_pow2p1_time = 0.0;
    // -> change funcs in pi_QL to use _l functions, otherwise distances are
    // wrong!!!
    // pi_Ql(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec,
    //       col_vec); // rename!. v@l has I-\Pi_l Q_l+1 u

    int row_stride = stride;
    int col_stride = stride;

    // std::cout << "before dcv = \n";
    // print_matrix_cuda(nr, nc, dcv, lddcv);
     
    ret = pow2p1_to_cpt(nr,    nc,
                        row_stride, col_stride,
                        dcv,        lddcv, 
                        dcwork,     lddcwork, B,
                        handle, 0, profile);
    pow2p1_to_cpt_time += ret.time;

    // std::cout << "before dcwork = \n";
    // print_matrix_cuda(nr_l[l], nc_l[l], dcwork, lddcwork);

    // std::cout << "nr_l[l] = " << nr_l[l] << std::endl;
    // std::cout << "nc_l[l] = " << nc_l[l] << std::endl;

    // std::cout << "dcirow_l[l] = \n";
    // print_matrix_cuda(1, nr_l[l], dcirow_l[l], nr_l[l]);

    // std::cout << "dcicol_l[l] = \n";
    // print_matrix_cuda(1, nc_l[l], dcicol_l[l], nc_l[l]);

    // std::cout << "dccoords_y_l[l] = \n";
    // print_matrix_cuda(1, nr_l[l], dccoords_y_l[l], nr_l[l]);

    // std::cout << "dccoords_x_l[l] = \n";
    // print_matrix_cuda(1, nc_l[l], dccoords_x_l[l], nc_l[l]);

    row_stride = 1;
    col_stride = 1;
    ret = pi_Ql_cuda(nr_l[l],         nc_l[l], 
                     nr_l[l],         nc_l[l], 
                     row_stride,      col_stride,
                     dcirow_l[l],     dcicol_l[l],
                     dcwork,          lddcwork,
                     dccoords_x_l[l], dccoords_y_l[l], B,
                     handle, 0, profile);
    pi_Ql_cuda_time = ret.time;

    // std::cout << "after dcwork = \n";
    // print_matrix_cuda(nr_l[l], nc_l[l], dcwork, lddcwork);

    row_stride = stride;
    col_stride = stride;
    ret = cpt_to_pow2p1(nr,         nc, 
                        row_stride, col_stride,
                        dcwork,     lddcwork,
                        dcv,        lddcv, B,
                        handle, 0, profile);
    cpt_to_pow2p1_time += ret.time;

    // std::cout << "after dcv = \n";
    // print_matrix_cuda(nr, nc, dcv, lddcv);


    // ret = pi_Ql_cuda(nr,              nc,
    //                  nr,              nc,
    //                  row_stride,      col_stride,
    //                  dcirow,          dcicol,
    //                  dcv,             lddcv, 
    //                  dccoords_x,      dccoords_y);
    // pi_Ql_cuda_time += ret.time;

    

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
                            dwork,      lddwork, B,
                            handle, 0, profile);
    copy_level_l_cuda_time = ret.time;

    // assign_num_level_l(l + 1, work.data(), 0.0, nr, nc, nrow, ncol);

    row_stride = Cstride;
    col_stride = Cstride;
    ret = pow2p1_to_cpt(nr,    nc,
                        row_stride, col_stride,
                        dwork,      lddwork,
                        dcwork,     lddcwork, B,
                        handle, 0, profile);
    pow2p1_to_cpt_time += ret.time;
    row_stride = 1;
    col_stride = 1;
    T val = 0.0;
    ret = assign_num_level_l_cuda(nr_l[l+1],         nc_l[l+1], 
                                  nr_l[l+1],         nc_l[l+1], 
                                  row_stride, col_stride,
                                  dcirow_l[l+1],     dcicol_l[l+1],
                                  dcwork,      lddcwork, 
                                  val, B,
                                  handle, 0, profile);
    assign_num_level_l_cuda_time = ret.time;

    row_stride = Cstride;
    col_stride = Cstride;
    ret = cpt_to_pow2p1(nr,         nc, 
                        row_stride, col_stride,
                        dcwork,     lddcwork,
                        dwork,      lddwork, B,
                        handle, 0, profile);
    cpt_to_pow2p1_time += ret.time;


    // row_stride = Cstride;
    // col_stride = Cstride;
    // ret = assign_num_level_l_cuda(nr,         nc,
    //                               nr,         nc,
    //                               row_stride, col_stride,
    //                               dcirow,     dcicol,
    //                               dwork,      lddwork, 
    //                               0.0);
    // assign_num_level_l_cuda_time += ret.time;

    row_stride = 1;
    col_stride = stride;
    ret = pow2p1_to_cpt(nr,    nc,
                        row_stride, col_stride,
                        dwork,      lddwork,
                        dcwork,     lddcwork, B,
                        handle, 0, profile);
    pow2p1_to_cpt_time += ret.time;


    // double * work = new double[nr * nc];
    // for (int i = 0; i < nr; i++) {
    //   for (int j = 0; j < nc; j++) {
    //     work[i * nc + j] = i * nc + j;
    //   }
    // }

    // cudaMemcpy2DHelper(dcwork, lddcwork * sizeof(double),
    //                    work,  nc * sizeof(double),
    //                    nc * sizeof(double), nr, H2D);

    // print_matrix_cuda(nr,    nc, dcwork,     lddcwork);
    // cudaMemcpy2DHelper(dcwork2, lddcwork2 * sizeof(double),
    //                    dcwork,  lddcwork * sizeof(double),
    //                    nc * sizeof(double), nr, D2D);
    row_stride = 1;
    col_stride = 1;
 
    // ret = mass_mult_l_row_cuda_sm(nr_l[0],    nc_l[l],
    //                               nr_l[0],    nc_l[l],
    //                               row_stride, col_stride,
    //                               dcirow_l[0], dcicol_l[l],
    //                               dcwork,     lddcwork,
    //                               dccoords_x_l[l],
    //                               16, 16);

    // print_matrix_cuda(nr,    nc, dcwork2,     lddcwork2);
 
    ret = mass_mult_l_row_cuda(nr_l[0],    nc_l[l],
                               nr_l[0],    nc_l[l],
                               row_stride, col_stride,
                               dcirow_l[0], dcicol_l[l],
                               dcwork,     lddcwork,
                               dccoords_x_l[l], B,
                               handle, 0, profile);
    mass_mult_l_row_cuda_time = ret.time;

    // compare_matrix_cuda(nr,    nc,
    //                     dcwork,     lddcwork,
    //                     dcwork2,     lddcwork2);

    // print_matrix_cuda(nr,    nc, dcwork,     lddcwork);

    row_stride = 1;
    col_stride = stride;
    ret = cpt_to_pow2p1(nr,         nc, 
                        row_stride, col_stride,
                        dcwork,     lddcwork,
                        dwork,      lddwork, B,
                        handle, 0, profile);
    cpt_to_pow2p1_time += ret.time;

    
    // row_stride = 1;
    // col_stride = stride;
    // ret = mass_mult_l_row_cuda(nr,         nc,
    //                            nr,         nc,
    //                            row_stride, col_stride,
    //                            dcirow,     dcicol,
    //                            dwork,     lddwork,
    //                            dccoords_x);
    // mass_mult_l_row_cuda_time += ret.time;


    row_stride = 1;
    col_stride = stride;
    ret = pow2p1_to_cpt(nr,    nc,
                        row_stride, col_stride,
                        dwork,      lddwork,
                        dcwork,     lddcwork, B,
                        handle, 0, profile);
    pow2p1_to_cpt_time += ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = restriction_l_row_cuda(nr_l[0],    nc_l[l],
                                 nr_l[0],    nc_l[l],
                                 row_stride, col_stride,
                                 dcirow_l[0], dcicol_l[l],
                                 dcwork,      lddcwork,
                                 dccoords_x_l[l], B,
                                 handle, 0, profile);
    restriction_l_row_cuda_time = ret.time;

    row_stride = 1;
    col_stride = stride;
    ret = cpt_to_pow2p1(nr,         nc, 
                        row_stride, col_stride,
                        dcwork,     lddcwork,
                        dwork,      lddwork, B,
                        handle, 0, profile);
    cpt_to_pow2p1_time += ret.time;

    // row_stride = 1;
    // col_stride = Cstride;
    // ret = restriction_l_row_cuda(nr,         nc,
    //                              nr,         nc,
    //                              row_stride, col_stride,
    //                              dcirow,     dcicol,
    //                              dwork,      lddwork,
    //                              dccoords_x);
    // restriction_l_row_cuda_time += ret.time;

    row_stride = 1;
    col_stride = Cstride;
    ret = pow2p1_to_cpt(nr,    nc,
                        row_stride, col_stride,
                        dwork,      lddwork,
                        dcwork,     lddcwork, B,
                        handle, 0, profile);
    pow2p1_to_cpt_time += ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = solve_tridiag_M_l_row_cuda(nr_l[0],    nc_l[l+1],
                                     nr_l[0],         nc_l[l+1],
                                     row_stride, col_stride,
                                     dcirow_l[0],     dcicol_l[l+1],
                                     dcwork,     lddcwork,
                                     dccoords_x_l[l+1], B,
                                     handle, 0, profile);
    solve_tridiag_M_l_row_cuda_time = ret.time;

    row_stride = 1;
    col_stride = Cstride;
    ret = cpt_to_pow2p1(nr,         nc, 
                        row_stride, col_stride,
                        dcwork,     lddcwork,
                        dwork,      lddwork, B,
                        handle, 0, profile);
    cpt_to_pow2p1_time += ret.time;


    // row_stride = 1;
    // col_stride = Cstride;
    // ret = solve_tridiag_M_l_row_cuda(nr,         nc,
    //                                  nr,         nc,
    //                                  row_stride, col_stride,
    //                                  dcirow,     dcicol,
    //                                  dwork,     lddwork,
    //                                  dccoords_x);
    // solve_tridiag_M_l_row_cuda_time += ret.time;


    if (nrow > 1) // do this if we have an 2-dimensional array
    {

      row_stride = stride;
      col_stride = Cstride;
      ret = pow2p1_to_cpt(nr,    nc,
                          row_stride, col_stride,
                          dwork,      lddwork,
                          dcwork,     lddcwork, B,
                          handle, 0, profile);
      pow2p1_to_cpt_time += ret.time;


      row_stride = 1;
      col_stride = 1;
      ret = mass_mult_l_col_cuda(nr_l[l],     nc_l[l+1],
                                 nr_l[l],     nc_l[l+1],
                                 row_stride,  col_stride,
                                 dcirow_l[l], dcicol_l[l+1],
                                 dcwork,     lddcwork,
                                 dccoords_y_l[l], B,
                                 handle, 0, profile);
      mass_mult_l_col_cuda_time = ret.time;

      row_stride = stride;
      col_stride = Cstride;
      ret = cpt_to_pow2p1(nr,         nc, 
                          row_stride, col_stride,
                          dcwork,     lddcwork,
                          dwork,      lddwork, B,
                          handle, 0, profile);
      cpt_to_pow2p1_time += ret.time;

      // row_stride = stride;
      // col_stride = Cstride;
      // ret = mass_mult_l_col_cuda(nr,         nc,
      //                            nr,         nc,
      //                            row_stride, col_stride,
      //                            dcirow,     dcicol,
      //                            dwork,     lddwork,
      //                            dccoords_y);
      // mass_mult_l_col_cuda_time += ret.time;

      row_stride = stride;
      col_stride = Cstride;
      ret = pow2p1_to_cpt(nr,    nc,
                          row_stride, col_stride,
                          dwork,      lddwork,
                          dcwork,     lddcwork, B,
                          handle, 0, profile);
      pow2p1_to_cpt_time += ret.time;

      row_stride = 1;
      col_stride = 1;
      ret = restriction_l_col_cuda(nr_l[l],         nc_l[l+1],
                                   nr_l[l],         nc_l[l+1],
                                   row_stride, col_stride,
                                   dcirow_l[l],     dcicol_l[l+1],
                                   dcwork, lddcwork,
                                   dccoords_y_l[l], B,
                                   handle, 0, profile);
      restriction_l_col_cuda_time = ret.time;

      row_stride = stride;
      col_stride = Cstride;
      ret = cpt_to_pow2p1(nr,         nc, 
                          row_stride, col_stride,
                          dcwork,     lddcwork,
                          dwork,      lddwork, B,
                          handle, 0, profile);
      cpt_to_pow2p1_time += ret.time;


      // row_stride = stride;
      // col_stride = Cstride;
      // ret = restriction_l_col_cuda(nr,         nc,
      //                              nr,         nc,
      //                              row_stride, col_stride,
      //                              dcirow,     dcicol,
      //                              dwork, lddwork,
      //                              dccoords_y);
      // restriction_l_col_cuda_time += ret.time;

      row_stride = Cstride;
      col_stride = Cstride;
      ret = pow2p1_to_cpt(nr,    nc,
                          row_stride, col_stride,
                          dwork,      lddwork,
                          dcwork,     lddcwork, B,
                          handle, 0, profile);
      pow2p1_to_cpt_time += ret.time;
      

      row_stride = 1;
      col_stride = 1;
      ret = solve_tridiag_M_l_col_cuda(nr_l[l+1],         nc_l[l+1],
                                       nr_l[l+1],         nc_l[l+1],
                                       row_stride, col_stride,
                                       dcirow_l[l+1],     dcicol_l[l+1],
                                       dcwork, lddcwork,
                                       dccoords_y_l[l+1], B,
                                       handle, 0, profile);
      solve_tridiag_M_l_col_cuda_time = ret.time;

      row_stride = Cstride;
      col_stride = Cstride;
      ret = cpt_to_pow2p1(nr,         nc, 
                          row_stride, col_stride,
                          dcwork,     lddcwork,
                          dwork,      lddwork, B,
                          handle, 0, profile);
      cpt_to_pow2p1_time += ret.time;


      // row_stride = Cstride;
      // col_stride = Cstride;
      // ret = solve_tridiag_M_l_col_cuda(nr,         nc,
      //                                  nr,         nc,
      //                                  row_stride, col_stride,
      //                                  dcirow,     dcicol,
      //                                  dwork, lddwork,
      //                                  dccoords_y);
      // solve_tridiag_M_l_col_cuda_time += ret.time;

    }

    // // Solved for (z_l, phi_l) = (c_{l+1}, vl)
    // // add_level_l(l + 1, v, work.data(), nr, nc, nrow, ncol);
    row_stride = Cstride;
    col_stride = Cstride;
    ret = add_level_l_cuda(nr,       nc,
                           nr,         nc,
                           row_stride, col_stride,
                           dcirow,     dcicol,
                           dcv,        lddcv, 
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

  } //end of loop

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
  cudaFreeHelper(dcwork2);
  cudaFreeHelper(dcicol);
  cudaFreeHelper(dcirow);
  cudaFreeHelper(dccoords_y);
  cudaFreeHelper(dccoords_x);

  for (int l = 0; l < l_target+1; l++) {
    cudaFreeHelper(dcirow_l[l]);
    cudaFreeHelper(dcicol_l[l]);
    cudaFreeHelper(dccoords_y_l[l]);
    cudaFreeHelper(dccoords_x_l[l]);
  }


  return mgard_cuda_ret(0, total_time);
}

template mgard_cuda_ret 
refactor_2D_cuda_compact_l2<double>(const int l_target,
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
refactor_2D_cuda_compact_l2<float>(const int l_target,
                    const int nrow,     const int ncol,
                    const int nr,       const int nc, 
                    int * dirow,        int * dicol,
                    int * dirowP,       int * dicolP,
                    float * dv,        int lddv, 
                    float * dwork,     int lddwork,
                    float * dcoords_x, float * dcoords_y,
                    int B,
                    mgard_cuda_handle & handle, bool profile);

} // end mgard_gen
} // end mgard_2d
