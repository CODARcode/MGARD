#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include "mgard_nuni_2d_cuda_mass_mult_l.h"
#include <fstream>
#include <cmath>

namespace mgard_2d {
namespace mgard_gen {  

void 
refactor_2D_cuda_compact_l2_sm_pf(const int l_target,
                    const int nrow,     const int ncol,
                    const int nr,       const int nc, 
                    int * dirow,        int * dicol,
                    int * dirowP,       int * dicolP,
                    double * dv,        int lddv, 
                    double * dwork,     int lddwork,
                    double * dcoords_x, double * dcoords_y) {

  double * dcv;
  size_t dcv_pitch;
  cudaMallocPitchHelper((void**)&dcv, &dcv_pitch, nc * sizeof(double), nr);
  int lddcv = dcv_pitch / sizeof(double);

  double * dcwork;
  size_t dcwork_pitch;
  cudaMallocPitchHelper((void**)&dcwork, &dcwork_pitch, nc * sizeof(double), nr);
  int lddcwork = dcwork_pitch / sizeof(double);

  double * dcwork2;
  size_t dcwork_pitch2;
  cudaMallocPitchHelper((void**)&dcwork2, &dcwork_pitch2, nc * sizeof(double), nr);
  int lddcwork2 = dcwork_pitch2 / sizeof(double);

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
  cudaMemcpyHelper(dcirow, cirow, nr * sizeof(int), H2D);

  //std::cout <<"test2\n";
  int * dcicol;
  cudaMallocHelper((void**)&dcicol, nc * sizeof(int));
  cudaMemcpyHelper(dcicol, cicol, nc * sizeof(int), H2D);

 // std::cout <<"test3\n";
  double * coords_x = new double[ncol];
  double * coords_y = new double[nrow];
  cudaMemcpyHelper(coords_x, dcoords_x, ncol * sizeof(double), D2H);
  //std::cout <<"test4\n";
  cudaMemcpyHelper(coords_y, dcoords_y, nrow * sizeof(double), D2H);


  int * irow = new int[nr];
  int * icol = new int[nc];
  //std::cout <<"test5\n";
  cudaMemcpyHelper(irow, dirow, nr * sizeof(int), D2H);
 // std::cout <<"test6\n";
  cudaMemcpyHelper(icol, dicol, nc * sizeof(int), D2H);

  double * ccoords_x = new double[nc];
  double * ccoords_y = new double[nr];

  for (int i = 0; i < nc; i++) {
    ccoords_x[i] = coords_x[icol[i]];
  }

  for (int i = 0; i < nr; i++) {
    ccoords_y[i] = coords_y[irow[i]];
  }

  double * dccoords_x;
  //std::cout <<"test7\n";
  cudaMallocHelper((void**)&dccoords_x, nc * sizeof(double));
  cudaMemcpyHelper(dccoords_x, ccoords_x, nc * sizeof(double), H2D);

  double * dccoords_y;
 // std::cout <<"test8\n"; 
  cudaMallocHelper((void**)&dccoords_y, nr * sizeof(double));
  cudaMemcpyHelper(dccoords_y, ccoords_y, nr * sizeof(double), H2D);

  int * nr_l = new int[l_target+1];
  int * nc_l = new int[l_target+1];

  int ** cirow_l = new int*[l_target+1];
  int ** cicol_l = new int*[l_target+1];

  double ** ccoords_y_l = new double*[l_target+1];
  double ** ccoords_x_l = new double*[l_target+1];

  int ** dcirow_l = new int*[l_target+1];
  int ** dcicol_l = new int*[l_target+1];

  double ** dccoords_y_l = new double*[l_target+1];
  double ** dccoords_x_l = new double*[l_target+1];

  for (int l = 0; l < l_target+1; l++) {
    int stride = std::pow(2, l);
    nr_l[l] = ceil((float)nr/std::pow(2, l));
    nc_l[l] = ceil((float)nc/std::pow(2, l));
    cirow_l[l] = new int[nr_l[l]];
    cicol_l[l] = new int[nc_l[l]];

    ccoords_y_l[l] = new double[nr_l[l]];
    ccoords_x_l[l] = new double[nc_l[l]];

    for (int i = 0; i < nr_l[l]; i++) {
      cirow_l[l][i] = i;
      ccoords_y_l[l][i] = ccoords_y[i * stride];
    }

    for (int i = 0; i < nc_l[l]; i++) {
      cicol_l[l][i] = i;
      ccoords_x_l[l][i] = ccoords_x[i * stride];
    }

    cudaMallocHelper((void**)&(dcirow_l[l]), nr_l[l] * sizeof(int));
    cudaMemcpyHelper(dcirow_l[l], cirow_l[l], nr_l[l] * sizeof(int), H2D);

    cudaMallocHelper((void**)&(dcicol_l[l]), nc_l[l] * sizeof(int));
    cudaMemcpyHelper(dcicol_l[l], cicol_l[l], nc_l[l] * sizeof(int), H2D);

    cudaMallocHelper((void**)&(dccoords_y_l[l]), nr_l[l] * sizeof(double));
    cudaMemcpyHelper(dccoords_y_l[l], ccoords_y_l[l], nr_l[l] * sizeof(double), H2D);

    cudaMallocHelper((void**)&(dccoords_x_l[l]), nc_l[l] * sizeof(double));
    cudaMemcpyHelper(dccoords_x_l[l], ccoords_x_l[l], nc_l[l] * sizeof(double), H2D);
  }





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
                       dcv,   lddcv);
  org_to_pow2p1_time = ret.time;



  for (int l = 0; l < l_target; ++l) {
    std::cout << "l = " << l << std::endl;
    int stride = std::pow(2, l); // current stride
    int Cstride = stride * 2;    // coarser stride

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
                                     dcwork,     lddcwork);
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
                     dccoords_x_l[l], dccoords_y_l[l]);
    pi_Ql_cuda_time += ret.time;

    // std::cout << "after dcwork = \n";
    // print_matrix_cuda(nr_l[l], nc_l[l], dcwork, lddcwork);

    row_stride = stride;
    col_stride = stride;
    ret = cpt_to_pow2p1(nr,         nc, 
                                     row_stride, col_stride,
                                     dcwork,     lddcwork,
                                     dcv,        lddcv);
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
                            dwork,      lddwork);
    copy_level_l_cuda_time += ret.time;

    // assign_num_level_l(l + 1, work.data(), 0.0, nr, nc, nrow, ncol);

    row_stride = Cstride;
    col_stride = Cstride;
    ret = pow2p1_to_cpt(nr,    nc,
                                     row_stride, col_stride,
                                     dwork,      lddwork,
                                     dcwork,     lddcwork);
    pow2p1_to_cpt_time += ret.time;
    row_stride = 1;
    col_stride = 1;
    ret = assign_num_level_l_cuda(nr_l[l+1],         nc_l[l+1], 
                                  nr_l[l+1],         nc_l[l+1], 
                                  row_stride, col_stride,
                                  dcirow_l[l+1],     dcicol_l[l+1],
                                  dcwork,      lddcwork, 
                                  0.0);
    assign_num_level_l_cuda_time += ret.time;

    row_stride = Cstride;
    col_stride = Cstride;
    ret = cpt_to_pow2p1(nr,         nc, 
                                     row_stride, col_stride,
                                     dcwork,     lddcwork,
                                     dwork,      lddwork);
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
                                     dcwork,     lddcwork);
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
 
    ret = mass_mult_l_row_cuda_sm_pf(nr_l[0],    nc_l[l],
                                  nr_l[0],    nc_l[l],
                                  row_stride, col_stride,
                                  dcirow_l[0], dcicol_l[l],
                                  dcwork,     lddcwork,
                                  dccoords_x_l[l],
                                  16, 16);

    // print_matrix_cuda(nr,    nc, dcwork2,     lddcwork2);
 
    // ret = mass_mult_l_row_cuda(nr_l[0],    nc_l[l],
    //                            nr_l[0],    nc_l[l],
    //                            row_stride, col_stride,
    //                            dcirow_l[0], dcicol_l[l],
    //                            dcwork,     lddcwork,
    //                            dccoords_x_l[l]);
    mass_mult_l_row_cuda_time += ret.time;

    // compare_matrix_cuda(nr,    nc,
    //                     dcwork,     lddcwork,
    //                     dcwork2,     lddcwork2);

    // print_matrix_cuda(nr,    nc, dcwork,     lddcwork);

    double data_size = nr_l[0] * nc_l[l] * sizeof(double);
    double mem_throughput = (data_size/ret.time)/10e9;
    std::cout << "mass_mult_l_row_cuda_mem_throughput (" << nr_l[0] << ", " << nc_l[l] << "): " << mem_throughput << "GB/s. \n";

    row_stride = 1;
    col_stride = stride;
    ret = cpt_to_pow2p1(nr,         nc, 
                                     row_stride, col_stride,
                                     dcwork,     lddcwork,
                                     dwork,      lddwork);
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
                                     dcwork,     lddcwork);
    pow2p1_to_cpt_time += ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = restriction_l_row_cuda(nr_l[0],    nc_l[l],
                                 nr_l[0],    nc_l[l],
                                 row_stride, col_stride,
                                 dcirow_l[0], dcicol_l[l],
                                 dcwork,      lddcwork,
                                 dccoords_x_l[l]);
    restriction_l_row_cuda_time += ret.time;

    row_stride = 1;
    col_stride = stride;
    ret = cpt_to_pow2p1(nr,         nc, 
                                     row_stride, col_stride,
                                     dcwork,     lddcwork,
                                     dwork,      lddwork);
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
                                     dcwork,     lddcwork);
    pow2p1_to_cpt_time += ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = solve_tridiag_M_l_row_cuda(nr_l[0],    nc_l[l+1],
                                     nr_l[0],         nc_l[l+1],
                                     row_stride, col_stride,
                                     dcirow_l[0],     dcicol_l[l+1],
                                     dcwork,     lddcwork,
                                     dccoords_x_l[l+1]);
    solve_tridiag_M_l_row_cuda_time += ret.time;

    row_stride = 1;
    col_stride = Cstride;
    ret = cpt_to_pow2p1(nr,         nc, 
                                     row_stride, col_stride,
                                     dcwork,     lddcwork,
                                     dwork,      lddwork);
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
                                       dcwork,     lddcwork);
      pow2p1_to_cpt_time += ret.time;


      row_stride = 1;
      col_stride = 1;
      ret = mass_mult_l_col_cuda(nr_l[l],     nc_l[l+1],
                                 nr_l[l],     nc_l[l+1],
                                 row_stride,  col_stride,
                                 dcirow_l[l], dcicol_l[l+1],
                                 dcwork,     lddcwork,
                                 dccoords_y_l[l]);
      mass_mult_l_col_cuda_time += ret.time;

      row_stride = stride;
      col_stride = Cstride;
      ret = cpt_to_pow2p1(nr,         nc, 
                                       row_stride, col_stride,
                                       dcwork,     lddcwork,
                                       dwork,      lddwork);
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
                                       dcwork,     lddcwork);
      pow2p1_to_cpt_time += ret.time;

      row_stride = 1;
      col_stride = 1;
      ret = restriction_l_col_cuda(nr_l[l],         nc_l[l+1],
                                   nr_l[l],         nc_l[l+1],
                                   row_stride, col_stride,
                                   dcirow_l[l],     dcicol_l[l+1],
                                   dcwork, lddcwork,
                                   dccoords_y_l[l]);
      restriction_l_col_cuda_time += ret.time;

      row_stride = stride;
      col_stride = Cstride;
      ret = cpt_to_pow2p1(nr,         nc, 
                                       row_stride, col_stride,
                                       dcwork,     lddcwork,
                                       dwork,      lddwork);
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
                                       dcwork,     lddcwork);
      pow2p1_to_cpt_time += ret.time;
      

      row_stride = 1;
      col_stride = 1;
      ret = solve_tridiag_M_l_col_cuda(nr_l[l+1],         nc_l[l+1],
                                       nr_l[l+1],         nc_l[l+1],
                                       row_stride, col_stride,
                                       dcirow_l[l+1],     dcicol_l[l+1],
                                       dcwork, lddcwork,
                                       dccoords_y_l[l+1]);
      solve_tridiag_M_l_col_cuda_time += ret.time;

      row_stride = Cstride;
      col_stride = Cstride;
      ret = cpt_to_pow2p1(nr,         nc, 
                                       row_stride, col_stride,
                                       dcwork,     lddcwork,
                                       dwork,      lddwork);
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
                           dwork,      lddwork);
    add_level_cuda_time += ret.time;
  }

  ret = pow2p1_to_org(nrow,  ncol,
                               nr,    nc,
                               dirow, dicol,
                               dcv,   lddcv,
                               dv,    lddv);
  pow2p1_to_org_time = ret.time;


  




  std::ofstream timing_results;
  timing_results.open ("refactor_2D_cuda_cpt_l2_sm_pf.csv");
  timing_results << "org_to_pow2p1_time," << org_to_pow2p1_time << std::endl;
  timing_results << "pow2p1_to_org_time," << pow2p1_to_org_time << std::endl;

  timing_results << "pow2p1_to_cpt_time," << pow2p1_to_cpt_time << std::endl;
  timing_results << "cpt_to_pow2p1_time," << cpt_to_pow2p1_time << std::endl;

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
}
}
