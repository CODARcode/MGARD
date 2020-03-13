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
refactor_2D_cuda_compact_l2_sm(const int l_target,
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

  // int * cirow = new int[nr];
  // int * cicol = new int[nc];

  // for (int i = 0; i < nr; i++) {
  //   cirow[i] = i;
  // }

  // for (int i = 0; i < nc; i++) {
  //   cicol[i] = i;
  // }

  // int * dcirow;
  // cudaMallocHelper((void**)&dcirow, nr * sizeof(int));
  // cudaMemcpyHelper(dcirow, cirow, nr * sizeof(int), H2D);

  // int * dcicol;
  // cudaMallocHelper((void**)&dcicol, nc * sizeof(int));
  // cudaMemcpyHelper(dcicol, cicol, nc * sizeof(int), H2D);

  // double * coords_x = new double[ncol];
  // double * coords_y = new double[nrow];
  // cudaMemcpyHelper(coords_x, dcoords_x, ncol * sizeof(double), D2H);
  // cudaMemcpyHelper(coords_y, dcoords_y, nrow * sizeof(double), D2H);


  // int * irow = new int[nr];
  // int * icol = new int[nc];
  // cudaMemcpyHelper(irow, dirow, nr * sizeof(int), D2H);
  // cudaMemcpyHelper(icol, dicol, nc * sizeof(int), D2H);

  // double * ccoords_x = new double[nc];
  // double * ccoords_y = new double[nr];

  // for (int i = 0; i < nc; i++) {
  //   ccoords_x[i] = coords_x[icol[i]];
  // }

  // for (int i = 0; i < nr; i++) {
  //   ccoords_y[i] = coords_y[irow[i]];
  // }

  T * dccoords_x;
  cudaMallocHelper((void**)&dccoords_x, nc * sizeof(T));
  //cudaMemcpyHelper(dccoords_x, ccoords_x, nc * sizeof(double), H2D);

  T * dccoords_y;
  cudaMallocHelper((void**)&dccoords_y, nr * sizeof(T));
  //cudaMemcpyHelper(dccoords_y, ccoords_y, nr * sizeof(double), H2D);

  int * nr_l = new int[l_target+1];
  int * nc_l = new int[l_target+1];

  // int ** cirow_l = new int*[l_target+1];
  // int ** cicol_l = new int*[l_target+1];

  // double ** ccoords_y_l = new double*[l_target+1];
  // double ** ccoords_x_l = new double*[l_target+1];

  // int ** dcirow_l = new int*[l_target+1];
  // int ** dcicol_l = new int*[l_target+1];

  // double ** dccoords_y_l = new double*[l_target+1];
  // double ** dccoords_x_l = new double*[l_target+1];

  org_to_pow2p1(ncol, nc, dicol, 
                dcoords_x, dccoords_x,
                B, handle, 0, profile);
  org_to_pow2p1(nrow, nr, dirow, 
                dcoords_y, dccoords_y,
                B, handle, 0, profile);

  T ** ddist_y_l = new T*[l_target+1];
  T ** ddist_x_l = new T*[l_target+1];

  for (int l = 0; l < l_target+1; l++) {
    int stride = std::pow(2, l);
    nr_l[l] = ceil((float)nr/std::pow(2, l));
    nc_l[l] = ceil((float)nc/std::pow(2, l));
    // cirow_l[l] = new int[nr_l[l]];
    // cicol_l[l] = new int[nc_l[l]];

    // ccoords_y_l[l] = new double[nr_l[l]];
    // ccoords_x_l[l] = new double[nc_l[l]];

    // for (int i = 0; i < nr_l[l]; i++) {
    //   cirow_l[l][i] = i;
    //   ccoords_y_l[l][i] = ccoords_y[i * stride];
    // }

    // for (int i = 0; i < nc_l[l]; i++) {
    //   cicol_l[l][i] = i;
    //   ccoords_x_l[l][i] = ccoords_x[i * stride];
    // }



    // cudaMallocHelper((void**)&(dcirow_l[l]), nr_l[l] * sizeof(int));
    // cudaMemcpyHelper(dcirow_l[l], cirow_l[l], nr_l[l] * sizeof(int), H2D);

    // cudaMallocHelper((void**)&(dcicol_l[l]), nc_l[l] * sizeof(int));
    // cudaMemcpyHelper(dcicol_l[l], cicol_l[l], nc_l[l] * sizeof(int), H2D);

    // cudaMallocHelper((void**)&(dccoords_y_l[l]), nr_l[l] * sizeof(double));
    // cudaMemcpyHelper(dccoords_y_l[l], ccoords_y_l[l], nr_l[l] * sizeof(double), H2D);

    // cudaMallocHelper((void**)&(dccoords_x_l[l]), nc_l[l] * sizeof(double));
    // cudaMemcpyHelper(dccoords_x_l[l], ccoords_x_l[l], nc_l[l] * sizeof(double), H2D);

    cudaMallocHelper((void**)&ddist_x_l[l], nc_l[l] * sizeof(T));
    calc_cpt_dist(nc, stride, dccoords_x, ddist_x_l[l],
                  B, handle, 0, profile);

    cudaMallocHelper((void**)&ddist_y_l[l], nr_l[l] * sizeof(T));
    calc_cpt_dist(nr, stride, dccoords_y, ddist_y_l[l],
                  B, handle, 0, profile);
  }





  mgard_cuda_ret ret;
  double total_time = 0.0;
  std::ofstream timing_results;
  if (profile) {
    timing_results.open ("refactor_2D_cuda_cpt_l2_sm.csv");
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
  org_to_pow2p1_time += ret.time;



  for (int l = 0; l < l_target; ++l) {
    //if (verb) std::cout << "l = " << l << std::endl;
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
     
    ret = pow2p1_to_cpt(nr,         nc,
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

    // double * work = new double[nr_l[l] * nc_l[l]];
    // for (int i = 0; i < nr_l[l]; i++) {
    //   for (int j = 0; j < nc_l[l]; j++) {
    //     work[i * nc_l[l] + j] = i * nc_l[l] + j;
    //   }
    // }

    // lddcwork = nc_l[l];
    // lddcwork2 = nc_l[l];
    // cudaMemcpy2DHelper(dcwork, lddcwork * sizeof(double),
    //                    work,  nc_l[l] * sizeof(double),
    //                    nc_l[l] * sizeof(double), nr_l[l], H2D);

    // if (l == 6) print_matrix_cuda(nr_l[l],    nc_l[l], dcwork,     lddcwork);
    // cudaMemcpy2DHelper(dcwork2, lddcwork2 * sizeof(double),
    //                    dcwork,  lddcwork * sizeof(double),
    //                    nc_l[l] * sizeof(double), nr_l[l], D2D);



    row_stride = 1;
    col_stride = 1;
    // ret = pi_Ql_cuda(nr_l[l],         nc_l[l], 
    //                  nr_l[l],         nc_l[l], 
    //                  row_stride,      col_stride,
    //                  dcirow_l[l],     dcicol_l[l],
    //                  dcwork,          lddcwork,
    //                  dccoords_x_l[l], dccoords_y_l[l]);

    // if (l == 6) print_matrix_cuda(nr_l[l],    nc_l[l], dcwork,     lddcwork);
    

    ret = pi_Ql_cuda_sm(nr_l[l],         nc_l[l], 
                        row_stride,      col_stride,
                        dcwork,          lddcwork,
                        ddist_x_l[l],    ddist_y_l[l], 
                        B,
                        handle, 0, profile);
    

    // if (l == 6) print_matrix_cuda(nr_l[l],    nc_l[l], dcwork2,     lddcwork2);

    // compare_matrix_cuda(nr_l[l],    nc_l[l],
    //                     dcwork,     lddcwork,
    //                     dcwork2,     lddcwork2);

    pi_Ql_cuda_time = ret.time;
    // double data_size = nr_l[l] * nc_l[l] * sizeof(double);
    // double mem_throughput = (data_size/ret.time)/1e9;
    // if (verb) std::cout << "pi_Ql_cuda_mem_throughput (" << nr_l[l] << ", " << nc_l[l] << "): " << mem_throughput << "GB/s. \n";

    // std::cout << "after dcwork = \n";
    // print_matrix_cuda(nr_l[l], nc_l[l], dcwork, lddcwork);

    row_stride = stride;
    col_stride = stride;
    ret = cpt_to_pow2p1(nr,         nc, 
                        row_stride, col_stride,
                        dcwork,     lddcwork,
                        dcv,        lddcv,  B,
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
    // ret = copy_level_l_cuda(nr,         nc,
    //                         nr,         nc,
    //                         row_stride, col_stride,
    //                         dcirow,     dcicol,
    //                         dcv,        lddcv, 
    //                         dwork,      lddwork);
    ret = copy_level_l_cuda_l2_sm(nr,         nc,
                                  row_stride, col_stride,
                                  dcv,        lddcv, 
                                  dwork,      lddwork, B,
                                  handle, 0, profile);
    copy_level_l_cuda_time = ret.time;
    // data_size = nr_l[l] * nc_l[l] * sizeof(double);
    // mem_throughput = (data_size/ret.time)/1e9;
    // if (verb) std::cout << "copy_level_l_cuda_mem_throughput (" << nr_l[l] << ", " << nc_l[l] << "): " << mem_throughput << "GB/s. \n";


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
    // ret = assign_num_level_l_cuda(nr_l[l+1],         nc_l[l+1], 
    //                               nr_l[l+1],         nc_l[l+1], 
    //                               row_stride, col_stride,
    //                               dcirow_l[l+1],     dcicol_l[l+1],
    //                               dcwork,      lddcwork, 
    //                               0.0);
    ret = assign_num_level_l_cuda_l2_sm(nr_l[l+1],  nc_l[l+1], 
                                        row_stride, col_stride,
                                        dcwork,      lddcwork, 
                                        (T)0.0, B,
                                        handle, 0, profile);
    assign_num_level_l_cuda_time = ret.time;
    // data_size = nr_l[l+1] * nc_l[l+1] * sizeof(double);
    // mem_throughput = (data_size/ret.time)/1e9;
    // if (verb) std::cout << "assign_num_level_l_cuda_cuda_mem_throughput (" << nr_l[l+1] << ", " << nc_l[l+1] << "): " << mem_throughput << "GB/s. \n";

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

    row_stride = stride;//1;
    col_stride = stride;
    ret = pow2p1_to_cpt(nr,    nc,
                        row_stride, col_stride,
                        dwork,      lddwork,
                        dcwork,     lddcwork, B,
                        handle, 0, profile);
    pow2p1_to_cpt_time += ret.time;


    // double * work = new double[nr_l[0] * nc_l[l]];
    // for (int i = 0; i < nr_l[0]; i++) {
    //   for (int j = 0; j < nc_l[l]; j++) {
    //     work[i * nc_l[l] + j] = i * nc_l[l] + j;
    //   }
    // }

    // cudaMemcpy2DHelper(dcwork, lddcwork * sizeof(double),
    //                    work,  nc_l[l] * sizeof(double),
    //                    nc_l[l] * sizeof(double), nr_l[0], H2D);

    // print_matrix_cuda(nr_l[0],    nc_l[l], dcwork,     lddcwork);
    // cudaMemcpy2DHelper(dcwork2, lddcwork2 * sizeof(double),
    //                    dcwork,  lddcwork * sizeof(double),
    //                    nc_l[l] * sizeof(double), nr_l[0], D2D);
    row_stride = 1;
    col_stride = 1;
 
    ret = mass_mult_l_row_cuda_sm(nr_l[l],    nc_l[l],
                                  row_stride, col_stride,
                                  dcwork,     lddcwork,
                                  ddist_x_l[l],
                                  B, B,
                                  handle, 0, profile);

    // print_matrix_cuda(nr_l[0],    nc_l[l], dcwork,     lddcwork);
 
    // ret = mass_mult_l_row_cuda(nr_l[0],    nc_l[l],
    //                            nr_l[0],    nc_l[l],
    //                            row_stride, col_stride,
    //                            dcirow_l[0], dcicol_l[l],
    //                            dcwork2,     lddcwork2,
    //                            dccoords_x_l[l]);
    mass_mult_l_row_cuda_time = ret.time;

    // compare_matrix_cuda(nr,    nc,
    //                     dcwork,     lddcwork,
    //                     dcwork2,     lddcwork2);

    // print_matrix_cuda(nr_l[0],    nc_l[l], dcwork2,     lddcwork2);

    // data_size = nr_l[0] * nc_l[l] * sizeof(double);
    // mem_throughput = (data_size/ret.time)/1e9;
    // if (verb) std::cout << "mass_mult_l_row_cuda_mem_throughput (" << nr_l[0] << ", " << nc_l[l] << "): " << mem_throughput << "GB/s. \n";

    // row_stride = stride;//1;
    // col_stride = stride;
    // ret = cpt_to_pow2p1(nr,         nc, 
    //                     row_stride, col_stride,
    //                     dcwork,     lddcwork,
    //                     dwork,      lddwork);
    // cpt_to_pow2p1_time += ret.time;



    // row_stride = stride;//1;
    // col_stride = stride;
    // ret = pow2p1_to_cpt(nr,         nc,
    //                     row_stride, col_stride,
    //                     dwork,      lddwork,
    //                     dcwork,     lddcwork);
    // pow2p1_to_cpt_time += ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = restriction_l_row_cuda_sm(nr_l[l],     nc_l[l],
                                    row_stride,  col_stride,
                                    dcwork,      lddcwork,
                                    ddist_x_l[l],
  				                           B, B,
                                    handle, 0, profile);
    restriction_l_row_cuda_time = ret.time;

    // data_size = nr_l[0] * nc_l[l] * sizeof(double);
    // mem_throughput = (data_size/ret.time)/1e9;
    // if (verb) std::cout << "restriction_l_row_cuda_mem_throughput (" << nr_l[0] << ", " << nc_l[l] << "): " << mem_throughput << "GB/s. \n";


    // row_stride = stride;//1;
    // col_stride = stride;
    // ret = cpt_to_pow2p1(nr,         nc, 
    //                     row_stride, col_stride,
    //                     dcwork,     lddcwork,
    //                     dwork,      lddwork);
    // cpt_to_pow2p1_time += ret.time;

  
    // row_stride = stride; //1;
    // col_stride = stride;//Cstride;
    // ret = pow2p1_to_cpt(nr,    nc,
    //                     row_stride, col_stride,
    //                     dwork,      lddwork,
    //                     dcwork,     lddcwork);
    // pow2p1_to_cpt_time += ret.time;

    row_stride = 1;
    col_stride = 2;//1;
    // ret = solve_tridiag_M_l_row_cuda(nr_l[0],    nc_l[l],
    //                                  nr_l[0],         nc_l[l],
    //                                  row_stride, col_stride,
    //                                  dcirow_l[0],     dcicol_l[l],
    //                                  dcwork,     lddcwork,
    //                                  dccoords_x_l[l]);
    // print_matrix_cuda(nr_l[0],    nc_l[l+1], dcwork,     lddcwork);
    ret = solve_tridiag_M_l_row_cuda_sm(nr_l[l],    nc_l[l],
                                        row_stride, col_stride,
                                        dcwork,     lddcwork,
                                        ddist_x_l[l+1],
                                        B, B,
                                        handle, 0, profile);
    // print_matrix_cuda(nr_l[0],    nc_l[l+1], dcwork2,     lddcwork2);
    solve_tridiag_M_l_row_cuda_time = ret.time;

    // data_size = nr_l[0] * nc_l[l+1] * sizeof(double);
    // mem_throughput = (data_size/ret.time)/1e9;
    // if (verb) std::cout << "solve_tridiag_M_l_row_cuda_mem_throughput (" << nr_l[0] << ", " << nc_l[l+1] << "): " << mem_throughput << "GB/s. \n";


    // row_stride = stride;//1;
    // col_stride = stride;//Cstride;
    // ret = cpt_to_pow2p1(nr,         nc, 
    //                     row_stride, col_stride,
    //                     dcwork,     lddcwork,
    //                     dwork,      lddwork);
    // cpt_to_pow2p1_time += ret.time;



    if (nrow > 1) // do this if we have an 2-dimensional array
    {

      // row_stride = stride;
      // col_stride = stride;//Cstride;
      // ret = pow2p1_to_cpt(nr,         nc,
      //                     row_stride, col_stride,
      //                     dwork,      lddwork,
      //                     dcwork,     lddcwork);
      // pow2p1_to_cpt_time += ret.time;

      // double * work = new double[nr_l[l] * nc_l[l+1]];
      // for (int i = 0; i < nr_l[l]; i++) {
      //   for (int j = 0; j < nc_l[l+1]; j++) {
      //     work[i * nc_l[l+1] + j] = i * nc_l[l+1] + j;
      //   }
      // }

      // lddcwork = nc_l[l+1];
      // lddcwork2 = nc_l[l+1];
      // cudaMemcpy2DHelper(dcwork, lddcwork * sizeof(double),
      //                    work,  nc_l[l+1] * sizeof(double),
      //                    nc_l[l+1] * sizeof(double), nr_l[l], H2D);

      // print_matrix_cuda(nr_l[l],    nc_l[l+1], dcwork,     lddcwork);
      // cudaMemcpy2DHelper(dcwork2, lddcwork2 * sizeof(double),
      //                    dcwork,  lddcwork * sizeof(double),
      //                    nc_l[l+1] * sizeof(double), nr_l[l], D2D);


      row_stride = 1;
      col_stride = 2;//1;
      // ret = mass_mult_l_col_cuda(nr_l[l],     nc_l[l+1],
      //                            nr_l[l],     nc_l[l+1],
      //                            row_stride,  col_stride,
      //                            dcirow_l[l], dcicol_l[l+1],
      //                            dcwork,     lddcwork,
      //                            dccoords_y_l[l]);
      // print_matrix_cuda(nr_l[l],    nc_l[l+1], dcwork,     lddcwork);
      ret = mass_mult_l_col_cuda_sm(nr_l[l],     nc_l[l],
                                   row_stride,  col_stride,
                                   dcwork,     lddcwork,
                                   ddist_y_l[l],
                                   B, B,
                                   handle, 0, profile);
      // print_matrix_cuda(nr_l[l],    nc_l[l+1], dcwork2,     lddcwork2);
      mass_mult_l_col_cuda_time = ret.time;

      // data_size = nr_l[l] * nc_l[l+1] * sizeof(double);
      // mem_throughput = (data_size/ret.time)/1e9;
      // if (verb) std::cout << "mass_mult_l_col_cuda_mem_throughput (" 
      // << nr_l[l] << ", " << nc_l[l+1] << "): " << mem_throughput << "GB/s. \n";



      // row_stride = stride;
      // col_stride = stride;//Cstride;
      // ret = cpt_to_pow2p1(nr,         nc, 
      //                     row_stride, col_stride,
      //                     dcwork,     lddcwork,
      //                     dwork,      lddwork);
      // cpt_to_pow2p1_time += ret.time;

      // row_stride = stride;
      // col_stride = stride;//Cstride;
      // ret = pow2p1_to_cpt(nr,         nc,
      //                     row_stride, col_stride,
      //                     dwork,      lddwork,
      //                     dcwork,     lddcwork);
      // pow2p1_to_cpt_time += ret.time;

      // double * work = new double[nr_l[l] * nc_l[l+1]];
      // for (int i = 0; i < nr_l[l]; i++) {
      //   for (int j = 0; j < nc_l[l+1]; j++) {
      //     work[i * nc_l[l+1] + j] = i * nc_l[l+1] + j;
      //   }
      // }

      // lddcwork = nc_l[l+1];
      // lddcwork2 = nc_l[l+1];
      // cudaMemcpy2DHelper(dcwork, lddcwork * sizeof(double),
      //                    work,  nc_l[l+1] * sizeof(double),
      //                    nc_l[l+1] * sizeof(double), nr_l[l], H2D);

      // print_matrix_cuda(nr_l[l],    nc_l[l+1], dcwork,     lddcwork);
      // cudaMemcpy2DHelper(dcwork2, lddcwork2 * sizeof(double),
      //                    dcwork,  lddcwork * sizeof(double),
      //                    nc_l[l+1] * sizeof(double), nr_l[l], D2D);

      row_stride = 1;
      col_stride = 2;//1;
      // ret = restriction_l_col_cuda(nr_l[l],         nc_l[l+1],
      //                              nr_l[l],         nc_l[l+1],
      //                              row_stride, col_stride,
      //                              dcirow_l[l],     dcicol_l[l+1],
      //                              dcwork, lddcwork,
      //                              dccoords_y_l[l]);
      // print_matrix_cuda(nr_l[l],    nc_l[l+1], dcwork,     lddcwork);
      ret = restriction_l_col_cuda_sm(nr_l[l],         nc_l[l],
                                      row_stride, col_stride,
                                      dcwork, lddcwork,
                                      ddist_y_l[l],
                                      B, B,
                                      handle, 0, profile);
      // print_matrix_cuda(nr_l[l],    nc_l[l+1], dcwork2,     lddcwork2);
      restriction_l_col_cuda_time = ret.time;
      // data_size = nr_l[l] * nc_l[l+1] * sizeof(double);
      // mem_throughput = (data_size/ret.time)/1e9;
      // if (verb) std::cout << "restriction_l_col_cuda_mem_throughput (" 
      // << nr_l[l] << ", " << nc_l[l+1] << "): " << mem_throughput << "GB/s. \n";

      // row_stride = stride;
      // col_stride = stride;//stride;
      // ret = cpt_to_pow2p1(nr,         nc, 
      //                     row_stride, col_stride,
      //                     dcwork,     lddcwork,
      //                     dwork,      lddwork);
      // cpt_to_pow2p1_time += ret.time;

      // row_stride = stride;//Cstride;
      // col_stride = stride;//Cstride;
      // ret = pow2p1_to_cpt(nr,    nc,
      //                     row_stride, col_stride,
      //                     dwork,      lddwork,
      //                     dcwork,     lddcwork);
      // pow2p1_to_cpt_time += ret.time;
      
      // double * work = new double[nr_l[l] * nc_l[l]];
      // for (int i = 0; i < nr_l[l]; i++) {
      //   for (int j = 0; j < nc_l[l]; j++) {
      //     work[i * nc_l[l] + j] = i * nc_l[l] + j;
      //   }
      // }

      // lddcwork = nc_l[l];
      // lddcwork2 = nc_l[l];
      // cudaMemcpy2DHelper(dcwork, lddcwork * sizeof(double),
      //                    work,  nc_l[l] * sizeof(double),
      //                    nc_l[l] * sizeof(double), nr_l[l], H2D);

      // print_matrix_cuda(nr_l[l],    nc_l[l], dcwork,     lddcwork);
      // cudaMemcpy2DHelper(dcwork2, lddcwork2 * sizeof(double),
      //                    dcwork,  lddcwork * sizeof(double),
      //                    nc_l[l] * sizeof(double), nr_l[l], D2D);

      row_stride = 2;//1;
      col_stride = 2;//1;
      // ret = solve_tridiag_M_l_col_cuda(nr_l[l],     nc_l[l],
      //                                  nr_l[l],     nc_l[l],
      //                                  row_stride,    col_stride,
      //                                  dcirow_l[l], dcicol_l[l],
      //                                  dcwork,        lddcwork,
      //                                  dccoords_y_l[l]);
      // print_matrix_cuda(nr_l[l],    nc_l[l], dcwork,     lddcwork);

      ret = solve_tridiag_M_l_col_cuda_sm(nr_l[l],     nc_l[l],
                                          row_stride,    col_stride,
                                          dcwork,        lddcwork,
                                          ddist_y_l[l+1],
                                          B, B,
                                          handle, 0, profile);
      // print_matrix_cuda(nr_l[l],    nc_l[l], dcwork2,     lddcwork2);
      // compare_matrix_cuda(nr_l[l], nc_l[l],
      //                     dcwork,    lddcwork,
      //                     dcwork2,   lddcwork2);
      solve_tridiag_M_l_col_cuda_time = ret.time;

      // data_size = nr_l[l+1] * nc_l[l+1] * sizeof(double);
      // mem_throughput = (data_size/ret.time)/1e9;
      // if (verb) std::cout << "solve_tridiag_M_l_col_cuda_throughput (" 
      // << nr_l[l+1] << ", " << nc_l[l+1] << "): " << mem_throughput << "GB/s. \n";

      row_stride = stride;//Cstride;
      col_stride = stride;//Cstride;
      ret = cpt_to_pow2p1(nr,         nc, 
                          row_stride, col_stride,
                          dcwork,     lddcwork,
                          dwork,      lddwork,  B,
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
    // ret = add_level_l_cuda(nr,       nc,
    //                        nr,         nc,
    //                        row_stride, col_stride,
    //                        dcirow,     dcicol,
    //                        dcv,        lddcv, 
    //                        dwork,      lddwork);
    ret = add_level_l_cuda_l2_sm(nr,         nc,
                                 row_stride, col_stride,
                                 dcv,        lddcv, 
                                 dwork,      lddwork,  B,
                                 handle, 0, profile);

    add_level_cuda_time = ret.time;

    // data_size = nr_l[l+1] * nc_l[l+1] * sizeof(double);
    // mem_throughput = (data_size/ret.time)/1e9;
    // if (verb) std::cout << "add_level_l_cuda_throughput (" 
    // << nr_l[l+1] << ", " << nc_l[l+1] << "): " << mem_throughput << "GB/s. \n";

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

  } // end of loop

  ret = pow2p1_to_org(nrow,  ncol,
                      nr,    nc,
                      dirow, dicol,
                      dcv,   lddcv,
                      dv,    lddv, B,
                      handle, 0, profile);
  pow2p1_to_org_time += ret.time;



  if (profile) {
    timing_results << 0 << ",org_to_pow2p1_time," << org_to_pow2p1_time << std::endl;
    timing_results << 0 << ",pow2p1_to_org_time," << pow2p1_to_org_time << std::endl;
    timing_results.close();

    total_time += org_to_pow2p1_time;
    total_time += pow2p1_to_org_time;
  }

  for (int l = 0; l < l_target+1; l++) {
    // delete [] cirow_l[l];
    // delete [] cicol_l[l];
    // delete [] ccoords_y_l[l];
    // delete [] ccoords_x_l[l];
    // cudaFreeHelper(dcirow_l[l]);
    // cudaFreeHelper(dcicol_l[l]);
    // cudaFreeHelper(dccoords_y_l[l]);
    // cudaFreeHelper(dccoords_x_l[l]);
    cudaFreeHelper(ddist_x_l[l]);
    cudaFreeHelper(ddist_y_l[l]);
  }

  cudaFreeHelper(dcv);
  cudaFreeHelper(dcwork);
  // cudaFreeHelper(dcirow);
  // cudaFreeHelper(dcicol);

  // delete [] irow;
  // delete [] icol;
  // delete [] cirow;
  // delete [] cicol;
  delete [] nr_l;
  delete [] nc_l;
  // delete [] cirow_l;
  // delete [] cicol_l;
  delete [] ddist_y_l;
  delete [] ddist_x_l;

  return mgard_cuda_ret(0, total_time);
}

template <typename T>
mgard_cuda_ret 
prep_2D_cuda_l2_sm(const int nrow,     const int ncol,
                   const int nr,       const int nc, 
                   int * dirow,        int * dicol,
                   int * dirowP,       int * dicolP,
                   T * dv,        int lddv, 
                   T * dwork,     int lddwork,
                   T * dcoords_x, T * dcoords_y,
                   int B,
                   mgard_cuda_handle & handle, bool profile) {

  T * dcwork;
  size_t dcwork_pitch;
  cudaMallocPitchHelper((void**)&dcwork, &dcwork_pitch, nc * sizeof(T), nr);
  int lddcwork = dcwork_pitch / sizeof(T);

  T * dccoords_x;
  T * dccoords_y;
  cudaMallocHelper((void**)&dccoords_x, nc * sizeof(T));
  cudaMallocHelper((void**)&dccoords_y, nr * sizeof(T));

  org_to_pow2p1(ncol, nc, dicol, 
                dcoords_x, dccoords_x,
                B, handle, 0, profile);
  org_to_pow2p1(nrow, nr, dirow, 
                dcoords_y, dccoords_y,
                B, handle, 0, profile);


  T * ddist_x;
  cudaMallocHelper((void**)&ddist_x, nc * sizeof(T));
  calc_cpt_dist(nc, 1, dccoords_x, ddist_x,
                B, handle, 0, profile);
  T * ddist_y;
  cudaMallocHelper((void**)&ddist_y, nr * sizeof(T));
  calc_cpt_dist(nr, 1, dccoords_y, ddist_y,
                B, handle, 0, profile);


  mgard_cuda_ret ret;
  double total_time = 0.0;
  std::ofstream timing_results;
  if (profile) {
    timing_results.open ("prep_2D_cuda_cpt_l2_sm.csv");
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
                                      dwork,      lddwork, B,
                                      handle, 0, profile);
  copy_level_cuda_time = ret.time;

  ret = assign_num_level_l_cuda(nrow,       ncol,
                                nr,         nc,
                                row_stride, col_stride,
                                dirow,      dicol,
                                dwork,      lddwork, 
                                (T)0.0, B,
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

  ret = restriction_first_row_cuda(nrow,       ncol,
                                   row_stride, dicolP, nc,
                                   dwork,      lddwork,
                                   dcoords_x, B,
                                   handle, 0, profile);
  restriction_first_row_cuda_time = ret.time;


  ret = org_to_pow2p1(nrow,  ncol,
                      nr,    nc,
                      dirow, dicol,
                      dwork,      lddwork,
                      dcwork,      lddcwork, B,
                      handle, 0, profile);
  org_to_pow2p1_time += ret.time;

  ret = solve_tridiag_M_l_row_cuda_sm(nr,    nc,
                                      row_stride, col_stride,
                                      dcwork,     lddcwork,
                                      ddist_x,
                                      B, B,
                                      handle, 0, profile);
  solve_tridiag_M_l_row_cuda_time = ret.time;

  ret = pow2p1_to_org(nrow,  ncol,
                      nr,    nc,
                      dirow, dicol,
                      dcwork,      lddcwork,
                      dwork,      lddwork, B,
                      handle, 0, profile);
  pow2p1_to_org_time += ret.time;

  // ret = solve_tridiag_M_l_row_cuda(nrow,       ncol,
  //                                  nr,         nc,
  //                                  row_stride, col_stride,
  //                                  dirow,      dicol, 
  //                                  dwork,      lddwork, 
  //                                  dcoords_x);
  


  if (nrow > 1)
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

    ret = org_to_pow2p1(nrow,  ncol,
                        nr,    nc,
                        dirow, dicol,
                        dwork,      lddwork,
                        dcwork,      lddcwork, B,
                        handle, 0, profile);
    org_to_pow2p1_time += ret.time;

    ret = solve_tridiag_M_l_col_cuda_sm(nr,    nc,
                                      row_stride, col_stride,
                                      dcwork,     lddcwork,
                                      ddist_y,
                                      B, B,
                                      handle, 0, profile);
    solve_tridiag_M_l_col_cuda_time = ret.time;

    ret = pow2p1_to_org(nrow,  ncol,
                        nr,    nc,
                        dirow, dicol,
                        dcwork,      lddcwork,
                        dwork,      lddwork, B,
                        handle, 0, profile);
    pow2p1_to_org_time += ret.time;




    // ret = solve_tridiag_M_l_col_cuda(nrow,       ncol,
    //                                  nr,         nc,
    //                                  row_stride, col_stride,
    //                                  dirow,      dicol,
    //                                  dwork,      lddwork, 
    //                                  dcoords_y);
    // solve_tridiag_M_l_col_cuda_time = ret.time;


 }

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

  cudaFreeHelper(dcwork);
  cudaFreeHelper(dccoords_x);
  cudaFreeHelper(dccoords_y);
  cudaFreeHelper(ddist_x);
  cudaFreeHelper(ddist_y);

  return mgard_cuda_ret(0, total_time);
}

template <typename T> 
mgard_cuda_ret 
recompose_2D_cuda_l2_sm(const int l_target,
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

  int * dcirow;
  cudaMallocHelper((void**)&dcirow, nr * sizeof(int));
  cudaMemcpyAsyncHelper(dcirow, cirow, nr * sizeof(int), H2D,
                        handle, 0, profile);

  int * dcicol;
  cudaMallocHelper((void**)&dcicol, nc * sizeof(int));
  cudaMemcpyAsyncHelper(dcicol, cicol, nc * sizeof(int), H2D,
                        handle, 0, profile);

  T * coords_x = new T[ncol];
  T * coords_y = new T[nrow];
  cudaMemcpyAsyncHelper(coords_x, dcoords_x, ncol * sizeof(T), D2H,
                        handle, 0, profile);
  cudaMemcpyAsyncHelper(coords_y, dcoords_y, nrow * sizeof(T), D2H,
                        handle, 0, profile);


  int * irow = new int[nr];
  int * icol = new int[nc];
  cudaMemcpyAsyncHelper(irow, dirow, nr * sizeof(int), D2H,
                        handle, 0, profile);
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

  // double * dccoords_x;
  // cudaMallocHelper((void**)&dccoords_x, nc * sizeof(double));
  // cudaMemcpyHelper(dccoords_x, ccoords_x, nc * sizeof(double), H2D);

  // double * dccoords_y;
  // cudaMallocHelper((void**)&dccoords_y, nr * sizeof(double));
  // cudaMemcpyHelper(dccoords_y, ccoords_y, nr * sizeof(double), H2D);

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

  T * dccoords_x;
  T * dccoords_y;
  cudaMallocHelper((void**)&dccoords_x, nc * sizeof(T));
  cudaMallocHelper((void**)&dccoords_y, nr * sizeof(T));



  org_to_pow2p1(ncol, nc, dicol, 
                dcoords_x, dccoords_x,
                B, handle, 0, profile);
  org_to_pow2p1(nrow, nr, dirow, 
                dcoords_y, dccoords_y,
                B, handle, 0, profile);

  T ** ddist_y_l = new T*[l_target+1];
  T ** ddist_x_l = new T*[l_target+1];

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

    cudaMallocHelper((void**)&ddist_x_l[l], nc_l[l] * sizeof(T));
    calc_cpt_dist(nc, stride, dccoords_x, ddist_x_l[l],
                  B, handle, 0, profile);
    // calc_cpt_dist(nc_l[l], 1, dccoords_x_l[l], ddist_x_l[l]);

    cudaMallocHelper((void**)&ddist_y_l[l], nr_l[l] * sizeof(T));
    calc_cpt_dist(nr, stride, dccoords_y, ddist_y_l[l],
                  B, handle, 0, profile);
    // calc_cpt_dist(nr_l[l], 1, dccoords_y_l[l], ddist_y_l[l]);

    // printf("outside dccoords_x:\n");
    // print_matrix_cuda(1, nc, dccoords_x, nc);
    // printf("outside dccoords_y:\n");
    // print_matrix_cuda(1, nr, dccoords_y, nr);

    // printf("outside ddist_x:\n");
    // print_matrix_cuda(1, nc_l[l], ddist_x_l[l], nc_l[l]);
    // printf("outside ddist_y:\n");
    // print_matrix_cuda(1, nr_l[l], ddist_y_l[l], nr_l[l]);
  }

  mgard_cuda_ret ret;
  double total_time = 0.0;
  std::ofstream timing_results;
  if (profile) {
    timing_results.open ("recompose_2D_cuda_cpt_l2_sm.csv");
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

  ret = org_to_pow2p1(nrow,  ncol,
                       nr,    nc,
                       dirow, dicol,
                       dv,    lddv,
                       dcv,   lddcv,
                       B,
                       handle, 0, profile);
  org_to_pow2p1_time += ret.time;

  for (int l = l_target; l > 0; --l) {
    //std::cout << "l = " << l << std::endl;
    int stride = std::pow(2, l); // current stride
    int Pstride = stride / 2;
    int Cstride = stride * 2;

    pow2p1_to_cpt_time = 0.0;
    cpt_to_pow2p1_time = 0.0;

    // copy_level_l(l - 1, v, work.data(), nr, nc, nrow, ncol);
    int row_stride = Pstride;
    int col_stride = Pstride;

    ret = copy_level_l_cuda(nrow,       ncol,
                            nr,         nc,
                            row_stride, col_stride,
                            //dirow,      dicol,
                            //dv,         lddv,
                            dcirow,      dcicol,
                            dcv,         lddcv,
                            dwork,      lddwork, B,
                            handle, 0, profile);
    copy_level_l_cuda_time = ret.time;
    // double data_size = nr * nc * sizeof(double);
    // double mem_throughput = (data_size/ret.time)/1e9;
    // std::cout << "copy_level_l_cuda_mem_throughput (" << nr << ", " << nc << "): " << mem_throughput << "GB/s. \n";

    // assign_num_level_l(l, work.data(), 0.0, nr, nc, nrow, ncol);
    row_stride = stride;
    col_stride = stride;
    ret = pow2p1_to_cpt(nr,    nc,
                        row_stride, col_stride,
                        dwork,      lddwork,
                        dcwork,     lddcwork, B,
                        handle, 0, profile);
    pow2p1_to_cpt_time += ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = assign_num_level_l_cuda(nr_l[l],         nc_l[l],
                                  nr_l[l],         nc_l[l],
                                  row_stride, col_stride,
                                  // dirow,      dicol,
                                  dcirow_l[l],     dcicol_l[l],
                                  dcwork,      lddcwork, 
                                  (T)0.0, B,
                                  handle, 0, profile);
    assign_num_level_l_cuda_time = ret.time;
    // data_size = nr_l[l] * nc_l[l] * sizeof(double);
    // mem_throughput = (data_size/ret.time)/1e9;
    // std::cout << "assign_num_level_l_cuda_cuda_mem_throughput (" << nr_l[l] << ", " << nc_l[l] << "): " << mem_throughput << "GB/s. \n";

    row_stride = stride;
    col_stride = stride;
    ret = cpt_to_pow2p1(nr,         nc, 
                        row_stride, col_stride,
                        dcwork,     lddcwork,
                        dwork,      lddwork, B,
                        handle, 0, profile);
    cpt_to_pow2p1_time += ret.time;


    // ret = assign_num_level_l_cuda(nrow,       ncol,
    //                               nr,         nc,
    //                               row_stride, col_stride,
    //                               // dirow,      dicol,
    //                               dcirow,      dcicol,
    //                               dwork,      lddwork, 
    //                               0.0);
    // assign_num_level_l_cuda_time += ret.time;

    row_stride = 1;
    col_stride = Pstride;
    ret = pow2p1_to_cpt(nr,    nc,
                        row_stride, col_stride,
                        dwork,      lddwork,
                        dcwork,     lddcwork, B,
                        handle, 0, profile);
    pow2p1_to_cpt_time += ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = mass_mult_l_row_cuda_sm(nr_l[0],    nc_l[l-1],
                                  row_stride, col_stride,
                                  dcwork,     lddcwork,
                                  ddist_x_l[l-1],
                                  B, B,
                                  handle, 0, profile);
    mass_mult_l_row_cuda_time = ret.time;

    // data_size = nr_l[0] * nc_l[l-1] * sizeof(double);
    // mem_throughput = (data_size/ret.time)/1e9;
    // std::cout << "mass_mult_l_row_cuda_mem_throughput (" << nr_l[0] << ", " << nc_l[l-1] << "): " << mem_throughput << "GB/s. \n";

    row_stride = 1;
    col_stride = Pstride;
    ret = cpt_to_pow2p1(nr,         nc, 
                        row_stride, col_stride,
                        dcwork,     lddcwork,
                        dwork,      lddwork, B,
                        handle, 0, profile);
    cpt_to_pow2p1_time += ret.time;

    // ret = mass_mult_l_row_cuda(nrow,       ncol,
    //                            nr,         nc,
    //                            row_stride, col_stride,
    //                            //dirow,      dicol,
    //                            dcirow,      dcicol,
    //                            dwork,      lddwork,
    //                            //dcoords_x);
    //                            dccoords_x);
    // mass_mult_l_row_cuda_time += ret.time;


    row_stride = 1;
    col_stride = Pstride;
    ret = cpt_to_pow2p1(nr,         nc, 
                        row_stride, col_stride,
                        dcwork,     lddcwork,
                        dwork,      lddwork, B,
                        handle, 0, profile);
    cpt_to_pow2p1_time += ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = restriction_l_row_cuda_sm(nr_l[0],     nc_l[l-1],
                                    row_stride,  col_stride,
                                    dcwork,      lddcwork,
                                    ddist_x_l[l-1],
                                    B, B,
                                    handle, 0, profile);
    restriction_l_row_cuda_time = ret.time;
    // data_size = nr_l[0] * nc_l[l-1] * sizeof(double);
    // mem_throughput = (data_size/ret.time)/1e9;
    // std::cout << "restriction_l_row_cuda_mem_throughput (" << nr_l[0] << ", " << nc_l[l-1] << "): " << mem_throughput << "GB/s. \n";

    row_stride = 1;
    col_stride = Pstride;
    ret = cpt_to_pow2p1(nr,         nc, 
                        row_stride, col_stride,
                        dcwork,     lddcwork,
                        dwork,      lddwork, B,
                        handle, 0, profile);
    cpt_to_pow2p1_time += ret.time;

    // ret = restriction_l_row_cuda(nrow,       ncol,
    //                              nr,         nc,
    //                              row_stride, col_stride,
    //                              //dirow,      dicol,
    //                              dcirow,      dcicol,
    //                              dwork,      lddwork,
    //                              //dcoords_x);
    //                              dccoords_x);
    // restriction_l_row_cuda_time += ret.time;


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
    ret = solve_tridiag_M_l_row_cuda_sm(nr_l[0],    nc_l[l],
                                        row_stride, col_stride,
                                        dcwork,     lddcwork,
                                        ddist_x_l[l],
                                         B, B,
                                        handle, 0, profile);

    // ret = solve_tridiag_M_l_row_cuda(nrow,       ncol,
    //                                  nr,         nc,
    //                                  row_stride, col_stride,
    //                                  //dirow,      dicol,
    //                                  dcirow,      dcicol,
    //                                  dwork,      lddwork,
    //                                  //dcoords_x);
    //                                  dccoords_x);
    solve_tridiag_M_l_row_cuda_time = ret.time;

    row_stride = 1;
    col_stride = stride;
    ret = cpt_to_pow2p1(nr,         nc, 
                        row_stride, col_stride,
                        dcwork,     lddcwork,
                        dwork,      lddwork, B,
                        handle, 0, profile);
    cpt_to_pow2p1_time += ret.time;



    if (nrow > 1) // check if we have 1-D array..
    {
      row_stride = Pstride;
      col_stride = stride;
      ret = pow2p1_to_cpt(nr,         nc,
                          row_stride, col_stride,
                          dwork,      lddwork,
                          dcwork,     lddcwork, B,
                          handle, 0, profile);
      pow2p1_to_cpt_time += ret.time;

      row_stride = 1;
      col_stride = 1;
      ret = mass_mult_l_col_cuda_sm(nr_l[l-1],     nc_l[l],
                                    row_stride,  col_stride,
                                    dcwork,     lddcwork,
                                    ddist_y_l[l-1],
                                     B, B,
                                    handle, 0, profile);
      mass_mult_l_col_cuda_time = ret.time;
      // data_size = nr_l[l-1] * nc_l[l] * sizeof(double);
      // mem_throughput = (data_size/ret.time)/1e9;
      // std::cout << "mass_mult_l_col_cuda_mem_throughput (" 
      // << nr_l[l-1] << ", " << nc_l[l] << "): " << mem_throughput << "GB/s. \n";

      row_stride = Pstride;
      col_stride = stride;
      ret = cpt_to_pow2p1(nr,         nc, 
                          row_stride, col_stride,
                          dcwork,     lddcwork,
                          dwork,      lddwork, B,
                          handle, 0, profile);
      cpt_to_pow2p1_time += ret.time;


      // ret = mass_mult_l_col_cuda(nrow,       ncol,
      //                            nr,         nc,
      //                            row_stride, col_stride,
      //                            //dirow,      dicol,
      //                            dcirow,      dcicol,
      //                            dwork,      lddwork,
      //                            //dcoords_y);
      //                            dccoords_y);
      // mass_mult_l_col_cuda_time += ret.time;


      row_stride = Pstride;
      col_stride = stride;
      ret = pow2p1_to_cpt(nr,         nc,
                          row_stride, col_stride,
                          dwork,      lddwork,
                          dcwork,     lddcwork, B,
                          handle, 0, profile);
      pow2p1_to_cpt_time += ret.time;

      row_stride = 1;
      col_stride = 1;
      ret = restriction_l_col_cuda_sm(nr_l[l-1],         nc_l[l],
                                      row_stride, col_stride,
                                      dcwork, lddcwork,
                                      ddist_y_l[l-1],
                                       B, B,
                                      handle, 0, profile);
      restriction_l_col_cuda_time = ret.time;
      // data_size = nr_l[l-1] * nc_l[l] * sizeof(double);
      // mem_throughput = (data_size/ret.time)/1e9;
      // std::cout << "restriction_l_col_cuda_mem_throughput (" 
      // << nr_l[l-1] << ", " << nc_l[l] << "): " << mem_throughput << "GB/s. \n";

      row_stride = Pstride;
      col_stride = stride;
      ret = cpt_to_pow2p1(nr,         nc, 
                          row_stride, col_stride,
                          dcwork,     lddcwork,
                          dwork,      lddwork, B,
                          handle, 0, profile);
      cpt_to_pow2p1_time += ret.time;

      // ret = restriction_l_col_cuda(nrow,       ncol,
      //                              nr,         nc,
      //                              row_stride, col_stride,
      //                              //dirow,       dicol,
      //                              dcirow,      dcicol,
      //                              dwork, lddwork,
      //                              //dcoords_y);
      //                              dccoords_y);
      // restriction_l_col_cuda_time += ret.time;

      row_stride = stride;
      col_stride = stride;

      ret = pow2p1_to_cpt(nr,    nc,
                          row_stride, col_stride,
                          dwork,      lddwork,
                          dcwork,     lddcwork, B,
                          handle, 0, profile);
      pow2p1_to_cpt_time += ret.time;

      row_stride = 1;
      col_stride = 1;
      ret = solve_tridiag_M_l_col_cuda_sm(nr_l[l],     nc_l[l],
                                          row_stride,    col_stride,
                                          dcwork,        lddcwork,
                                          ddist_y_l[l],
                                           B, B,
                                          handle, 0, profile);
      // ret = solve_tridiag_M_l_col_cuda(nrow,       ncol,
      //                                  nr,         nc,
      //                                  row_stride, col_stride,
      //                                  //dirow,      dicol,
      //                                  dcirow,      dcicol,
      //                                  dwork,      lddwork,
      //                                  //dcoords_y);
      //                                  dccoords_y);
      solve_tridiag_M_l_col_cuda_time = ret.time;

      row_stride = stride;
      col_stride = stride;
      ret = cpt_to_pow2p1(nr,         nc, 
                          row_stride, col_stride,
                          dcwork,     lddcwork,
                          dwork,      lddwork, B,
                          handle, 0, profile);
      cpt_to_pow2p1_time += ret.time;

    }

    row_stride = stride;
    col_stride = stride;
    ret = subtract_level_l_cuda(nrow,       ncol, 
                          nr,         nc,
                          row_stride, col_stride,
                          //dirow,      dicol,
                          dcirow,      dcicol,
                          dwork,      lddwork,
                          //dv,         lddv);
                          dcv,         lddcv, B,
                          handle, 0, profile);
    subtract_level_l_cuda_time = ret.time;


    // double * dcwork2;
    // size_t dcwork_pitch2;
    // cudaMallocPitchHelper((void**)&dcwork2, &dcwork_pitch2, nc * sizeof(double), nr);
    // int lddcwork2 = dcwork_pitch2 / sizeof(double);

    // double * work = new double[nr * nc];
    // for (int i = 0; i < nr; i++) {
    //   for (int j = 0; j < nc; j++) {
    //     work[i * nc + j] = (i*i+2)*(j*j+2);
    //   }
    // }

    // cudaMemcpy2DHelper(dcwork, lddcwork * sizeof(double),
    //                    work,  nc * sizeof(double),
    //                    nc * sizeof(double), nr, H2D);

    // // print_matrix_cuda(nr,    nc, dcwork,     lddcwork);
    // cudaMemcpy2DHelper(dcwork2, lddcwork2 * sizeof(double),
    //                    dcwork,  lddcwork * sizeof(double),
    //                    nc * sizeof(double), nr, D2D);




    // row_stride = stride;
    // col_stride = stride;
    // printf("row_stride = %d\n", row_stride);
    // printf("col_stride = %d\n", col_stride);
    // ret = prolongate_l_row_cuda(nrow,        ncol, 
    //                        nr,         nc,
    //                        row_stride, col_stride,
    //                        //dirow,      dicol,
    //                        dcirow,      dcicol,
    //                        //dwork,      lddwork,
    //                        dcwork,      lddcwork,
    //                        //dcoords_x);
    //                        dccoords_x);
    // prolongate_l_row_cuda_time += ret.time;
    // print_matrix_cuda(nr, nc, dcwork,      lddcwork);

    // if (nrow > 1) {
    //   row_stride = stride;
    //   col_stride = Pstride;
    //   printf("row_stride = %d\n", row_stride);
    //   printf("col_stride = %d\n", col_stride);
    //   ret = prolongate_l_col_cuda(nrow,        ncol, 
    //                               nr,         nc,
    //                               row_stride, col_stride,
    //                               //dirow,      dicol,
    //                               dcirow,      dcicol,
    //                               //dwork,      lddwork,
    //                               dcwork,      lddcwork,                                  
    //                               //dcoords_y);
    //                               dccoords_y);
    //   prolongate_l_col_cuda_time += ret.time;
    //   // print_matrix_cuda(nr, nc, dcwork,      lddcwork);
    // }


    // row_stride = stride;
    // col_stride = Pstride;
    // ret = prolongate_l_row_cuda_sm(nr,    nc,
    //                                row_stride, col_stride,
    //                                dwork,     lddwork,
    //                                ddist_x_l[l-1],
    //                                2);

    // row_stride = Pstride;
    // col_stride = Pstride;
    // ret = prolongate_l_col_cuda_sm(nr,  nc,
    //                                row_stride, col_stride,
    //                                dwork,     lddwork,
    //                                dccoords_y,
    //                                2);

    row_stride = stride;
    col_stride = Pstride;
    ret = pow2p1_to_cpt(nr,         nc,
                        row_stride, col_stride,
                        dwork,      lddwork,
                        dcwork,     lddcwork, B,
                        handle, 0, profile);
    pow2p1_to_cpt_time += ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = prolongate_l_row_cuda_sm(nr_l[l],    nc_l[l-1],
                                   row_stride, col_stride,
                                   dcwork,     lddcwork,
                                   ddist_x_l[l-1],
                                    B,
                                   handle, 0, profile);
    prolongate_l_row_cuda_time = ret.time;
    // data_size = nr_l[l] * nc_l[l-1] * sizeof(double);
    // mem_throughput = (data_size/ret.time)/1e9;
    // std::cout << "prolongate_l_row_cuda_throughput (" 
    // << nr_l[l] << ", " << nc_l[l-1] << "): " << mem_throughput << "GB/s. \n";

    row_stride = stride;
    col_stride = Pstride;
    ret = cpt_to_pow2p1(nr,         nc, 
                        row_stride, col_stride,
                        dcwork,     lddcwork,
                        dwork,      lddwork, B,
                        handle, 0, profile);
    cpt_to_pow2p1_time += ret.time;


    row_stride = Pstride;
    col_stride = Pstride;
    ret = pow2p1_to_cpt(nr,         nc,
                        row_stride, col_stride,
                        dwork,      lddwork,
                        dcwork,     lddcwork, B,
                        handle, 0, profile);
    pow2p1_to_cpt_time += ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = prolongate_l_col_cuda_sm(nr_l[l-1],  nc_l[l-1],
                                   row_stride, col_stride,
                                   dcwork,     lddcwork,
                                   ddist_y_l[l-1],
                                    B,
                                   handle, 0, profile);
    prolongate_l_col_cuda_time = ret.time;
    // data_size = nr_l[l-1] * nc_l[l-1] * sizeof(double);
    // mem_throughput = (data_size/ret.time)/1e9;
    // std::cout << "prolongate_l_col_cuda_throughput (" 
    // << nr_l[l-1] << ", " << nc_l[l-1] << "): " << mem_throughput << "GB/s. \n";

    row_stride = Pstride;
    col_stride = Pstride;
    ret = cpt_to_pow2p1(nr,         nc, 
                        row_stride, col_stride,
                        dcwork,     lddcwork,
                        dwork,      lddwork, B,
                        handle, 0, profile);
    cpt_to_pow2p1_time += ret.time;

    // print_matrix_cuda(nr, nc, dcwork2,      lddcwork2);

    // compare_matrix_cuda(nr,    nc,
    //                     dcwork,     lddcwork,
    //                     dcwork2,     lddcwork2);

    // assign_num_level_l(l, v, 0.0, nr, nc, nrow, ncol);
    row_stride = stride;
    col_stride = stride;
    ret = assign_num_level_l_cuda(nrow,       ncol,
                                  nr,         nc,
                                  row_stride, col_stride,
                                  //dirow,      dicol,
                                  dcirow,      dcicol,
                                  //dv,         lddv, 
                                  dcv,         lddcv,
                                  (T)0.0,  B,
                                  handle, 0, profile);
    assign_num_level_l_cuda_time2 = ret.time;
    // data_size = nr_l[l] * nc_l[l] * sizeof(double);
    // mem_throughput = (data_size/ret.time)/1e9;
    // std::cout << "assign_num_level_l_cuda_throughput (" 
    // << nr_l[l] << ", " << nc_l[l] << "): " << mem_throughput << "GB/s. \n";

    // subtract_level_l(l - 1, v, work.data(), nr, nc, nrow, ncol);

    row_stride = Pstride;
    col_stride = Pstride;
    ret = subtract_level_l_cuda(nrow,       ncol, 
                                nr,         nc,
                                row_stride, col_stride,
                                //dirow,      dicol,
                                dcirow,      dcicol,
                                //dv,         lddv, 
                                dcv,         lddcv, 
                                dwork,      lddwork, B,
                                handle, 0, profile);
    subtract_level_l_cuda_time2 = ret.time;
    // data_size = nr_l[l-1] * nc_l[l-1] * sizeof(double);
    // mem_throughput = (data_size/ret.time)/1e9;
    // std::cout << "subtract_level_l_cuda_throughput (" 
    // << nr_l[l-1] << ", " << nc_l[l-1] << "): " << mem_throughput << "GB/s. \n";

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

  ret = pow2p1_to_org(nrow,  ncol,
                      nr,    nc,
                      dirow, dicol,
                      dcv,   lddcv,
                      dv,    lddv, B,
                      handle, 0, profile);
  pow2p1_to_org_time += ret.time;

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
  cudaFreeHelper(dcirow);
  cudaFreeHelper(dcicol);
  cudaFreeHelper(dccoords_x);
  cudaFreeHelper(dccoords_y);
  for (int l = 0; l < l_target+1; l++) {
    cudaFreeHelper(dcirow_l[l]);
    cudaFreeHelper(dcicol_l[l]);
    cudaFreeHelper(dccoords_y_l[l]);
    cudaFreeHelper(dccoords_x_l[l]);
    cudaFreeHelper(ddist_x_l[l]);
    cudaFreeHelper(ddist_y_l[l]);
  }


  return mgard_cuda_ret(0, total_time);

}



template <typename T>
mgard_cuda_ret
postp_2D_cuda_l2_sm(const int nrow,     const int ncol,
                    const int nr,       const int nc, 
                    int * dirow,        int * dicol,
                    int * dirowP,       int * dicolP,
                    T * dv,        int lddv, 
                    T * dwork,     int lddwork,
                    T * dcoords_x, T * dcoords_y,
                    int B,
                    mgard_cuda_handle & handle, bool profile) {

  T * dcwork;
  size_t dcwork_pitch;
  cudaMallocPitchHelper((void**)&dcwork, &dcwork_pitch, nc * sizeof(T), nr);
  int lddcwork = dcwork_pitch / sizeof(T);

  T * dccoords_x;
  T * dccoords_y;
  cudaMallocHelper((void**)&dccoords_x, nc * sizeof(T));
  cudaMallocHelper((void**)&dccoords_y, nr * sizeof(T));

  org_to_pow2p1(ncol, nc, dicol, 
                dcoords_x, dccoords_x,
                B, handle, 0, profile);
  org_to_pow2p1(nrow, nr, dirow, 
                dcoords_y, dccoords_y,
                B, handle, 0, profile);


  T * ddist_x;
  cudaMallocHelper((void**)&ddist_x, nc * sizeof(T));
  calc_cpt_dist(nc, 1, dccoords_x, ddist_x,
                B, handle, 0, profile);
  T * ddist_y;
  cudaMallocHelper((void**)&ddist_y, nr * sizeof(T));
  calc_cpt_dist(nr, 1, dccoords_y, ddist_y,
                B, handle, 0, profile);



  mgard_cuda_ret ret;
  double total_time;
  std::ofstream timing_results;
  if (profile) {
    timing_results.open ("postp_2D_cuda_cpt_l2_sm.csv");
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
 // mgard_cannon::copy_level(nrow, ncol, 0, v, work);
  int row_stride = 1;
  int col_stride = 1;
  ret = mgard_cannon::copy_level_cuda(nrow,       ncol, 
                                row_stride, col_stride,
                                dv,         lddv,
                                dwork,      lddwork, B,
                                handle, 0, profile);
  copy_level_cuda_time = ret.time;

  // assign_num_level_l(0, work.data(), 0.0, nr, nc, nrow, ncol);

  row_stride = 1;
  col_stride = 1;
  ret = assign_num_level_l_cuda(nrow,       ncol,
                          nr,         nc,
                          row_stride, col_stride,
                          dirow,      dicol,
                          dwork,      lddwork,
                          (T)0.0, B,
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
  ret = org_to_pow2p1(nrow,  ncol,
                      nr,    nc,
                      dirow, dicol,
                      dwork,      lddwork,
                      dcwork,      lddcwork, B,
                      handle, 0, profile);
  org_to_pow2p1_time += ret.time;
  ret = solve_tridiag_M_l_row_cuda_sm(nr,    nc,
                                      row_stride, col_stride,
                                      dcwork,     lddcwork,
                                      ddist_x,
                                       B, B,
                                      handle, 0, profile);
  solve_tridiag_M_l_row_cuda_time = ret.time;

  ret = pow2p1_to_org(nrow,  ncol,
                      nr,    nc,
                      dirow, dicol,
                      dcwork,      lddcwork,
                      dwork,      lddwork, B,
                      handle, 0, profile);
  pow2p1_to_org_time += ret.time;

  // ret = solve_tridiag_M_l_row_cuda(nrow,       ncol,
  //                            nr,         nc,
  //                            row_stride, col_stride,
  //                            dirow,      dicol,
  //                            dwork,      lddwork,
  //                            dcoords_x);
  // solve_tridiag_M_l_row_cuda_time += ret.time;

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
    ret = org_to_pow2p1(nrow,  ncol,
                        nr,    nc,
                        dirow, dicol,
                        dwork,      lddwork,
                        dcwork,      lddcwork, B,
                        handle, 0, profile);
    org_to_pow2p1_time += ret.time;

    ret = solve_tridiag_M_l_col_cuda_sm(nr,    nc,
                                      row_stride, col_stride,
                                      dcwork,     lddcwork,
                                      ddist_y,
                                       B, B,
                                      handle, 0, profile);
    solve_tridiag_M_l_col_cuda_time = ret.time;

    ret = pow2p1_to_org(nrow,  ncol,
                        nr,    nc,
                        dirow, dicol,
                        dcwork,      lddcwork,
                        dwork,      lddwork, B,
                        handle, 0, profile);
    pow2p1_to_org_time += ret.time;


    // ret = solve_tridiag_M_l_col_cuda(nrow,       ncol,
    //                            nr,         nc,
    //                            row_stride, col_stride,
    //                            dirow,      dicol,
    //                            dwork,      lddwork,
    //                            dcoords_y);
    // solve_tridiag_M_l_col_cuda_time += ret.time;
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
    // print_matrix(nrow, ncol, work.data(), ldwork);
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
  // print_matrix(nrow, ncol, work.data(), ldwork);
  
  ret = assign_num_level_l_cuda(nrow,       ncol,
                          nr,         nc,
                          row_stride, col_stride,
                          dirow,      dicol,
                          dv,         lddv,
                          (T)0.0, B,
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
    timing_results << l << ",pow2p1_to_org_time," << pow2p1_to_org_time << std::endl;

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

  cudaFreeHelper(dcwork);
  cudaFreeHelper(dccoords_x);
  cudaFreeHelper(dccoords_y);
  cudaFreeHelper(ddist_x);
  cudaFreeHelper(ddist_y);

  return mgard_cuda_ret(0, total_time);

}

template mgard_cuda_ret
refactor_2D_cuda_compact_l2_sm<double>(const int l_target,
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
refactor_2D_cuda_compact_l2_sm<float>(const int l_target,
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
prep_2D_cuda_l2_sm<double>(const int nrow,     const int ncol,
                   const int nr,       const int nc, 
                   int * dirow,        int * dicol,
                   int * dirowP,       int * dicolP,
                   double * dv,        int lddv, 
                   double * dwork,     int lddwork,
                   double * dcoords_x, double * dcoords_y,
                   int B,
                   mgard_cuda_handle & handle, bool profile);
template mgard_cuda_ret 
prep_2D_cuda_l2_sm<float>(const int nrow,     const int ncol,
                   const int nr,       const int nc, 
                   int * dirow,        int * dicol,
                   int * dirowP,       int * dicolP,
                   float * dv,        int lddv, 
                   float * dwork,     int lddwork,
                   float * dcoords_x, float * dcoords_y,
                   int B,
                   mgard_cuda_handle & handle, bool profile);

template mgard_cuda_ret 
recompose_2D_cuda_l2_sm<double>(const int l_target,
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
recompose_2D_cuda_l2_sm<float>(const int l_target,
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
postp_2D_cuda_l2_sm<double>(const int nrow,     const int ncol,
                    const int nr,       const int nc, 
                    int * dirow,        int * dicol,
                    int * dirowP,       int * dicolP,
                    double * dv,        int lddv, 
                    double * dwork,     int lddwork,
                    double * dcoords_x, double * dcoords_y,
                    int B,
                    mgard_cuda_handle & handle, bool profile);

template mgard_cuda_ret
postp_2D_cuda_l2_sm<float>(const int nrow,     const int ncol,
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
