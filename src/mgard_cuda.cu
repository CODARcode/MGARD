
#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"
#include "mgard_cuda.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_nuni_2d_cuda_opt.h"
#include "mgard_nuni.h"
#include "mgard.h"
#include <iomanip> 
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>

namespace mgard
{

unsigned char *
refactor_qz_2D_cuda (int nrow, int ncol, const double *u, int &outsize, double tol, int opt)
{

  std::vector<double> row_vec (ncol);
  std::vector<double> col_vec (nrow);
  std::vector<double> v(u, u+nrow*ncol), work(nrow * ncol);

  double norm = mgard_2d::mgard_common::max_norm(v);

  
  if (mgard::is_2kplus1 (nrow)
      && mgard::is_2kplus1 (ncol)) // input is (2^q + 1) x (2^p + 1)
    {
      int nlevel;
      mgard::set_number_of_levels (nrow, ncol, nlevel);
      tol /= float (nlevel + 1);      
      
      int l_target = nlevel - 1;


      double * dv;
      size_t dv_pitch;
      cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
      int lddv = dv_pitch / sizeof(double);
      cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                         v.data(), ncol  * sizeof(double), 
                         ncol * sizeof(double), nrow, 
                         H2D);

      double * dv2;
      size_t dv_pitch2;
      cudaMallocPitchHelper((void**)&dv2, &dv_pitch2, ncol * sizeof(double), nrow);
      int lddv2 = dv_pitch2 / sizeof(double);
      cudaMemcpy2DHelper(dv2, lddv2 * sizeof(double), 
                         v.data(), ncol  * sizeof(double), 
                         ncol * sizeof(double), nrow, 
                         H2D);


      double * dwork;
      size_t dwork_pitch;
      cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, nrow * sizeof(double), ncol);
      int lddwork = dwork_pitch / sizeof(double);
      cudaMemset2DHelper(dwork, dwork_pitch, 0, nrow * sizeof(double), ncol);

      mgard::refactor_cuda (nrow, ncol, l_target, dv, lddv, dwork, lddwork);

      int size_ratio = sizeof (double) / sizeof (int);
      std::vector<int> qv (nrow * ncol + size_ratio);

      int * dqv;
      cudaMallocHelper((void**)&dqv, (nrow * ncol + size_ratio) * sizeof(int));
      int lddqv = ncol;
      cudaMemsetHelper(dqv, 0, (nrow * ncol + size_ratio) * sizeof(int));
      
      quantize_2D_iterleave_cuda (nrow, ncol, dv, lddv, dqv, lddqv, norm, tol);
      cudaMemcpyHelper(qv.data(), dqv, (nrow * ncol + size_ratio) * sizeof(int), D2H);
      cudaFreeHelper(dwork);
      
      std::vector<unsigned char> out_data;

      mgard::compress_memory_z (qv.data (), sizeof (int) * qv.size (), out_data);
      outsize = out_data.size ();
      unsigned char *buffer = (unsigned char *)malloc (outsize);
      std::copy (out_data.begin (), out_data.end (), buffer);
      return buffer;
    }
  else
    {

      std::vector<double> coords_x(ncol), coords_y(nrow);
      
      std::iota(std::begin(coords_x), std::end(coords_x), 0);
      std::iota(std::begin(coords_y), std::end(coords_y), 0);

      double * dv;
      size_t dv_pitch;
      cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
      int lddv = dv_pitch / sizeof(double);
      cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                         v.data(), ncol  * sizeof(double), 
                         ncol * sizeof(double), nrow, 
                         H2D);

      double * dwork;
      size_t dwork_pitch;
      cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
      int lddwork = dwork_pitch / sizeof(double);
      cudaMemset2DHelper(dwork, dwork_pitch, 0, ncol * sizeof(double), nrow);

      double * dcoords_x;
      cudaMallocHelper((void**)&dcoords_x, ncol * sizeof(double));
      cudaMemcpyHelper(dcoords_x, coords_x.data(), ncol * sizeof(double), H2D);

      double * dcoords_y;
      cudaMallocHelper((void**)&dcoords_y, nrow * sizeof(double));
      cudaMemcpyHelper(dcoords_y, coords_y.data(), nrow * sizeof(double), H2D);

      int nlevel_x = std::log2(ncol-1);
      int nc = std::pow(2, nlevel_x ) + 1; //ncol new

      int nlevel_y = std::log2(nrow-1);
      int nr = std::pow(2, nlevel_y ) + 1; //nrow new

      int * irow  = new int[nr];
      int * irowP = new int[nrow-nr];
      int irow_ptr  = 0;
      int irowP_ptr = 0;

      for (int i = 0; i < nr; i++) {
        int irow_r = mgard_2d::mgard_gen::get_lindex_cuda(nr, nrow, i);
        irow[irow_ptr] = irow_r;
        if (irow_ptr > 0 && irow[irow_ptr - 1] != irow[irow_ptr] - 1) {
          irowP[irowP_ptr] = irow[irow_ptr] - 1;
          irowP_ptr ++;
        } 
        irow_ptr++;
      }

      std::cout << "irow: ";
      for (int i = 0; i < nr; i++) std::cout << irow[i] << ", ";
      std::cout << std::endl;

      // std::cout << "irowP: ";
      // for (int i = 0; i < nrow-nr; i++) std::cout << irowP[i] << ", ";
      // std::cout << std::endl;


      int * icol  = new int[nc];
      int * icolP = new int[ncol-nc];
      int icol_ptr  = 0;
      int icolP_ptr = 0;

      for (int i = 0; i < nc; i++) {
        int icol_r = mgard_2d::mgard_gen::get_lindex_cuda(nc, ncol, i);
        icol[icol_ptr] = icol_r;
        if (icol_ptr > 0 && icol[icol_ptr - 1] != icol[icol_ptr] - 1) {
          icolP[icolP_ptr] = icol[icol_ptr] - 1;
          icolP_ptr ++;
        } 
        icol_ptr++;
      }

      // std::cout << "icol: ";
      // for (int i = 0; i < nc; i++) std::cout << icol[i] << ", ";
      // std::cout << std::endl;

      // std::cout << "icolP: ";
      // for (int i = 0; i < ncol-nc; i++) std::cout << icolP[i] << ", ";
      // std::cout << std::endl;

      int * dirow;
      cudaMallocHelper((void**)&dirow, nr * sizeof(int));
      cudaMemcpyHelper(dirow, irow, nr * sizeof(int), H2D);

      int * dicol;
      cudaMallocHelper((void**)&dicol, nc * sizeof(int));
      cudaMemcpyHelper(dicol, icol, nc * sizeof(int), H2D);

      int * dirowP;
      cudaMallocHelper((void**)&dirowP, (nrow-nr) * sizeof(int));
      cudaMemcpyHelper(dirowP, irowP, (nrow-nr) * sizeof(int), H2D);

      int * dicolP;
      cudaMallocHelper((void**)&dicolP, (ncol-nc) * sizeof(int));
      cudaMemcpyHelper(dicolP, icolP, (ncol-nc) * sizeof(int), H2D);

      

      

      int nlevel = std::min(nlevel_x, nlevel_y);
      tol /= nlevel + 1;

      int l_target = nlevel-1;
      //l_target = 0;
      // mgard_2d::mgard_gen::prep_2D(nr, nc, nrow, ncol, l_target, v.data(),  work, coords_x, coords_y, row_vec, col_vec);
      std::cout << "***prep_2D_cuda***" << std::endl;
      mgard_2d::mgard_gen::prep_2D_cuda(nrow,      ncol,
                                        nr,        nc, 
                                        dirow,     dicol,
                                        dirowP,    dicolP,
                                        dv,        lddv,
                                        dwork,     lddwork,
                                        dcoords_x, dcoords_y);

     

       // mgard_2d::mgard_gen::refactor_2D(nr, nc, nrow, ncol, l_target, v.data(),  work, coords_x, coords_y, row_vec, col_vec);


      if (opt == 0) {
        std::cout << "***refactor_2D_cuda***" << std::endl;
        mgard_2d::mgard_gen::refactor_2D_cuda(l_target,
                                              nrow,      ncol,
                                              nr,        nc, 
                                              dirow,     dicol,
                                              dirowP,    dicolP,
                                              dv,        lddv,
                                              dwork,     lddwork,
                                              dcoords_x, dcoords_y);
      } else if (opt == 1) {
        std::cout << "***refactor_2D_cuda_compact_l1***" << std::endl;
        mgard_2d::mgard_gen::refactor_2D_cuda_compact_l1(l_target,
                                                 nrow,      ncol,
                                                 nr,        nc, 
                                                 dirow,     dicol,
                                                 dirowP,    dicolP,
                                                 dv,        lddv,
                                                 dwork,     lddwork,
                                                 dcoords_x, dcoords_y);
        
      } else if (opt == 2) {
        std::cout << "***refactor_2D_cuda_compact_l2***" << std::endl;
        mgard_2d::mgard_gen::refactor_2D_cuda_compact_l2_sm_pf(l_target,
                                                 nrow,      ncol,
                                                 nr,        nc, 
                                                 dirow,     dicol,
                                                 dirowP,    dicolP,
                                                 dv,        lddv,
                                                 dwork,     lddwork,
                                                 dcoords_x, dcoords_y);
        
      }
      //  cudaMemcpy2DHelper(v.data(), ncol  * sizeof(double), 
      //                    dv, lddv * sizeof(double), 
      //                    ncol * sizeof(double), nrow, 
      //                    D2H);

      // cudaMemcpy2DHelper(work.data(), ncol  * sizeof(double), 
      //                    dwork, lddwork * sizeof(double), 
      //                    ncol * sizeof(double), nrow, 
      //                    D2H);

      // work.clear ();
      // col_vec.clear ();
      // row_vec.clear ();

      int size_ratio = sizeof (double) / sizeof (int);
      std::vector<int> qv (nrow * ncol + size_ratio);

      tol /= nlevel + 1;
      // mgard::quantize_2D_interleave (nrow, ncol, v.data (), qv, norm, tol);

      int * dqv;
      cudaMallocHelper((void**)&dqv, (nrow * ncol + size_ratio) * sizeof(int));
      int lddqv = ncol;
      cudaMemsetHelper(dqv, 0, (nrow * ncol + size_ratio) * sizeof(int));

      mgard::quantize_2D_iterleave_cuda (nrow, ncol, dv, lddv, dqv, lddqv, norm, tol);

      cudaMemcpyHelper(qv.data(), dqv, (nrow * ncol + size_ratio) * sizeof(int), D2H);
      cudaFreeHelper(dwork);

      std::vector<unsigned char> out_data;

      mgard::compress_memory_z (qv.data (), sizeof (int) * qv.size (), out_data);

      outsize = out_data.size ();
      unsigned char *buffer = (unsigned char *)malloc (outsize);
      std::copy (out_data.begin (), out_data.end (), buffer);
      return buffer;
    }
}

double* recompose_udq_2D_cuda(int nrow, int ncol, unsigned char *data, int data_len, int opt)
{
  int size_ratio = sizeof(double)/sizeof(int);

  if (mgard::is_2kplus1 (nrow)
    && mgard::is_2kplus1 (ncol)) // input is (2^q + 1) x (2^p + 1)
  {
    int ncol_new = ncol;
    int nrow_new = nrow;
    
    int nlevel_new;
    mgard::set_number_of_levels (nrow_new, ncol_new, nlevel_new);
    int l_target = nlevel_new-1;

    
    std::vector<int> out_data(nrow_new*ncol_new + size_ratio);

    mgard::decompress_memory_z(data, data_len, out_data.data(), out_data.size()*sizeof(int)); // decompress input buffer
    
    double *v = (double *)malloc (nrow_new*ncol_new*sizeof(double));

    int ldv = ncol_new;
    int * dout_data;
    cudaMallocHelper((void**)&dout_data, (nrow_new*ncol_new + size_ratio) * sizeof(int));
    int lddout_data = ncol;
    cudaMemcpyHelper(dout_data, out_data.data(), (nrow_new*ncol_new + size_ratio) * sizeof(int), H2D);

    double * dv;
    size_t dv_pitch;
    cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol_new * sizeof(double), nrow_new);
    int lddv = dv_pitch / sizeof(double);

    dequantize_2D_iterleave_cuda(nrow_new, ncol_new, dv, lddv, dout_data, lddout_data);
   
    double * dwork;
    size_t dwork_pitch;
    cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, nrow * sizeof(double), ncol);
    int lddwork = dwork_pitch / sizeof(double);
    cudaMemset2DHelper(dwork, dwork_pitch, 0, ncol * sizeof(double), nrow);



    recompose_cuda (nrow, ncol, l_target, dv, lddv, dwork, lddwork);
    cudaMemcpy2DHelper(v, ldv*sizeof(double), dv, lddv*sizeof(double), ncol * sizeof(double), nrow, D2H);
    
    return v;

  }
  else
  {
    std::vector<double> coords_x(ncol), coords_y(nrow);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);
    
    int nlevel_x = std::log2(ncol-1);
    int nc = std::pow(2, nlevel_x ) + 1; //ncol new

    int nlevel_y = std::log2(nrow-1);
    int nr = std::pow(2, nlevel_y ) + 1; //nrow new

    int nlevel = std::min(nlevel_x, nlevel_y);

    int l_target = nlevel-1;

    //int l_target = 0;
    std::vector<int> out_data(nrow*ncol + size_ratio);


    mgard::decompress_memory_z(data, data_len, out_data.data(), out_data.size()*sizeof(int)); // decompress input buffer

    double *v = (double *)malloc (nrow*ncol*sizeof(double));

    double * dv;
    size_t dv_pitch;
    cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
    int lddv = dv_pitch / sizeof(double);
    // cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
    //                    v.data(), ncol  * sizeof(double), 
    //                    ncol * sizeof(double), nrow, 
    //                    H2D);

    double * dwork;
    size_t dwork_pitch;
    cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
    int lddwork = dwork_pitch / sizeof(double);
    cudaMemset2DHelper(dwork, dwork_pitch, 0, ncol * sizeof(double), nrow);

    double * dcoords_x;
    cudaMallocHelper((void**)&dcoords_x, ncol * sizeof(double));
    cudaMemcpyHelper(dcoords_x, coords_x.data(), ncol * sizeof(double), H2D);

    double * dcoords_y;
    cudaMallocHelper((void**)&dcoords_y, nrow * sizeof(double));
    cudaMemcpyHelper(dcoords_y, coords_y.data(), nrow * sizeof(double), H2D);

    int * irow  = new int[nr];
    int * irowP = new int[nrow-nr];
    int irow_ptr  = 0;
    int irowP_ptr = 0;

    for (int i = 0; i < nr; i++) {
      int irow_r = mgard_2d::mgard_gen::get_lindex_cuda(nr, nrow, i);
      irow[irow_ptr] = irow_r;
      if (irow_ptr > 0 && irow[irow_ptr - 1] != irow[irow_ptr] - 1) {
        irowP[irowP_ptr] = irow[irow_ptr] - 1;
        irowP_ptr ++;
      } 
      irow_ptr++;
    }

    // std::cout << "irow: ";
    // for (int i = 0; i < nr; i++) std::cout << irow[i] << ", ";
    // std::cout << std::endl;

    // std::cout << "irowP: ";
    // for (int i = 0; i < nrow-nr; i++) std::cout << irowP[i] << ", ";
    // std::cout << std::endl;


    int * icol  = new int[nc];
    int * icolP = new int[ncol-nc];
    int icol_ptr  = 0;
    int icolP_ptr = 0;

    for (int i = 0; i < nc; i++) {
      int icol_r = mgard_2d::mgard_gen::get_lindex_cuda(nc, ncol, i);
      icol[icol_ptr] = icol_r;
      if (icol_ptr > 0 && icol[icol_ptr - 1] != icol[icol_ptr] - 1) {
        icolP[icolP_ptr] = icol[icol_ptr] - 1;
        icolP_ptr ++;
      } 
      icol_ptr++;
    }

    // std::cout << "icol: ";
    // for (int i = 0; i < nc; i++) std::cout << icol[i] << ", ";
    // std::cout << std::endl;

    // std::cout << "icolP: ";
    // for (int i = 0; i < ncol-nc; i++) std::cout << icolP[i] << ", ";
    // std::cout << std::endl;

    int * dirow;
    cudaMallocHelper((void**)&dirow, nr * sizeof(int));
    cudaMemcpyHelper(dirow, irow, nr * sizeof(int), H2D);

    int * dicol;
    cudaMallocHelper((void**)&dicol, nc * sizeof(int));
    cudaMemcpyHelper(dicol, icol, nc * sizeof(int), H2D);

    int * dirowP;
    cudaMallocHelper((void**)&dirowP, (nrow-nr) * sizeof(int));
    cudaMemcpyHelper(dirowP, irowP, (nrow-nr) * sizeof(int), H2D);

    int * dicolP;
    cudaMallocHelper((void**)&dicolP, (ncol-nc) * sizeof(int));
    cudaMemcpyHelper(dicolP, icolP, (ncol-nc) * sizeof(int), H2D);

    int ldv = ncol;
    int * dout_data;
    cudaMallocHelper((void**)&dout_data, (nrow*ncol + size_ratio) * sizeof(int));
    int lddout_data = ncol;
    cudaMemcpyHelper(dout_data, out_data.data(), (nrow*ncol + size_ratio) * sizeof(int), H2D);

    // mgard::dequantize_2D_interleave(nrow, ncol, v, out_data) ;
    dequantize_2D_iterleave_cuda(nrow, ncol, dv, lddv, dout_data, lddout_data);

    
    std::vector<double> row_vec(ncol);
    std::vector<double> col_vec(nrow);
    std::vector<double> work(nrow*ncol);



    // mgard_2d::mgard_gen::recompose_2D(nr, nc, nrow, ncol, l_target, v,  work, coords_x, coords_y, row_vec, col_vec);
    std::cout << "***recompose_2D_cuda***" << std::endl;
    mgard_2d::mgard_gen::recompose_2D_cuda(l_target,
                                           nrow,     ncol,
                                           nr,       nc, 
                                           dirow,    dicol,
                                           dirowP,   dicolP,
                                           dv,       lddv,
                                           dwork,    lddwork,
                                           dcoords_x, dcoords_y);


    // cudaMemcpy2DHelper(v, ncol  * sizeof(double), 
    //                    dv, lddv * sizeof(double), 
    //                    ncol * sizeof(double), nrow, 
    //                    D2H);

    // cudaMemcpy2DHelper(work.data(), ncol  * sizeof(double), 
    //                    dwork, lddwork * sizeof(double), 
    //                    ncol * sizeof(double), nrow, 
    //                    D2H);


    // mgard_2d::mgard_gen::postp_2D(nr, nc, nrow, ncol, l_target, v,  work, coords_x, coords_y, row_vec, col_vec);
    std::cout << "***postp_2D_cuda***" << std::endl;
    mgard_2d::mgard_gen::postp_2D_cuda(nrow,     ncol,
                                       nr,       nc, 
                                       dirow,    dicol,
                                       dirowP,   dicolP,
                                       dv,       lddv,
                                       dwork,    lddwork,
                                       dcoords_x, dcoords_y);

    cudaMemcpy2DHelper(v, ldv  * sizeof(double), 
                       dv, lddv * sizeof(double), 
                       ncol * sizeof(double), nrow, 
                       D2H);

    return v;
  }
}


void
refactor_cuda (const int nrow, const int ncol, 
              const int l_target, 
              double * dv, int lddv, 
              double * dwork, int lddwork) 
{
  mgard_ret ret;
  // double * dwork;
  // size_t dwork_pitch;
  // cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, nrow * sizeof(double), ncol);
  // int lddwork = dwork_pitch / sizeof(double);
  // cudaMemset2DHelper(dwork, dwork_pitch, 0, nrow * sizeof(double), ncol);
  // //int l = l_target - 1;

//#ifdef TIMING
  double pi_Ql_time = 0.0;
  double copy_level_time = 0.0;
  double assign_num_level_time = 0.0;
  double mass_matrix_multiply_row_time = 0.0;
  double restriction_row_time = 0.0;
  double solve_tridiag_M_row_time = 0.0;
  double mass_matrix_multiply_col_time = 0.0;
  double restriction_col_time = 0.0;
  double solve_tridiag_M_col_time = 0.0;
  double add_level_time = 0.0;
//#endif

  int row_stride;
  int col_stride;
  for (int l = 0; l < l_target; ++l)
  {
    int stride = std::pow (2, l); // current stride
    int Cstride = stride * 2;     // coarser stride

    row_stride = stride;
    col_stride = stride;
    ret = pi_Ql_cuda(nrow, ncol, stride, stride, dv, lddv);
    pi_Ql_time += ret.time;

    row_stride = stride;
    col_stride = stride;
    ret = copy_level_cuda (nrow, ncol, stride, stride, dv, lddv, dwork, lddwork);
    copy_level_time += ret.time;

    row_stride = Cstride;
    col_stride = Cstride;
    ret = assign_num_level_cuda(nrow, ncol, Cstride, Cstride, dwork, lddwork, 0.0);
    assign_num_level_time += ret.time;


    row_stride = 1;
    col_stride = stride;
    ret = mass_matrix_multiply_row_cuda(nrow, ncol, row_stride, col_stride, dwork, lddwork);
    mass_matrix_multiply_row_time += ret.time;

    row_stride = 1;
    col_stride = Cstride;
    ret = restriction_row_cuda(nrow, ncol, row_stride, col_stride, dwork, lddwork);
    restriction_row_time += ret.time;

    row_stride = 1;
    col_stride = Cstride;
    ret = solve_tridiag_M_row_cuda(nrow, ncol, row_stride, col_stride, dwork, lddwork);
    solve_tridiag_M_row_time += ret.time;

    if (nrow > 1) {
      row_stride = stride;
      col_stride = Cstride;
      ret = mass_matrix_multiply_col_cuda(nrow, ncol, row_stride, col_stride, dwork, lddwork);
      mass_matrix_multiply_col_time += ret.time;

      row_stride = Cstride;
      col_stride = Cstride;
      ret = restriction_col_cuda(nrow, ncol, row_stride, col_stride, dwork, lddwork);
      restriction_col_time += ret.time;

      row_stride = Cstride;
      col_stride = Cstride;
      ret = solve_tridiag_M_col_cuda(nrow, ncol, row_stride, col_stride, dwork, lddwork);
      solve_tridiag_M_col_time += ret.time;
    }

    row_stride = Cstride;
    col_stride = Cstride;
    ret = add_level_cuda (nrow, ncol, row_stride, col_stride, dv, lddv, dwork, lddwork);
    add_level_time += ret.time;

  }
  cudaFreeHelper(dwork);

//#ifdef TIMING
  std::cout << "pi_Ql_time = " << pi_Ql_time << std::endl;
  std::cout << "copy_level_time = " << copy_level_time << std::endl;
  std::cout << "assign_num_level_time = " << assign_num_level_time << std::endl;
  std::cout << "mass_matrix_multiply_row_time = " << mass_matrix_multiply_row_time << std::endl;
  std::cout << "restriction_row_time = " << restriction_row_time << std::endl;
  std::cout << "solve_tridiag_M_row_time = " << solve_tridiag_M_row_time << std::endl;
  std::cout << "mass_matrix_multiply_col_time = " << mass_matrix_multiply_col_time << std::endl;
  std::cout << "restriction_col_time = " << restriction_col_time << std::endl;
  std::cout << "solve_tridiag_M_col_time = " << solve_tridiag_M_col_time << std::endl;
  std::cout << "add_level_time = " << add_level_time << std::endl;
//#endif
}


void
recompose_cuda (const int nrow, const int ncol, 
               const int l_target, 
               double * dv,    int lddv,
               double * dwork, int lddwork) {

  double copy_level_time = 0.0;
  double assign_num_level_time = 0.0;
  double mass_matrix_multiply_row_time = 0.0;
  double restriction_row_time = 0.0;
  double solve_tridiag_M_row_time = 0.0;
  double mass_matrix_multiply_col_time = 0.0;
  double restriction_col_time = 0.0;
  double solve_tridiag_M_col_time = 0.0;
  double subtract_level_time = 0.0;
  double interpolate_from_level_nMl_row_time = 0.0;
  double interpolate_from_level_nMl_col_time = 0.0;

  int row_stride;
  int col_stride;
  mgard_ret ret;
  for (int l = l_target; l > 0; --l)
    {

      int stride = std::pow (2, l); // current stride
      int Pstride = stride / 2;

      row_stride = Pstride;
      col_stride = Pstride;
      ret = copy_level_cuda(nrow, ncol, row_stride, col_stride, dv, lddv, dwork, lddwork);
      copy_level_time += ret.time;

      row_stride = stride;
      col_stride = stride;
      ret = assign_num_level_cuda(nrow, ncol, row_stride, col_stride, dwork, lddwork, 0.0);
      assign_num_level_time += ret.time;

      row_stride = 1;
      col_stride = Pstride;
      ret = mass_matrix_multiply_row_cuda (nrow, ncol, row_stride, col_stride, dwork, lddwork);
      mass_matrix_multiply_row_time += ret.time;

      row_stride = 1;
      col_stride = stride;
      ret = restriction_row_cuda(nrow, ncol, row_stride, col_stride, dwork, lddwork);
      restriction_row_time += ret.time;

      row_stride = 1;
      col_stride = stride;
      ret = solve_tridiag_M_row_cuda(nrow, ncol, row_stride, col_stride, dwork, lddwork);
      solve_tridiag_M_row_time += ret.time;

      //int col_stride = stride;

      row_stride = Pstride;
      col_stride = stride;
      ret = mass_matrix_multiply_col_cuda (nrow, ncol, row_stride, col_stride, dwork, lddwork);
      mass_matrix_multiply_col_time += ret.time;

      row_stride = stride;
      col_stride = stride;
      ret = restriction_col_cuda(nrow, ncol, row_stride, col_stride, dwork, lddwork);
      restriction_col_time += ret.time;

      row_stride = stride;
      col_stride = stride;
      ret = solve_tridiag_M_col_cuda(nrow, ncol, row_stride, col_stride, dwork, lddwork);
      solve_tridiag_M_col_time += ret.time;

      row_stride = stride;
      col_stride = stride;
      ret = subtract_level_cuda(nrow, ncol, row_stride, col_stride, dwork, lddwork, dv, lddv);
      subtract_level_time += ret.time;

      row_stride = stride;
      col_stride = stride;
      ret = interpolate_from_level_nMl_row_cuda(nrow, ncol, row_stride, col_stride, dwork, lddwork);
      interpolate_from_level_nMl_row_time += ret.time;
      if (nrow > 1)
      {

        row_stride = stride;
        col_stride = Pstride;
        ret = interpolate_from_level_nMl_col_cuda(nrow, ncol, row_stride, col_stride, dwork, lddwork);
        interpolate_from_level_nMl_col_time += ret.time;
      }

      row_stride = stride;
      col_stride = stride;
      ret = assign_num_level_cuda(nrow, ncol, row_stride, col_stride, dv, lddv, 0.0);
      assign_num_level_time += ret.time;

      row_stride = Pstride;
      col_stride = Pstride;
      ret = subtract_level_cuda(nrow, ncol, row_stride, col_stride, dv, lddv, dwork, lddwork);
      subtract_level_time += ret.time;
    }

  std::cout << "copy_level_time = " << copy_level_time << std::endl;
  std::cout << "assign_num_level_time = " << assign_num_level_time << std::endl;
  std::cout << "mass_matrix_multiply_row_time = " << mass_matrix_multiply_row_time << std::endl;
  std::cout << "restriction_row_time = " << restriction_row_time << std::endl;
  std::cout << "solve_tridiag_M_row_time = " << solve_tridiag_M_row_time << std::endl;
  std::cout << "mass_matrix_multiply_col_time = " << mass_matrix_multiply_col_time << std::endl;
  std::cout << "restriction_col_time = " << restriction_col_time << std::endl;
  std::cout << "solve_tridiag_M_col_time = " << solve_tridiag_M_col_time << std::endl;
  std::cout << "subtract_level_time = " << subtract_level_time << std::endl;
  std::cout << "interpolate_from_level_nMl_row_time = " << interpolate_from_level_nMl_row_time << std::endl;
  std::cout << "interpolate_from_level_nMl_col_time = " << interpolate_from_level_nMl_col_time << std::endl;

}


__global__ void 
_pi_Ql_cuda(int nrow,       int ncol,
           int row_stride, int col_stride,
           double * dv,    int lddv) {
    
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * row_stride * 2;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * col_stride * 2;

    // pi_Ql

    register double a00 = dv[get_idx(lddv, x,              y             )];
    register double a01 = dv[get_idx(lddv, x,              y+col_stride  )];
    register double a02 = dv[get_idx(lddv, x,              y+col_stride*2)];
    register double a10 = dv[get_idx(lddv, x+row_stride,   y             )];
    register double a11 = dv[get_idx(lddv, x+row_stride,   y+col_stride  )];
    register double a12 = dv[get_idx(lddv, x+row_stride,   y+col_stride*2)];
    register double a20 = dv[get_idx(lddv, x+row_stride*2, y             )];
    register double a21 = dv[get_idx(lddv, x+row_stride*2, y+col_stride  )];
    register double a22 = dv[get_idx(lddv, x+row_stride*2, y+col_stride*2)];

    // if (x ==0 && y == 0) {
    //  printf("a00 = %f \n", a00);
    //  printf("a02 = %f \n", a02);
    //  printf("a20 = %f \n", a20);
    //  printf("a22 = %f \n", a22);
    // }

    a01 -= 0.5 * (a00 + a02);
    //a21 -= 0.5 * (a20 + a22);
    a10 -= 0.5 * (a00 + a20);
    //a12 -= 0.5 * (a02 + a22);
    a11 -= 0.25 * (a00 + a02 + a20 + a22);

    dv[get_idx(lddv, x,            y+col_stride)] = a01;
    dv[get_idx(lddv, x+row_stride, y           )] = a10;
    dv[get_idx(lddv, x+row_stride, y+col_stride)] = a11;
    //v[get_idx(ncol, x+stride,   y+stride*2)]  = a12;
    //v[get_idx(ncol, x+stride*2, y+stride)]    = a21;

    if (x + row_stride * 2 == nrow - 1) {
        a21 -= 0.5 * (a20 + a22);
        dv[get_idx(lddv, x+row_stride*2, y+col_stride)] = a21;
    }
    if (y + col_stride * 2 == ncol - 1) {
        a12 -= 0.5 * (a02 + a22);
        dv[get_idx(lddv, x+row_stride, y+col_stride*2)] = a12;
    }
}


mgard_ret 
pi_Ql_cuda(int nrow,       int ncol, 
          int row_stride, int col_stride,
          double * v,     int ldv) {
    double * dv;
    int lddv;

#if CPU == 1
    size_t dv_pitch;
    cudaMallocPitch(&dv, &dv_pitch, ncol * sizeof(double), nrow);
    lddv = dv_pitch / sizeof(double);
    cudaMemcpy2D(dv, lddv * sizeof(double), 
                 v,     ldv  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dv = v;
    lddv = ldv;
#endif

    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int B = 16;
    int total_thread_x = nrow/(row_stride * 2);
    int total_thread_y = ncol/(col_stride * 2);
    int tbx = min(B, total_thread_x);
    int tby = min(B, total_thread_y);
    int gridx = ceil(total_thread_x/tbx);
    int gridy = ceil(total_thread_y/tby);
    dim3 threadsPerBlock(tbx, tby);
    dim3 blockPerGrid(gridx, gridy);
    //std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
    //std::cout << "grid: " << gridx << ", " << gridy <<std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    _pi_Ql_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                  row_stride, col_stride,
                                                  dv,         lddv);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();

    gpuErrchk(cudaGetLastError ()); 
  
#if CPU == 1
    cudaMemcpy2D(v,  ldv * sizeof(double), 
                 dv, lddv * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
    cudaFree(dv);
#endif
    return mgard_ret(0, time);
    
}


__global__ void 
_copy_level_cuda(int nrow,       int ncol,
                int row_stride, int col_stride, 
                double * dv,    int lddv, 
                double * dwork, int lddwork) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * row_stride;
    int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * col_stride;
    //printf("x = %d, y = %d, stride = %d \n", x,y,stride);
    for (int x = x0; x < nrow; x += blockDim.x * gridDim.x * row_stride) {
        for (int y = y0; y < ncol; y += blockDim.y * gridDim.y * col_stride) {
            
            dwork[get_idx(lddv, x, y)] = dv[get_idx(lddwork, x, y)];
            //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
            //y += blockDim.y * gridDim.y * stride;
        }
        //x += blockDim.x * gridDim.x * stride;
    }
}






mgard_ret 
copy_level_cuda(int nrow,       int ncol, 
               int row_stride, int col_stride,
               double * v,     int ldv, 
               double * work,  int ldwork) {
    double * dv;
    int lddv;
    double * dwork;
    int lddwork;

#if CPU == 1
    size_t dv_pitch;
    cudaMallocPitch(&dv, &dv_pitch, ncol * sizeof(double), nrow);
    lddv = dv_pitch / sizeof(double);
    cudaMemcpy2D(dv, lddv * sizeof(double), 
                 v,     ldv  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);

    size_t dwork_pitch;
    cudaMallocPitch(&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
    lddwork = dwork_pitch / sizeof(double);
    cudaMemcpy2D(dwork, lddwork * sizeof(double), 
                 work,  ldwork  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dv = v;
    lddv = ldv;
    dwork = work;
    lddwork = ldwork;
#endif

    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int B = 16;
    int total_thread_x = ceil((double)nrow/row_stride);
    int total_thread_y = ceil((double)ncol/col_stride);
    int tbx = min(B, total_thread_x);
    int tby = min(B, total_thread_y);
    int gridx = ceil(total_thread_x/tbx);
    int gridy = ceil(total_thread_y/tby);
    dim3 threadsPerBlock(tbx, tby);
    dim3 blockPerGrid(gridx, gridy);

    //std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
    //std::cout << "grid: " << gridx << ", " << gridy <<std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    _copy_level_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                                                       row_stride, col_stride, 
                                                       dv,         lddv, 
                                                       dwork,      lddwork);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();

    gpuErrchk(cudaGetLastError ()); 
#if CPU == 1
    cudaMemcpy2D(work,  ldwork * sizeof(double), 
                 dwork, lddwork * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
    cudaFree(dv);
    cudaFree(dwork);
#endif
    return mgard_ret(0, time);
    //cudaMemcpy(work, dev_work, nrow * ncol * sizeof(double), cudaMemcpyDeviceToHost);

}

__global__ void 
_assign_num_level_cuda(int nrow,       int ncol, 
                      int row_stride, int col_stride,
                      double * dv,    int lddv, 
                      double num) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int y0 = (blockIdx.x * blockDim.x + threadIdx.x) * row_stride;
    int x0 = (blockIdx.y * blockDim.y + threadIdx.y) * col_stride;
    //printf("x = %d, y = %d, stride = %d \n", x,y,stride);
    for (int y = y0; y < nrow; y += blockDim.y * gridDim.y * row_stride) {
        for (int x = x0; x < ncol; x += blockDim.x * gridDim.x * col_stride) {
            dv[get_idx(lddv, y, x)] = num;
            //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
            //y += blockDim.y * gridDim.y * stride;
        }
        //x += blockDim.x * gridDim.x * stride;
    }
}


mgard_ret  
assign_num_level_cuda(int nrow,       int ncol, 
                     int row_stride, int col_stride,
                     double * v,     int ldv, 
                     double num) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride

    double * dv;
    int lddv;

#if CPU == 1
    size_t dv_pitch;
    cudaMallocPitch(&dv, &dv_pitch, ncol * sizeof(double), nrow);
    lddv = dv_pitch / sizeof(double);
    cudaMemcpy2D(dv, lddv * sizeof(double), 
                 v,     ldv  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dv = v;
    lddv = ldv;
#endif 

    int B = 16;
    int total_thread_x = ceil((double)nrow/row_stride);
    int total_thread_y = ceil((double)ncol/col_stride);
    int tbx = min(B, total_thread_x);
    int tby = min(B, total_thread_y);
    int gridx = ceil(total_thread_x/tbx);
    int gridy = ceil(total_thread_y/tby);
    dim3 threadsPerBlock(tbx, tby);
    dim3 blockPerGrid(gridx, gridy);

    //std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
    //std::cout << "grid: " << gridx << ", " << gridy <<std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    _assign_num_level_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow, ncol,
                                 row_stride, col_stride,
                                 dv, lddv, 
                                 num);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();

    gpuErrchk(cudaGetLastError ()); 

#if CPU == 1
    cudaMemcpy2D(v, ldv * sizeof(double), 
                 dv, lddv * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
    cudaFree(dv);
#endif
    return mgard_ret(0, time);
}

__global__ void 
_mass_matrix_multiply_row_cuda(int nrow,       int ncol, 
                              int row_stride, int col_stride,
                              double * dv,    int lddv) {
    //int stride = pow (2, l); // current stride

    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
    //int y = threadIdx.y * stride;
    for (int x = idx; x < nrow; x += (blockDim.x * gridDim.x) * row_stride) {
        //printf("thread working on %d \n", x);
        double * vec = dv + x * lddv;
        register double temp1, temp2;
        register double fac = 0.5;
        temp1 = vec[0];
        vec[0] = fac * (2.0 * temp1 + vec[col_stride]);
        for (int i = col_stride; i < ncol - col_stride; i += col_stride) {
            temp2 = vec[i];
            vec[i] = fac * (temp1 + 4 * temp2 + vec[i+col_stride]);
            temp1 = temp2;
        }
        vec[ncol-1] = fac * (2.0 * vec[ncol-1] + temp1);
    }
}


mgard_ret 
mass_matrix_multiply_row_cuda(int nrow,       int ncol,
                             int row_stride, int col_stride,
                             double * work,  int ldwork) {
    // int stride = pow (2, l); // current stride
    // int Cstride = stride * 2; // coarser stride

    double * dwork;
    int lddwork;

#if CPU == 1
    size_t dwork_pitch;
    cudaMallocPitch(&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
    lddwork = dwork_pitch / sizeof(double);
    cudaMemcpy2D(dwork, lddwork * sizeof(double), 
                 work,  ldwork  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dwork = work;
    lddwork = ldwork;
#endif


    int B = 128;
    int total_thread_x = nrow/row_stride;
    int tbx = min(B, total_thread_x);
    int gridx = ceil(total_thread_x/tbx);
    dim3 threadsPerBlock(tbx, 1);
    dim3 blockPerGrid(gridx, 1);

    //std::cout << "thread block: " << tbx <<std::endl;
    //std::cout << "grid: " << gridx  <<std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    _mass_matrix_multiply_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                                     row_stride, col_stride,
                                                                     dwork,      lddwork);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    gpuErrchk(cudaGetLastError ()); 
    
#if CPU == 1
    cudaMemcpy2D(work,  ldwork * sizeof(double), 
                 dwork, lddwork * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
    cudaFree(dwork);
#endif
    return mgard_ret(0, time);
}


__global__ void 
_restriction_row_cuda(int nrow,       int ncol, 
                     int row_stride, int col_stride, 
                     double * dv,    int lddv ) {
    // int stride = pow (2, l); // current stride
    // int Pstride = stride/2;
    int P_col_stride = col_stride/2;

    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
    //int y = threadIdx.y * stride;
    for (int x = idx; x < nrow; x += (blockDim.x * gridDim.x) * row_stride) {
        double * vec = dv + x * lddv;
        
        vec[0] += 0.5 * vec[P_col_stride];
        for (int i = col_stride; i < ncol - col_stride; i += col_stride) {
            vec[i] += 0.5 * (vec[i-P_col_stride] + vec[i+P_col_stride]);
        }
        vec[ncol-1] += 0.5 * vec[ncol-P_col_stride-1];
    }
}

mgard_ret 
restriction_row_cuda(int nrow,       int ncol, 
                    int row_stride, int col_stride,
                    double * work,  int ldwork) {
    // int stride = pow (2, l); // current stride
    // int Cstride = stride * 2; // coarser stride

    double * dwork;
    int lddwork;

#if CPU == 1
    size_t dwork_pitch;
    cudaMallocPitch(&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
    lddwork = dwork_pitch / sizeof(double);
    cudaMemcpy2D(dwork, lddwork * sizeof(double), 
                 work,  ldwork  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dwork = work;
    lddwork = ldwork;
#endif

    int B = 128;
    int total_thread_x = nrow/row_stride;
    int tbx = min(B, total_thread_x);
    int gridx = ceil(total_thread_x/tbx);
    dim3 threadsPerBlock(tbx, 1);
    dim3 blockPerGrid(gridx, 1);

    auto start = std::chrono::high_resolution_clock::now();
    _restriction_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                            row_stride, col_stride,
                                                            dwork,      lddwork);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    gpuErrchk(cudaGetLastError ()); 

#if CPU == 1
    cudaMemcpy2D(work,  ldwork * sizeof(double), 
                 dwork, lddwork * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
    cudaFree(dwork);
#endif
    return mgard_ret(0, time);
}


__global__ void 
_solve_tridiag_M_row_cuda(int nrow,       int ncol, 
                         int row_stride, int col_stride, 
                         double * dv,    int lddv) {

    // int stride = pow (2, l); // current stride
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
    //int y = threadIdx.y * stride;

    for (int x = idx; x < nrow; x += (blockDim.x * gridDim.x) * row_stride) {
        double * vec = dv + x * lddv;

        register double am = 2.0;
        register double bm = 1.0 / am;

        int size = ncol;
        int nlevel = (int)log2f(size - 1);
        int n = pow(2, nlevel - 1) + 1;

       // if (x == 0) printf("CUDA: %d, %d, %d \n", size, nlevel, n);
        
        double * coeff = new double[n];
        int counter = 1;
        coeff[0] = 2.0;

        for (int i = col_stride; i < ncol - col_stride; i += col_stride) {
            vec[i] -= vec[i-col_stride]/am;
            am = 4.0 - bm;
            bm = 1.0 / am;
            coeff[counter] = am;
            ++counter;
        }
        am = 2.0 - bm;
        vec[ncol - 1] -= vec[ncol - col_stride - 1] * bm;

        coeff[counter] = am;

        vec[ncol-1] /= am;

        counter--;

        for (int i = ncol-1-col_stride; i >= 0; i -= col_stride) {
            vec[i] = (vec[i] - vec[i + col_stride]) / coeff[counter];
            --counter;
            bm = 4.0 - am;
            am = 1.0 / bm;
        }
        delete [] coeff;
    }
 
}

mgard_ret 
solve_tridiag_M_row_cuda(int nrow,       int ncol,
                        int row_stride, int col_stride,
                        double * work,  int ldwork) {
    // int stride = pow (2, l); // current stride
    // int Cstride = stride * 2; // coarser stride

    double * dwork;
    int lddwork;

#if CPU == 1
    size_t dwork_pitch;
    cudaMallocPitch(&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
    lddwork = dwork_pitch / sizeof(double);
    cudaMemcpy2D(dwork, lddwork * sizeof(double), 
                 work,  ldwork  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dwork = work;
    lddwork = ldwork;
#endif

    int B = 128;
    int total_thread_x = nrow/row_stride;
    int tbx = min(B, total_thread_x);
    int gridx = ceil(total_thread_x/tbx);
    dim3 threadsPerBlock(tbx, 1);
    dim3 blockPerGrid(gridx, 1);

    //std::cout << "thread block: " << tbx <<std::endl;
    //std::cout << "grid: " << gridx  <<std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    _solve_tridiag_M_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,     
                                                                row_stride, col_stride,
                                                                dwork,      lddwork);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    gpuErrchk(cudaGetLastError ()); 
    
#if CPU == 1
    cudaMemcpy2D(work,  ldwork * sizeof(double), 
                 dwork, lddwork * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
    cudaFree(dwork);
#endif
    return mgard_ret(0, time);

}


__global__ void 
_mass_matrix_multiply_col_cuda(int nrow,       int ncol, 
                              int row_stride, int col_stride, 
                              double * dv,    int lddv) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
    //int y = threadIdx.y * stride;
    for (int x = idx; x < ncol; x += (blockDim.x * gridDim.x) * col_stride) {

        double * vec = dv + x;

        register double temp1, temp2;
        register double fac = 0.5;
        temp1 = vec[0];
        vec[0] = fac * (2.0 * temp1 + vec[row_stride*lddv]);
        for (int i = row_stride; i < nrow - row_stride; i += row_stride) {
            temp2 = vec[i*lddv];
            vec[i*lddv] = fac * (temp1 + 4 * temp2 + vec[(i+row_stride)*lddv]);
            temp1 = temp2;
        }
        vec[(nrow-1)*lddv] = fac * (2.0 * vec[(nrow-1)*lddv] + temp1);
    }
}


mgard_ret 
mass_matrix_multiply_col_cuda(int nrow,       int ncol, 
                             int row_stride, int col_stride, 
                             double * work,  int ldwork) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride

    double * dwork;
    int lddwork;

#if CPU == 1
    size_t dwork_pitch;
    cudaMallocPitch(&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
    lddwork = dwork_pitch / sizeof(double);
    cudaMemcpy2D(dwork, lddwork * sizeof(double), 
                 work,  ldwork  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dwork = work;
    lddwork = ldwork;
#endif

    int B = 128;
    int total_thread_x = ceil(ncol/col_stride);
    int tbx = min(B, total_thread_x);
    int gridx = ceil(total_thread_x/tbx);
    dim3 threadsPerBlock(tbx, 1);
    dim3 blockPerGrid(gridx, 1);

    //std::cout << "thread block: " << tbx <<std::endl;
    //std::cout << "grid: " << gridx  <<std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    _mass_matrix_multiply_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow, ncol, 
                                                                     row_stride, col_stride,
                                                                     dwork, lddwork);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    gpuErrchk(cudaGetLastError ()); 
    
#if CPU == 1
    cudaMemcpy2D(work,  ldwork * sizeof(double), 
                 dwork, lddwork * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
    cudaFree(dwork);
#endif
    return mgard_ret(0, time);
}


__global__ void 
_restriction_col_cuda(int nrow,       int ncol, 
                     int row_stride, int col_stride, 
                     double * dv,    int lddv) {
    // int stride = pow (2, l); // current stride
    // int Pstride = stride/2;
    int P_row_stride = row_stride/2;

    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
    //int y = threadIdx.y * stride;
    for (int x = idx; x < ncol; x += blockDim.x * gridDim.x * col_stride) {
        double * vec = dv + x; //stride = Cstride outside
        
        vec[0] += 0.5 * vec[P_row_stride*lddv];
        for (int i = row_stride; i < nrow - row_stride; i += row_stride) {
            vec[i*lddv] += 0.5 * (vec[(i-P_row_stride)*lddv] + vec[(i+P_row_stride)*lddv]);
        }
        vec[(nrow-1)*lddv] += 0.5 * vec[(nrow-P_row_stride-1)*lddv];
    }
}


mgard_ret 
restriction_col_cuda(int nrow,       int ncol, 
                    int row_stride, int col_stride, 
                    double * work, int ldwork) {
   // int stride = pow (2, l); // current stride

    double * dwork;
    int lddwork;
#if CPU == 1
    size_t dwork_pitch;
    cudaMallocPitch(&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
    lddwork = dwork_pitch / sizeof(double);
    cudaMemcpy2D(dwork, lddwork * sizeof(double), 
                 work,  ldwork  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dwork = work;
    lddwork = ldwork;
#endif

    //std::cout << "restriction_col_cuda - num of thread = " << ncol/stride+1 << std::endl;
    int B = 128;
    int total_thread_x = ceil(ncol/col_stride);
    int tbx = min(B, total_thread_x);
    int gridx = ceil(total_thread_x/tbx);
    dim3 threadsPerBlock(tbx, 1);
    dim3 blockPerGrid(gridx, 1);

    auto start = std::chrono::high_resolution_clock::now();
    _restriction_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                                                            row_stride, col_stride,
                                                            dwork,      lddwork);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    gpuErrchk(cudaGetLastError ()); 

#if CPU == 1
    cudaMemcpy2D(work,  ldwork * sizeof(double), 
                 dwork, lddwork * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
    cudaFree(dwork);
#endif
    return mgard_ret(0, time);
}



__global__ void 
_solve_tridiag_M_col_cuda(int nrow,       int ncol, 
                         int row_stride, int col_stride, 
                         double * dv,    int lddv) {
    //int stride = pow (2, l); // current stride
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
    //int y = threadIdx.y * stride;
    for (int x = idx; x < ncol; x += blockDim.x * gridDim.x * col_stride) {

        double * vec = dv + x; //stride = Cstride outside

        register double am = 2.0;
        register double bm = 1.0 / am;

        int size = nrow;
        int nlevel = (int)log2f(size - 1);
        int n = pow(2, nlevel - 1) + 1;

        //if (x == 0) printf("CUDA: %d, %d, %d \n", size, nlevel, n);
        
        double * coeff = new double[n];
        int counter = 1;
        coeff[0] = am;

        for (int i = row_stride; i < nrow - row_stride; i += row_stride) {
            vec[i*lddv] -= vec[(i-row_stride)*lddv]/am;
            am = 4.0 - bm;
            bm = 1.0 / am;
            coeff[counter] = am;
            ++counter;
        }
        am = 2.0 - bm;
        vec[(nrow - 1)*lddv] -= vec[(nrow - row_stride - 1)*lddv] * bm;

        coeff[counter] = am;

        vec[(nrow-1)*lddv] /= am;

        counter--;

        for (int i = nrow-1-row_stride; i >= 0; i -= row_stride) {
            vec[i*lddv] = (vec[i*lddv] - vec[(i + row_stride)*lddv]) / coeff[counter];
            --counter;
            bm = 4.0 - am;
            am = 1.0 / bm;
        }
        delete [] coeff;
    }
 
}

mgard_ret 
solve_tridiag_M_col_cuda(int nrow,       int ncol, 
                        int row_stride, int col_stride,
                        double * work,  int ldwork) {
    // int stride = pow (2, l); // current stride
    // int Cstride = stride * 2; // coarser stride

    double * dwork;
    int lddwork;
#if CPU == 1
    size_t dwork_pitch;
    cudaMallocPitch(&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
    lddwork = dwork_pitch / sizeof(double);
    cudaMemcpy2D(dwork, lddwork * sizeof(double), 
                 work,  ldwork  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dwork = work;
    lddwork = ldwork;
#endif

    int B = 128;
    int total_thread_x = ceil(ncol/col_stride);
    int tbx = min(B, total_thread_x);
    int gridx = ceil(total_thread_x/tbx);
    dim3 threadsPerBlock(tbx, 1);
    dim3 blockPerGrid(gridx, 1);


    auto start = std::chrono::high_resolution_clock::now();
    _solve_tridiag_M_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                                                                row_stride, col_stride,
                                                                dwork,      lddwork);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    gpuErrchk(cudaGetLastError ()); 
    
#if CPU == 1
    cudaMemcpy2D(work,  ldwork * sizeof(double), 
                 dwork, lddwork * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
    cudaFree(dwork);
#endif
    return mgard_ret(0, time);
}

__global__ void 
_add_level_cuda(int nrow,       int ncol, 
               int row_stride, int col_stride, 
               double * dv,    int lddv, 
               double * dwork, int lddwork) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int idx_x = (blockIdx.x * blockDim.x + threadIdx.x) * row_stride;
    int idx_y = (blockIdx.y * blockDim.y + threadIdx.y) * col_stride;
    //printf("x = %d, y = %d, stride = %d \n", x,y,stride);
    for (int x = idx_x; x < nrow; x += blockDim.x * gridDim.x * row_stride) {
        for (int y = idx_y; y < ncol; y += blockDim.y * gridDim.y * col_stride) {
             dv[get_idx(lddv, x, y)] += dwork[get_idx(lddwork, x, y)];
            //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
            //y += blockDim.y * gridDim.y * stride;
        }
        //x += blockDim.x * gridDim.x * stride;
    }
}


mgard_ret 
add_level_cuda(int nrow,       int ncol, 
              int row_stride, int col_stride, 
              double * v,     int ldv, 
              double * work, int ldwork) {
    // int stride = pow (2, l); // current stride
    // int Cstride = stride * 2; // coarser stride

    double * dv;
    int lddv;
    double * dwork;
    int lddwork;


#if CPU == 1
    size_t dwork_pitch;
    cudaMallocPitch(&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
    lddwork = dwork_pitch / sizeof(double);
    cudaMemcpy2D(dwork, lddwork * sizeof(double), 
                 work,  ldwork  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);

    size_t dv_pitch;
    cudaMallocPitch(&dv, &dv_pitch, ncol * sizeof(double), nrow);
    lddv = dv_pitch / sizeof(double);
    cudaMemcpy2D(dv, lddv * sizeof(double), 
                 v,  ldv  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dv = v;
    lddv = ldv;
    dwork = work;
    lddwork = ldwork;
#endif


    int B = 16;
    int total_thread_x = nrow/row_stride;
    int total_thread_y = ncol/col_stride;
    int tbx = min(B, total_thread_x);
    int tby = min(B, total_thread_y);
    int gridx = ceil(total_thread_x/tbx);
    int gridy = ceil(total_thread_y/tby);
    dim3 threadsPerBlock(tbx, tby);
    dim3 blockPerGrid(gridx, gridy);

    //std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
    //std::cout << "grid: " << gridx << ", " << gridy <<std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    _add_level_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                                                      row_stride, col_stride, 
                                                      dv,         lddv,
                                                      dwork,      lddwork);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    gpuErrchk(cudaGetLastError ()); 

#if CPU == 1
    cudaMemcpy2D(work,  ldwork * sizeof(double), 
                 dwork, lddwork * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
    cudaMemcpy2D(v,  ldv * sizeof(double), 
                 dv, lddv * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
    cudaFree(dv);
    cudaFree(dwork);
#endif
    return mgard_ret(0, time);
}

__global__ void 
_quantize_2D_iterleave_cuda(int nrow,    int ncol, 
                           double * dv, int lddv, 
                           int * dwork, int lddwork,
                           double quantizer) {
    int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    int y0 = blockIdx.y * blockDim.y + threadIdx.y;

    int size_ratio = sizeof(double) / sizeof(int);

    for (int x = x0; x < nrow; x += blockDim.x * gridDim.x) {
        for (int y = y0; y < ncol; y += blockDim.y * gridDim.y) {
            int quantum = (int)(dv[get_idx(lddv, x, y)] / quantizer);
            dwork[get_idx(lddwork, x, y) + size_ratio] = quantum;
        }
    }
}

mgard_ret 
quantize_2D_iterleave_cuda (int nrow,    int ncol, 
                           double * v,  int ldv, 
                           int * work,  int ldwork, 
                           double norm, double tol) {
    int size_ratio = sizeof(double) / sizeof(int);
    double quantizer = norm * tol;

    double * dv;
    int lddv;
    int * dwork;
    int lddwork;


#if CPU == 1
    // cudaMalloc(&dwork, (nrow * ncol + size_ratio) * sizeof(int));
    // //cudaMemcpy(dwork, &quantizer, sizeof(double), cudaMemcpyHostToDevice);
    // lddwork = ncol;

    // size_t dv_pitch;
    // cudaMallocPitch(&dv, &dv_pitch, nrow * sizeof(double), ncol);
    // lddv = dv_pitch / sizeof(double);
    // cudaMemcpy2D(dv, lddv * sizeof(double), 
    //              v,   ldv  * sizeof(double), 
    //              ncol * sizeof(double), nrow, 
    //              cudaMemcpyHostToDevice);
#endif

#if GPU == 1
     dv = v;
     lddv = ldv;
     dwork = work;
     lddwork = ldwork;
#endif

    int B = 16;
    int total_thread_x = nrow;
    int total_thread_y = ncol;
    int tbx = min(B, total_thread_x);
    int tby = min(B, total_thread_y);
    int gridx = ceil((float)total_thread_x/tbx);
    int gridy = ceil((float)total_thread_y/tby);
    dim3 threadsPerBlock(tbx, tby);
    dim3 blockPerGrid(gridx, gridy);

    cudaMemcpy(dwork, &quantizer, sizeof(double), cudaMemcpyHostToDevice);
    auto start = std::chrono::high_resolution_clock::now();
    _quantize_2D_iterleave_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow, ncol, 
                                                                  dv, lddv, 
                                                                  dwork, lddwork, 
                                                                  quantizer);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    gpuErrchk(cudaGetLastError ()); 

    //int * work2 = new int[nrow*ncol + size_ratio];
#if CPU == 1
    // cudaMemcpy(work, dwork, (nrow * ncol + size_ratio) * sizeof(int), cudaMemcpyDeviceToHost);

    // cudaMemcpy2D(v,  ldv * sizeof(double), 
    //              dv, lddv * sizeof(double),
    //              ncol * sizeof(double), nrow, 
    //              cudaMemcpyDeviceToHost);
    // cudaFree(dv);
    // cudaFree(dwork);
#endif
    return mgard_ret(0, time);
}


__global__ void 
_dequantize_2D_iterleave_cuda(int nrow,    int ncol, 
                             double * dv, int lddv, 
                             int * dwork, int lddwork,
                             double quantizer) {
    int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    int y0 = blockIdx.y * blockDim.y + threadIdx.y;

    int size_ratio = sizeof(double) / sizeof(int);

    for (int x = x0; x < nrow; x += blockDim.x * gridDim.x) {
        for (int y = y0; y < ncol; y += blockDim.y * gridDim.y) {
            dv[get_idx(lddv, x, y)] = quantizer * double(dwork[get_idx(lddwork, x, y) + size_ratio]);
        }
    }
}

mgard_ret  
dequantize_2D_iterleave_cuda (int nrow,   int ncol, 
                             double * v, int ldv, 
                             int * work, int ldwork) {
    int size_ratio = sizeof(double) / sizeof(int);
    double quantizer;

    double * dv;
    int lddv;
    int * dwork;
    int lddwork;


#if CPU == 1
    cudaMalloc(&dwork, (nrow * ncol + size_ratio) * sizeof(int));
    cudaMemcpy(dwork, work, (nrow * ncol + size_ratio) * sizeof(int), cudaMemcpyHostToDevice);
    lddwork = ncol;

    size_t dv_pitch;
    cudaMallocPitch(&dv, &dv_pitch, nrow * sizeof(double), ncol);
    lddv = dv_pitch / sizeof(double);
    cudaMemcpy2D(dv, lddv * sizeof(double), 
                 v,   ldv  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
     dv = v;
     lddv = ldv;
     dwork = work;
     lddwork = ldwork;
#endif

    int B = 16;
    int total_thread_x = nrow;
    int total_thread_y = ncol;
    int tbx = min(B, total_thread_x);
    int tby = min(B, total_thread_y);
    int gridx = ceil(total_thread_x/tbx);
    int gridy = ceil(total_thread_y/tby);
    dim3 threadsPerBlock(tbx, tby);
    dim3 blockPerGrid(gridx, gridy);

    cudaMemcpy(&quantizer, dwork, sizeof(double), cudaMemcpyDeviceToHost);
    auto start = std::chrono::high_resolution_clock::now();
    _dequantize_2D_iterleave_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,  ncol, 
                                                                    dv,    lddv, 
                                                                    dwork, lddwork, 
                                                                    quantizer);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    gpuErrchk(cudaGetLastError ()); 

    //int * work2 = new int[nrow*ncol + size_ratio];
#if CPU == 1
    cudaMemcpy(work, dwork, (nrow * ncol + size_ratio) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy2D(v,  ldv * sizeof(double), 
                 dv, lddv * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
    cudaFree(dv);
    cudaFree(dwork);
#endif
    return mgard_ret(0, time);
}


__global__ void 
_subtract_level_cuda(int nrow,       int ncol,
                    int row_stride, int col_stride, 
                    double * dv,    int lddv, 
                    double * dwork, int lddwork) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int idx_x = (blockIdx.x * blockDim.x + threadIdx.x) * row_stride;
    int idx_y = (blockIdx.y * blockDim.y + threadIdx.y) * col_stride;
    //printf("x = %d, y = %d, stride = %d \n", x,y,stride);
    for (int x = idx_x; x < nrow; x += blockDim.x * gridDim.x * row_stride) {
        for (int y = idx_y; y < ncol; y += blockDim.y * gridDim.y * col_stride) {
             dv[get_idx(lddv, x, y)] -= dwork[get_idx(lddwork, x, y)];
            //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
            //y += blockDim.y * gridDim.y * stride;
        }
        //x += blockDim.x * gridDim.x * stride;
    }
}

mgard_ret 
subtract_level_cuda(int nrow,       int ncol, 
                   int row_stride, int col_stride,
                   double * v,     int ldv,
                   double * work,  int ldwork) {
    // int stride = pow (2, l); // current stride
    // int Cstride = stride * 2; // coarser stride

    double * dv;
    int lddv;
    double * dwork;
    int lddwork;


#if CPU == 1
    size_t dwork_pitch;
    cudaMallocPitch(&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
    lddwork = dwork_pitch / sizeof(double);
    cudaMemcpy2D(dwork, lddwork * sizeof(double), 
                 work,  ldwork  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);

    size_t dv_pitch;
    cudaMallocPitch(&dv, &dv_pitch, ncol * sizeof(double), nrow);
    lddv = dv_pitch / sizeof(double);
    cudaMemcpy2D(dv, lddv * sizeof(double), 
                 v,  ldv  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dv = v;
    lddv = ldv;
    dwork = work;
    lddwork = ldwork;
#endif


    int B = 16;
    int total_thread_x = nrow/row_stride;
    int total_thread_y = ncol/col_stride;
    int tbx = min(B, total_thread_x);
    int tby = min(B, total_thread_y);
    int gridx = ceil(total_thread_x/tbx);
    int gridy = ceil(total_thread_y/tby);
    dim3 threadsPerBlock(tbx, tby);
    dim3 blockPerGrid(gridx, gridy);

    //std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
    //std::cout << "grid: " << gridx << ", " << gridy <<std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    _subtract_level_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                           row_stride, col_stride, 
                                                           dv,         lddv,
                                                           dwork,      lddwork);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    gpuErrchk(cudaGetLastError ()); 

#if CPU == 1
    cudaMemcpy2D(work,  ldwork * sizeof(double), 
                 dwork, lddwork * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
    cudaMemcpy2D(v,  ldv * sizeof(double), 
                 dv, lddv * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
    cudaFree(dv);
    cudaFree(dwork);
#endif
    return mgard_ret(0, time);
}


__global__ void 
_interpolate_from_level_nMl_row_cuda(int nrow,       int ncol, 
                                    int row_stride, int col_stride,
                                    double * dv,    int lddv) {
  // int stride = std::pow (2, l);
   // int Pstride = stride / 2;
    int P_col_stride = col_stride/2;

    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
    //int y = threadIdx.y * stride;
    for (int x = idx; x < nrow; x += (blockDim.x * gridDim.x) * row_stride) {
        //printf("thread working on %d \n", x);
        double * vec = dv + x * lddv;
        for (int i = col_stride; i < ncol; i += col_stride) {
            vec[i - P_col_stride] = 0.5 * (vec[i - col_stride] + vec[i]);
        }
    }
}


mgard_ret 
interpolate_from_level_nMl_row_cuda(int nrow,       int ncol, 
                                   int row_stride, int col_stride,
                                   double * work,  int ldwork) {
    // int stride = pow (2, l); // current stride
    // int Cstride = stride * 2; // coarser stride

    double * dwork;
    int lddwork;

#if CPU == 1
    size_t dwork_pitch;
    cudaMallocPitch(&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
    lddwork = dwork_pitch / sizeof(double);
    cudaMemcpy2D(dwork, lddwork * sizeof(double), 
                 work,  ldwork  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dwork = work;
    lddwork = ldwork;
#endif


    int B = 128;
    int total_thread_x = nrow/row_stride;
    int tbx = min(B, total_thread_x);
    int gridx = ceil(total_thread_x/tbx);
    dim3 threadsPerBlock(tbx, 1);
    dim3 blockPerGrid(gridx, 1);

    //std::cout << "thread block: " << tbx <<std::endl;
    //std::cout << "grid: " << gridx  <<std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    _interpolate_from_level_nMl_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                                                                           row_stride, col_stride,
                                                                           dwork,      lddwork);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    gpuErrchk(cudaGetLastError ()); 
    
#if CPU == 1
    cudaMemcpy2D(work,  ldwork * sizeof(double), 
                 dwork, lddwork * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);

    cudaFree(dwork);
#endif
    return mgard_ret(0, time);
}


__global__ void 
_interpolate_from_level_nMl_col_cuda(int nrow,       int ncol, 
                                    int row_stride, int col_stride, 
                                    double * dv,    int lddv) {
    // int stride = std::pow (2, l);
    // int Pstride = stride / 2;
    int P_row_stride = row_stride/2;

    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
    //int y = threadIdx.y * stride;
    for (int x = idx; x < ncol; x += (blockDim.x * gridDim.x) * col_stride) {
        double * vec = dv + x;
        for (int i = row_stride; i < nrow; i += row_stride) {
            vec[(i-P_row_stride)*lddv] = 0.5 * (vec[(i-row_stride)*lddv] + vec[(i)*lddv]);
        }
    }
}


mgard_ret  
interpolate_from_level_nMl_col_cuda(int nrow,       int ncol, 
                                   int row_stride, int col_stride, 
                                   double * work,  int ldwork) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride

    double * dwork;
    int lddwork;

#if CPU == 1
    size_t dwork_pitch;
    cudaMallocPitch(&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
    lddwork = dwork_pitch / sizeof(double);
    cudaMemcpy2D(dwork, lddwork * sizeof(double), 
                 work,  ldwork  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dwork = work;
    lddwork = ldwork;
#endif

    int B = 128;
    int total_thread_x = ceil(ncol/col_stride);
    int tbx = min(B, total_thread_x);
    int gridx = ceil(total_thread_x/tbx);
    dim3 threadsPerBlock(tbx, 1);
    dim3 blockPerGrid(gridx, 1);

    //std::cout << "thread block: " << tbx <<std::endl;
    //std::cout << "grid: " << gridx  <<std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    _interpolate_from_level_nMl_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                                                                           row_stride, col_stride,
                                                                           dwork,      lddwork);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    gpuErrchk(cudaGetLastError ()); 
    
#if CPU == 1
    cudaMemcpy2D(work,  ldwork * sizeof(double), 
                 dwork, lddwork * sizeof(double),
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
    cudaFree(dwork);
#endif
    return mgard_ret(0, time);
}

}





