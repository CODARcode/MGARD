#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_nuni_3d_cuda_cpt_l2_sm.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include "mgard_nuni_2d_cuda_kernels.h"
#include <chrono>


#include <fstream>
#include <cmath>

namespace mgard_common {
inline int get_index3(const int ncol, const int nfib, const int i, const int j,
                      const int k) {
  return (ncol * i + j) * nfib + k;
}

inline int get_index(const int ncol, const int i, const int j) {
  return ncol * i + j;
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

// template <typename T>
// void prep_3D_cuda(const int nr, const int nc, const int nf, const int nrow,
//              const int ncol, const int nfib, const int l_target, double *v,
//              std::vector<double> &work, std::vector<double> &work2d,
//              std::vector<double> &coords_x, std::vector<double> &coords_y,
//              std::vector<double> &coords_z, 
//              int * dirow, int * dicol, int * difib,
//              int * dirowP, int * dicolP, int * difibP,
//              int * dirowA, int * dicolA, int * difibA,
//              T * ddist_r, T * ddist_c, T * ddist_f,
//              T * dcoords_r, T * dcoords_c, T * dcoords_f, 
//              int B, mgard_cuda_handle & handle, bool profile ) {

template <typename T>
void prep_3D_cuda( const int nrow, const int ncol, const int nfib, 
                   const int nr, const int nc, const int nf,
                   int * dirow, int * dicol, int * difib,
                   int * dirowP, int * dicolP, int * difibP,
                   int * dirowA, int * dicolA, int * difibA,
                   T * ddist_r, T * ddist_c, T * ddist_f,
                   T * dcoords_r, T * dcoords_c, T * dcoords_f, 
                   T * dv, int lddv1, int lddv2,
                   T * dwork, int lddwork1, int lddwork2,
                   int B, mgard_cuda_handle & handle, bool profile ) {
  // int l = 0;
  // int stride = 1;

  // std::vector<double> v2d(nrow * ncol), fib_vec(nfib);
  // std::vector<double> row_vec(ncol);
  // std::vector<double> col_vec(nrow);

  mgard_cuda_ret ret;
  // T * dv;
  // size_t dv_pitch;
  // cudaMalloc3DHelper((void**)&dv, &dv_pitch, nfib * sizeof(T), ncol, nrow);
  // int lddv1 = dv_pitch / sizeof(T);
  // int lddv2 = ncol;

  // T * dwork;
  // size_t dwork_pitch;
  // cudaMalloc3DHelper((void**)&dwork, &dwork_pitch, nfib * sizeof(T), ncol, nrow);
  // int lddwork1 = dwork_pitch / sizeof(T);
  // int lddwork2 = ncol;

  // T * work2 = new T[nrow*ncol*nfib];


  // cudaMemcpy3DAsyncHelper(dv, lddv1 * sizeof(T), nfib * sizeof(T), ncol,
  //                         v,  nfib  * sizeof(T), nfib * sizeof(T), ncol,
  //                         nfib * sizeof(T), ncol, nrow,
  //                         H2D, handle, 0, profile);

  // print_matrix_cuda(1, ncol-nc, dicolP, ncol-nc);
  // print_matrix_cuda(3, 10, 10, dv, lddv1, lddv2, nfib);

  ret = pi_Ql3D_first_fib(nrow,        ncol,         nfib,
                          nr,          nc,           nf, 
                          dirow,       dicol,        difibP,
                          dv,          lddv1,        lddv2, 
                          ddist_r,     ddist_c,      ddist_f,
                          B, handle, 
                          0, profile);
  // std::cout << "pi_Ql3D_first_fib: " << nrow*ncol*nfib*sizeof(double)/ret.time/1e9 << "GB/s\n";
  // print_matrix_cuda(1, 10, 10, dv, lddv1, lddv2, nfib);

  ret = pi_Ql3D_first_col(nrow,        ncol,         nfib,
                          nr,          nc,           nf, 
                          dirow,       dicolP,        difib,
                          dv,          lddv1,        lddv2, 
                          ddist_r,     ddist_c,      ddist_f,
                          B, handle, 
                          0, profile);
  // std::cout << "pi_Ql3D_first_col: " << nrow*ncol*nfib*sizeof(double)/ret.time/1e9 << "GB/s\n";
  // print_matrix_cuda(3, 10, 10, dv, lddv1, lddv2, nfib);

  ret = pi_Ql3D_first_row(nrow,        ncol,         nfib,
                          nr,          nc,           nf, 
                          dirowP,      dicol,        difib,
                          dv,          lddv1,        lddv2, 
                          ddist_r,     ddist_c,      ddist_f,
                          B, handle, 
                          0, profile);
  // std::cout << "pi_Ql3D_first_row: " << nrow*ncol*nfib*sizeof(double)/ret.time/1e9 << "GB/s\n";
  // print_matrix_cuda(3, 10, 10, dv, lddv1, lddv2, nfib);

  ret = pi_Ql3D_first_fib_col(nrow,        ncol,         nfib,
                              nr,          nc,           nf, 
                              dirow,       dicolP,        difibP,
                              dv,          lddv1,        lddv2, 
                              ddist_r,     ddist_c,      ddist_f,
                              B, handle, 
                              0, profile);
  // std::cout << "pi_Ql3D_first_fib_col: " << nrow*ncol*nfib*sizeof(double)/ret.time/1e9 << "GB/s\n";

  ret = pi_Ql3D_first_fib_row(nrow,        ncol,         nfib,
                              nr,          nc,           nf, 
                              dirowP,      dicol,        difibP,
                              dv,          lddv1,        lddv2, 
                              ddist_r,     ddist_c,      ddist_f,
                              B, handle, 
                              0, profile);
  // std::cout << "pi_Ql3D_first_fib_row: " << nrow*ncol*nfib*sizeof(double)/ret.time/1e9 << "GB/s\n";

  ret = pi_Ql3D_first_col_row(nrow,        ncol,         nfib,
                              nr,          nc,           nf, 
                              dirowP,      dicolP,       difib,
                              dv,          lddv1,        lddv2, 
                              ddist_r,     ddist_c,      ddist_f,
                              B, handle, 
                              0, profile);
  // std::cout << "pi_Ql3D_first_col_row: " << nrow*ncol*nfib*sizeof(double)/ret.time/1e9 << "GB/s\n";

  ret = pi_Ql3D_first_fib_col_row(nrow,        ncol,         nfib,
                              nr,          nc,           nf, 
                              dirowP,      dicolP,       difibP,
                              dv,          lddv1,        lddv2, 
                              ddist_r,     ddist_c,      ddist_f,
                              B, handle, 
                              0, profile);
  // std::cout << "pi_Ql3D_first_fib_col_row: " << nrow*ncol*nfib*sizeof(double)/ret.time/1e9 << "GB/s\n";

  cudaMemcpy3DAsyncHelper(dwork, lddwork1 * sizeof(T), nfib * sizeof(T), ncol,
                          dv, lddv1 * sizeof(T), nfib * sizeof(T), ncol,
                          nfib * sizeof(T), ncol, nrow,
                          D2D, handle, 0, profile);

  assign_num_level_l_cuda(nrow,        ncol,         nfib,
                          nr,          nc,           nf, 
                          1, 1, 1,
                          dirow,       dicol,        difib,
                          dwork, lddwork1, lddwork2,
                          (T)0.0,
                          B, handle, 0, profile);

  
  for (int r = 0; r < nrow; r += 1) {
    T * dwork_local = dwork + r * lddwork1*lddwork2;
    mgard_2d::mgard_cannon::mass_matrix_multiply_col_cuda(ncol,  nfib, 
                                                          1, 1,
                                                          dwork_local,   lddwork1,
                                                          dcoords_c,
                                                          B, handle, 0, profile);
    mgard_2d::mgard_gen::restriction_first_col_cuda(ncol,  nfib,
                                                    nc,  nfib, 
                                                    1, 1, 
                                                    dicolP, difibA,
                                                    dwork_local,   lddwork1,
                                                    dcoords_c,
                                                    B, handle, 0, profile);
  }
  

  for (int r = 0; r < nr; r += 1) {
    int ir = get_lindex(nr, nrow, r);
    T * dwork_local = dwork + ir * lddwork1*lddwork2;
    mgard_2d::mgard_gen::solve_tridiag_M_l_col_cuda(ncol,  nfib,
                                                    nc,    nfib, //nf,
                                                    1, 1,
                                                    dicol,      difibA,
                                                    dwork_local,   lddwork1,
                                                    dcoords_c,
                                                    B, handle, 0, profile);
  }


  for (int c = 0; c < ncol; c += 1) {
    T * dwork_local = dwork + c * lddwork1;
    mgard_2d::mgard_cannon::mass_matrix_multiply_col_cuda(nrow,  nfib, 
                                                          1, 1,
                                                          dwork_local,   lddwork1*lddwork2,
                                                          dcoords_r,
                                                          B, handle, 0, profile);
    mgard_2d::mgard_gen::restriction_first_col_cuda(nrow,  nfib, 
                                                    nr,    nfib,
                                                    1, 1,
                                                    dirowP, difibA,
                                                    dwork_local,   lddwork1*lddwork2,
                                                    dcoords_r,
                                                    B, handle, 0, profile);
  }

  for (int c = 0; c < nc; c += 1) {
    int ic = get_lindex(nc, ncol, c);
    T * dwork_local = dwork + ic * lddwork1;
    mgard_2d::mgard_gen::solve_tridiag_M_l_col_cuda(nrow,  nfib,
                                                    nr,    nfib, //nf,
                                                    1, 1,
                                                    dirow,      difibA,
                                                    dwork_local,   lddwork1*lddwork2,
                                                    dcoords_r,
                                                    B, handle, 0, profile);
  }



  for (int r = 0; r < nr; r += 1) {
    int ir = get_lindex(nr, nrow, r);
    T * dwork_local = dwork + ir * lddwork1*lddwork2;
    mgard_2d::mgard_gen::mass_mult_l_row_cuda(ncol,       nfib,
                                             nc,         nfib,
                                             1, 1,
                                             dicol,    difibA,
                                             dwork_local,   lddwork1,
                                             dcoords_f,
                                             B, handle, 0, profile);
  }

   for (int r = 0; r < nr; r += 1) {
    int ir = get_lindex(nr, nrow, r);
    T * dwork_local = dwork + ir * lddwork1*lddwork2;
    mgard_2d::mgard_gen::restriction_first_row_cuda(ncol,  nfib, 
                                                    nc,    nf,
                                                    1, 1,
                                                    dicol, difibP,
                                                    dwork_local,   lddwork1,
                                                    dcoords_f,
                                                    B, handle, 0, profile);
    mgard_2d::mgard_gen::solve_tridiag_M_l_row_cuda(ncol,  nfib, 
                                                    nc,    nf,
                                                    1, 1,
                                                    dicol,      difib,
                                                    dwork_local,   lddwork1,
                                                    dcoords_f,
                                                    B, handle, 0, profile); 
  }

  mgard_gen::add_level_l_cuda(nrow, ncol, nfib,
                              nr, nc, nf,
                              1, 1, 1,
                              dirow, dicol, difib,
                              dv, lddv1, lddv2,
                              dwork, lddwork1, lddwork2,
                              B, handle, 0, profile); 


  // cudaMemcpy3DAsyncHelper(v,  nfib  * sizeof(T), nfib * sizeof(T), ncol,
  //                         dv, lddv1 * sizeof(T), nfib * sizeof(T), ncol,
  //                         nfib * sizeof(T), ncol, nrow,
  //                         D2H, handle, 0, profile);

  // cudaMemcpy3DAsyncHelper(work.data(), nfib * sizeof(T), nfib * sizeof(T), ncol,
  //                         dwork, lddwork1 * sizeof(T), nfib * sizeof(T), ncol,
  //                         nfib * sizeof(T), ncol, nrow,
  //                         D2H, handle, 0, profile);

  // pi_Ql3D_first(nr, nc, nf, nrow, ncol, nfib, l, v, coords_x, coords_y,
  //               coords_z, row_vec, col_vec, fib_vec);

  //mgard_gen::copy3_level(0, v, work.data(), nrow, ncol, nfib);
  // mgard_gen::assign3_level_l(0, work.data(), 0.0, nr, nc, nf, nrow, ncol, nfib);

  // for (int kfib = 0; kfib < nfib; kfib += stride) {
  //   mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kfib);
  //   // mgard_gen::refactor_2D_first(nr, nc, nrow, ncol, l, v2d.data(), work2d,
  //   //                              coords_x, coords_y, row_vec, col_vec);
  //   for (int irow = 0; irow < nrow; ++irow) {
  //   //        int ir = get_lindex(nr, nrow, irow);
  //   for (int jcol = 0; jcol < ncol; ++jcol) {
  //     row_vec[jcol] = work2d[mgard_common::get_index(ncol, irow, jcol)];
  //   }

  //   // mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);

  //   // restriction_first(row_vec, coords_x, nc, ncol);

  //   for (int jcol = 0; jcol < ncol; ++jcol) {
  //     work2d[mgard_common::get_index(ncol, irow, jcol)] = row_vec[jcol];
  //   }
  // }

  // for (int irow = 0; irow < nr; ++irow) {
  //   int ir = get_lindex(nr, nrow, irow);
  //   for (int jcol = 0; jcol < ncol; ++jcol) {
  //     row_vec[jcol] = work2d[mgard_common::get_index(ncol, ir, jcol)];
  //   }

  //   // mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol);

  //   for (int jcol = 0; jcol < ncol; ++jcol) {
  //     work2d[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol];
  //   }
  // }
  


  // //   //   //std::cout  << "recomposing-colsweep" << "\n";

  // //     // column-sweep, this is the slow one! Need something like column_copy
  // if (nrow > 1) // check if we have 1-D array..
  // {
  //   for (int jcol = 0; jcol < ncol; ++jcol) {
  //     //      int jr  = get_lindex(nc,  ncol,  jcol);
  //     for (int irow = 0; irow < nrow; ++irow) {
  //       col_vec[irow] = work2d[mgard_common::get_index(ncol, irow, jcol)];
  //     }

  //     // mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);

  //     // mgard_gen::restriction_first(col_vec, coords_y, nr, nrow);

  //     for (int irow = 0; irow < nrow; ++irow) {
  //       work2d[mgard_common::get_index(ncol, irow, jcol)] = col_vec[irow];
  //     }
  //   }

  //   for (int jcol = 0; jcol < nc; ++jcol) {
  //     int jr = get_lindex(nc, ncol, jcol);
  //     for (int irow = 0; irow < nrow; ++irow) {
  //       col_vec[irow] = work2d[mgard_common::get_index(ncol, irow, jr)];
  //     }

  //     // mgard_gen::solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow);
  //     for (int irow = 0; irow < nrow; ++irow) {
  //       work2d[mgard_common::get_index(ncol, irow, jr)] = col_vec[irow];
  //     }
  //   }
  // }
  //   mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kfib);
  // }

  // // print_matrix(6, 6, 6, work.data(), nfib, ncol);
  // // bool r = compare_matrix(nrow, ncol, nfib,  
  // //                work.data(), nfib, ncol, 
  // //                work2, nfib, ncol );
  // // std::cout <<"pass:"<< r << "\n";

  // for (int irow = 0; irow < nr; irow += stride) {
  //   int ir = get_lindex(nr, nrow, irow);
  //   for (int jcol = 0; jcol < nc; jcol += stride) {
  //     int jc = get_lindex(nc, ncol, jcol);
  //     for (int kfib = 0; kfib < nfib; ++kfib) {
  //       fib_vec[kfib] =
  //           work[mgard_common::get_index3(ncol, nfib, ir, jc, kfib)];
  //     }
  //     // mgard_cannon::mass_matrix_multiply(l, fib_vec, coords_z);
  //     // mgard_gen::restriction_first(fib_vec, coords_z, nf, nfib);
  //     // mgard_gen::solve_tridiag_M_l(l, fib_vec, coords_z, nf, nfib);
  //     for (int kfib = 0; kfib < nfib; ++kfib) {
  //       work[mgard_common::get_index3(ncol, nfib, ir, jc, kfib)] =
  //           fib_vec[kfib];
  //     }
  //   }
  // }

  // std::cout << "after cpu restriction_first:\n";
  // print_matrix(6, 6, 6, work.data(), nfib, ncol);

  // add3_level_l(0, v, work.data(), nr, nc, nf, nrow, ncol, nfib);
}

template void 
prep_3D_cuda<double>(const int nrow, const int ncol, const int nfib, 
                     const int nr, const int nc, const int nf,
                     int * dirow, int * dicol, int * difib,
                     int * dirowP, int * dicolP, int * difibP,
                     int * dirowA, int * dicolA, int * difibA,
                     double * ddist_r, double * ddist_c, double * ddist_f,
                     double * dcoords_r, double * dcoords_c, double * dcoords_f, 
                     double * dv, int lddv1, int lddv2,
                     double * dwork, int lddwork1, int lddwork2,
                     int B, mgard_cuda_handle & handle, bool profile);
template void 
prep_3D_cuda<float>(const int nrow, const int ncol, const int nfib, 
                     const int nr, const int nc, const int nf,
                     int * dirow, int * dicol, int * difib,
                     int * dirowP, int * dicolP, int * difibP,
                     int * dirowA, int * dicolA, int * difibA,
                     float * ddist_r, float * ddist_c, float * ddist_f,
                     float * dcoords_r, float * dcoords_c, float * dcoords_f, 
                     float * dv, int lddv1, int lddv2,
                     float * dwork, int lddwork1, int lddwork2,
                     int B, mgard_cuda_handle & handle, bool profile);



template <typename T>
mgard_cuda_ret 
refactor_3D_cuda_cpt_l2_sm(int l_target,
                           int nrow,           int ncol,           int nfib,
                           int nr,             int nc,             int nf,             
                           int * dirow,        int * dicol,        int * difib,
                           T * dcoords_r, T * dcoords_c, T * dcoords_f,
                           T * dv,        int lddv1,          int lddv2,
                           T * dwork,     int lddwork1,       int lddwork2,
                           int B, mgard_cuda_handle & handle, bool profile) {
  // printf("refactor_3D_cuda_cpt_l2_sm\n");
  //for debug
    
  // T * v = new T[nrow*ncol*nfib];
  // std::vector<T> work(nrow * ncol * nfib),work2d(nrow * ncol * nfib);
  // std::vector<T> v2d(nrow * ncol), fib_vec(nfib);
  // std::vector<T> row_vec(ncol);
  // std::vector<T> col_vec(nrow);
  // cudaMemcpy3DAsyncHelper(v, nfib  * sizeof(T), nfib * sizeof(T), ncol,
  //                    dv, lddv1 * sizeof(T), nfib * sizeof(T), ncol,
  //                    nfib * sizeof(T), ncol, nrow,
  //                    D2H, handle, 0, profile);
  // std::vector<T> coords_r(nrow), coords_c(ncol), coords_f(nfib);

  // cudaMemcpyAsyncHelper(coords_r.data(), dcoords_r, nrow * sizeof(T), D2H, handle, 0, profile);
  // cudaMemcpyAsyncHelper(coords_c.data(), dcoords_c, ncol * sizeof(T), D2H, handle, 0, profile);
  // cudaMemcpyAsyncHelper(coords_f.data(), dcoords_f, nfib * sizeof(T), D2H, handle, 0, profile);

  // T * dv2;
  // size_t dv2_pitch;
  // cudaMalloc3DHelper((void**)&dv2, &dv2_pitch, nfib * sizeof(T), ncol, nrow);
  // int lddv21 = dv2_pitch / sizeof(T);
  // int lddv22 = ncol;

  // cudaMemcpy3DAsyncHelper(dv2, lddv21  * sizeof(T), nfib * sizeof(T), lddv22,
  //                    dv, lddv1 * sizeof(T), nfib * sizeof(T), ncol,
  //                    nfib * sizeof(T), ncol, nrow,
  //                    D2D, handle, 0, profile);



  // T * dwork2;
  // size_t dwork2_pitch;
  // cudaMalloc3DHelper((void**)&dwork2, &dwork2_pitch, nfib * sizeof(T), ncol, nrow);
  // int lddwork21 = dwork2_pitch / sizeof(T);
  // int lddwork22 = ncol;

  // T * dcwork2;
  // size_t dcwork2_pitch;
  // cudaMalloc3DHelper((void**)&dcwork2, &dcwork2_pitch, nf * sizeof(T), nc, nr);
  // int lddcwork21 = dcwork2_pitch / sizeof(T);
  // int lddcwork22 = nc;
  //end for debug


  T * dcv;
  size_t dcv_pitch;
  cudaMalloc3DHelper((void**)&dcv, &dcv_pitch, nf * sizeof(T), nc, nr);
  int lddcv1 = dcv_pitch / sizeof(T);
  int lddcv2 = nc;

  T * dcwork;
  size_t dcwork_pitch;
  cudaMalloc3DHelper((void**)&dcwork, &dcwork_pitch, nf * sizeof(T), nc, nr);
  int lddcwork1 = dcwork_pitch / sizeof(T);
  int lddcwork2 = nc;

  T ** dcwork_2d_rc = new T*[handle.num_of_queues];
  int * lddcwork_2d_rc = new int[handle.num_of_queues];
  for (int i = 0; i < handle.num_of_queues; i++) {
    size_t dcwork_2d_rc_pitch;
    cudaMallocPitchHelper((void**)&dcwork_2d_rc[i], &dcwork_2d_rc_pitch, nc * sizeof(T), nr);
    lddcwork_2d_rc[i] = dcwork_2d_rc_pitch / sizeof(T);
  }

  T ** dcwork_2d_cf = new T*[handle.num_of_queues];
  int * lddcwork_2d_cf = new int[handle.num_of_queues];
  for (int i = 0; i < handle.num_of_queues; i++) {
    size_t dcwork_2d_cf_pitch;
    cudaMallocPitchHelper((void**)&dcwork_2d_cf[i], &dcwork_2d_cf_pitch, nf * sizeof(T), nc);
    lddcwork_2d_cf[i] = dcwork_2d_cf_pitch / sizeof(T);
  }




  T * dccoords_r;
  cudaMallocHelper((void**)&dccoords_r, nr * sizeof(T));
  T * dccoords_c;
  cudaMallocHelper((void**)&dccoords_c, nc * sizeof(T));
  T * dccoords_f;
  cudaMallocHelper((void**)&dccoords_f, nf * sizeof(T));

  org_to_pow2p1(nrow, nr, dirow, dcoords_r, dccoords_r,
                B, handle, 0, profile);
  org_to_pow2p1(ncol, nc, dicol, dcoords_c, dccoords_c,
                B, handle, 0, profile);
  org_to_pow2p1(nfib, nf, difib, dcoords_f, dccoords_f,
                B, handle, 0, profile);
  

  int * nr_l = new int[l_target+1];
  int * nc_l = new int[l_target+1];
  int * nf_l = new int[l_target+1];

  T ** ddist_r_l = new T*[l_target+1];
  T ** ddist_c_l = new T*[l_target+1];
  T ** ddist_f_l = new T*[l_target+1];
  
  for (int l = 0; l < l_target+1; l++) {
    int stride = std::pow(2, l);

    nr_l[l] = ceil((float)nr/std::pow(2, l));
    nc_l[l] = ceil((float)nc/std::pow(2, l));
    nf_l[l] = ceil((float)nf/std::pow(2, l));

    cudaMallocHelper((void**)&ddist_r_l[l], nr_l[l] * sizeof(T));
    calc_cpt_dist(nr, stride, dccoords_r, ddist_r_l[l],
                  B, handle, 0, profile);

    cudaMallocHelper((void**)&ddist_c_l[l], nc_l[l] * sizeof(T));
    calc_cpt_dist(nc, stride, dccoords_c, ddist_c_l[l],
                  B, handle, 0, profile);

    cudaMallocHelper((void**)&ddist_f_l[l], nf_l[l] * sizeof(T));
    calc_cpt_dist(nf, stride, dccoords_f, ddist_f_l[l],
                  B, handle, 0, profile);
  }

  T ** am_row = new T*[handle.num_of_queues];
  T ** bm_row = new T*[handle.num_of_queues];
  T ** am_col = new T*[handle.num_of_queues];
  T ** bm_col = new T*[handle.num_of_queues];
  T ** am_fib = new T*[handle.num_of_queues];
  T ** bm_fib = new T*[handle.num_of_queues];
  for (int i = 0; i < handle.num_of_queues; i++) {
    cudaMallocHelper((void**)&am_row[i], nr*sizeof(T));
    cudaMallocHelper((void**)&bm_row[i], nr*sizeof(T));
    cudaMallocHelper((void**)&am_col[i], nc*sizeof(T));
    cudaMallocHelper((void**)&bm_col[i], nc*sizeof(T));
    cudaMallocHelper((void**)&am_fib[i], nf*sizeof(T));
    cudaMallocHelper((void**)&bm_fib[i], nf*sizeof(T));
  }


  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::chrono::duration<double> elapsed;

  int row_stride;
  int col_stride;
  int fib_stride;
  
  mgard_cuda_ret ret;
  double total_time = 0.0;
  std::ofstream timing_results;
  if (profile) {
    timing_results.open (handle.csv_prefix + "refactor_3D_cuda_cpt_l2_sm.csv");
  }

  double org_to_pow2p1_time = 0.0;
  double pow2p1_to_org_time = 0.0;

  double pow2p1_to_cpt_time = 0.0;
  double cpt_to_pow2p1_time = 0.0;

  double pi_Ql_cuda_cpt_sm_time = 0.0;
  double copy_level_l_cuda_cpt_time = 0.0;
  double assign_num_level_l_cuda_cpt_time = 0.0;

  double mass_mult_l_row_cuda_sm_time = 0.0;
  double restriction_l_row_cuda_sm_time = 0.0;
  double solve_tridiag_M_l_row_cuda_sm_time = 0.0;

  double mass_mult_l_col_cuda_sm_time = 0.0;
  double restriction_l_col_cuda_sm_time = 0.0;
  double solve_tridiag_M_l_col_cuda_sm_time = 0.0;

  double mass_mult_l_fib_cuda_sm_time = 0.0;
  double restriction_l_fib_cuda_sm_time = 0.0;
  double solve_tridiag_M_l_fib_cuda_sm_time = 0.0;

  double correction_calculation_fused_time = 0.0;

  double add_level_l_cuda_cpt_time = 0.0;

  ret = org_to_pow2p1(nrow,   ncol,  nfib,  
                      nr,     nc,    nf,
                      dirow,  dicol, difib,
                      dv,    lddv1,  lddv2,
                      dcv,   lddcv1, lddcv2,
                      B, handle, 0, profile);
  org_to_pow2p1_time = ret.time;

  // printf("start v:\n");
  // print_matrix_cuda(1,   1,  10,
  //                   dv,    lddv1,  lddv2,
  //                   nfib); 
  // printf("start cv:\n");
  // print_matrix_cuda(1,     1,    10,
  //                   dcv,   lddcv1, lddcv2, 
  //                   nf); 
  //std::cout << "l_target = " << l_target << std::endl;
  for (int l = 0; l < l_target; ++l) {
    //std::cout << "l = " << l << std::endl;
    int stride = std::pow(2, l);
    int Cstride = stride * 2;

    pow2p1_to_cpt_time = 0.0;
    cpt_to_pow2p1_time = 0.0;

    mass_mult_l_row_cuda_sm_time = 0.0;
    restriction_l_row_cuda_sm_time = 0.0;
    solve_tridiag_M_l_row_cuda_sm_time = 0.0;

    mass_mult_l_col_cuda_sm_time = 0.0;
    restriction_l_col_cuda_sm_time = 0.0;
    solve_tridiag_M_l_col_cuda_sm_time = 0.0;

    mass_mult_l_fib_cuda_sm_time = 0.0;
    restriction_l_fib_cuda_sm_time = 0.0;
    solve_tridiag_M_l_fib_cuda_sm_time = 0.0;
    // print_matrix(nfib,  nrow,   ncol,
    //              v,  ncol, nrow);


    // pi_Ql3D(nr, nc, nf, nrow,   ncol, nfib, l, v, coords_c, coords_r, coords_f,
    //         row_vec, col_vec, fib_vec);

    // mgard_gen::copy3_level_l(l, v, work.data(), nr, nc, nf, nrow, ncol, nfib);
    // mgard_gen::assign3_level_l(l + 1, work.data(), 0.0, nr, nc, nf, nrow, ncol,
    //                            nfib);

    // for (int kfib = 0; kfib < nf; kfib += stride) {
    //   //           int kf = kfib;
    //   int kf = get_lindex(nf, nfib,
    //                       kfib); // get the real location of logical index irow
    //   mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kf);
    //   mgard_gen::refactor_2D(nr, nc, nrow, ncol, l, v2d.data(), work2d,
    //                          coords_c, coords_r, row_vec, col_vec);
    //   mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kf);
    // }

    // for (int irow = 0; irow < nr; irow += Cstride) {
    //   int ir = get_lindex(nr, nrow, irow);
    //   for (int jcol = 0; jcol < nc; jcol += Cstride) {
    //     int jc = get_lindex(nc, ncol, jcol);
    //     for (int kfib = 0; kfib < nfib; ++kfib) {
    //       fib_vec[kfib] =
    //           work[mgard_common::get_index3(ncol, nfib, ir, jc, kfib)];
    //     }
    //     mgard_gen::mass_mult_l(l, fib_vec, coords_f, nf, nfib);
    //     mgard_gen::restriction_l(l + 1, fib_vec, coords_f, nf, nfib);
    //     mgard_gen::solve_tridiag_M_l(l + 1, fib_vec, coords_f, nf, nfib);
    //     for (int kfib = 0; kfib < nfib; ++kfib) {
    //       work[mgard_common::get_index3(ncol, nfib, ir, jc, kfib)] =
    //           fib_vec[kfib];
    //     }
    //   }
    // }

    // add3_level_l(l + 1, v, work.data(), nr, nc, nf, nrow, ncol, nfib);


    // print_matrix(nfib,  nrow,   ncol,
    //                     v,  ncol, nrow);





    // cudaMemcpy3DAsyncHelper(dwork2, lddwork21 * sizeof(double), nfib * sizeof(double), ncol,
    //                    v, nfib  * sizeof(double), nfib * sizeof(double), ncol,
    //                    nfib * sizeof(double), ncol, nrow,
    //                    H2D, handle, 0, profile);

    // cudaMemcpy3DHelper(dwork2, lddwork21 * sizeof(double), nfib * sizeof(double), ncol,
    //                    work.data(), nfib  * sizeof(double), nfib * sizeof(double), ncol,
    //                    nfib * sizeof(double), ncol, nrow,
    //                    H2D);

    // org_to_pow2p1(nrow,   ncol,  nfib,  
    //               nr,     nc,    nf,    
    //               dirow,  dicol, difib, 
    //               dwork2, lddwork21,  lddwork22,
    //               dcwork2, lddcwork21,  lddcwork22,
    //               B, handle, 0, profile);


    

    // print_matrix(nrow,   ncol, nfib,  
    //                     v,  nfib, ncol);

    
    fib_stride = stride;
    row_stride = stride;
    col_stride = stride;
    ret = pow2p1_to_cpt(nr,         nc,         nf,
                        row_stride, col_stride, fib_stride,
                        dcv,        lddcv1,     lddcv2,
                        dcwork,     lddcwork1,  lddcwork2,
                        B, handle, 0, profile);
    pow2p1_to_cpt_time += ret.time;

    fib_stride = 1;
    row_stride = 1;
    col_stride = 1;
    ret = pi_Ql_cuda_cpt_sm(nr_l[l],      nc_l[l],      nf_l[l],
                            row_stride,   col_stride,   fib_stride,
                            dcwork,       lddcwork1,    lddcwork2,
                            ddist_r_l[l], ddist_c_l[l], ddist_f_l[l],
                            B, handle, 0, profile);
    pi_Ql_cuda_cpt_sm_time = ret.time;

    fib_stride = stride;
    row_stride = stride;
    col_stride = stride;
    ret = cpt_to_pow2p1(nr,         nc,         nf,
                        row_stride, col_stride, fib_stride,
                        dcwork,     lddcwork1,  lddcwork2,
                        dcv,        lddcv1,     lddcv2,
                        B, handle, 0, profile);
    cpt_to_pow2p1_time += ret.time;    


  //   printf("pi_Ql cv:\n");
  // print_matrix_cuda(nr,     nc,    nf,
  //                   dcv,   lddcv1, lddcv2, 
  //                   nf); 

    fib_stride = stride;
    row_stride = stride;
    col_stride = stride;
    ret = copy_level_l_cuda_cpt(nr,         nc,         nf,
                                row_stride, col_stride, fib_stride,
                                dcv,        lddcv1,     lddcv2,
                                dwork,      lddwork1,   lddwork2,
                                B, handle, 0, profile);
    copy_level_l_cuda_cpt_time = ret.time;

    // fib_stride = Cstride;
    // row_stride = Cstride;
    // col_stride = Cstride;
    // ret = pow2p1_to_cpt(nr,         nc,         nf,
    //                     row_stride, col_stride, fib_stride,
    //                     dwork,      lddwork1,   lddwork2,
    //                     dcwork,     lddcwork1,  lddcwork2,
    //                     B, handle, 0, profile);
    // pow2p1_to_cpt_time += ret.time;

    fib_stride = Cstride;
    row_stride = Cstride;
    col_stride = Cstride;
    ret = assign_num_level_l_cuda_cpt(nr_l[0],  nc_l[0],  nf_l[0],
                                      row_stride, col_stride, fib_stride,
                                      dwork,     lddwork1,  lddwork2,
                                      (T)0.0, B, handle, 0, profile);
    assign_num_level_l_cuda_cpt_time = ret.time;

    // fib_stride = Cstride;
    // row_stride = Cstride;
    // col_stride = Cstride;
    // ret = cpt_to_pow2p1(nr,         nc,         nf,
    //                     row_stride, col_stride, fib_stride,
    //                     dcwork,     lddcwork1,  lddcwork2,
    //                     dwork,      lddwork1,   lddwork2,
    //                     B, handle, 0, profile);
    // cpt_to_pow2p1_time += ret.time;

    // std::cout << "gpu before:\n";
    // print_matrix_cuda(nr,           nc,    nf,      
    //                   dwork,        lddwork1,     lddwork2,  
    //                   nf); 
    bool local_profiling;
    if (handle.num_of_queues > 1) {
      local_profiling = false;
    } else {
      local_profiling = true;
    }
    start = std::chrono::high_resolution_clock::now();
    fib_stride = stride;
    for (int f = 0; f < nf; f += fib_stride) {
      int queue_idx = (f / fib_stride) % handle.num_of_queues;
      
      T * slice = dwork + f;
      int ldslice = lddwork2;

      row_stride = stride * lddwork1;
      col_stride = stride * lddwork1;
      ret = pow2p1_to_cpt(nr*lddwork1,    nc*lddwork1,
                          row_stride, col_stride,
                          slice,      ldslice,
                          dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                          B, handle, queue_idx, local_profiling);
      pow2p1_to_cpt_time += ret.time;

      // std::cout << "f before = " << f << "\n";
      // print_matrix_cuda(nr_l[l],    nc_l[l],
      //                   dcwork,     lddcwork1); 

      row_stride = 1;
      col_stride = 1;
      ret = mgard_2d::mgard_gen::mass_mult_l_row_cuda_sm(nr_l[l],    nc_l[l],
                                                        row_stride, col_stride,
                                                        dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                                                        ddist_c_l[l],
                                                        B, B, handle, queue_idx, local_profiling);
      mass_mult_l_row_cuda_sm_time += ret.time;

      // // std::cout << "f after= " << f << "\n";
      // // print_matrix_cuda(nr_l[l],    nc_l[l],
      // //                   dcwork,     lddcwork1); 

      row_stride = 1;
      col_stride = 1;
      ret = mgard_2d::mgard_gen::restriction_l_row_cuda_sm(nr_l[l],     nc_l[l],
                                      row_stride,  col_stride,
                                      dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                                      ddist_c_l[l],
                                      B, B, handle, queue_idx, local_profiling);
      restriction_l_row_cuda_sm_time += ret.time;

      row_stride = 1;
      col_stride = 2;//1;
      ret = mgard_2d::mgard_gen::solve_tridiag_M_l_row_cuda_sm(nr_l[l],    nc_l[l],
                                          row_stride, col_stride,
                                          dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                                          ddist_c_l[l+1],
                                          am_row[queue_idx], bm_row[queue_idx],
                                          B, B, handle, queue_idx, local_profiling);
      solve_tridiag_M_l_row_cuda_sm_time += ret.time;


      // std::cout << "f before = " << f << "\n";
      // print_matrix_cuda(nr_l[0],     nc_l[l],
      //                   dcwork,     lddcwork1); 


      row_stride = 1;
      col_stride = 2;
      ret = mgard_2d::mgard_gen::mass_mult_l_col_cuda_sm(nr_l[l],     nc_l[l],
                                   row_stride,  col_stride,
                                   dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                                   ddist_r_l[l],
                                   B, B, handle, queue_idx, local_profiling);
      mass_mult_l_col_cuda_sm_time += ret.time;

      // std::cout << "f after= " << f << "\n";
      // print_matrix_cuda(nr_l[0],     nc_l[l],
      //                   dcwork,     lddcwork1); 

      row_stride = 1;
      col_stride = 2;
      ret = mgard_2d::mgard_gen::restriction_l_col_cuda_sm(nr_l[l],         nc_l[l],
                                      row_stride, col_stride,
                                      dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                                      ddist_r_l[l],
                                      B, B, handle, queue_idx, local_profiling);
      restriction_l_col_cuda_sm_time += ret.time;

      row_stride = 2;
      col_stride = 2;
      ret = mgard_2d::mgard_gen::solve_tridiag_M_l_col_cuda_sm(nr_l[l],     nc_l[l],
                                          row_stride,    col_stride,
                                          dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                                          ddist_r_l[l+1],
                                          am_col[queue_idx], bm_col[queue_idx],
                                          B, B, handle, queue_idx, local_profiling);
      solve_tridiag_M_l_col_cuda_sm_time += ret.time;

      row_stride = stride * lddwork1;
      col_stride = stride * lddwork1;
      ret = cpt_to_pow2p1(nr*lddwork1,    nc*lddwork1,
                          row_stride, col_stride,
                          dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                          slice,      ldslice,
                          B, handle, queue_idx, local_profiling);
      cpt_to_pow2p1_time += ret.time;
      //handle.sync_all();
    }

    handle.sync_all();
    // // // std::cout << "gpu after:\n";
    // // // print_matrix_cuda(nr,           nc,    nf,      
    // // //                   dwork,        lddwork1,     lddwork2,  
    // // //                   nf); 

    //local_profiling = true;

    row_stride = Cstride;
    for (int r = 0; r < nr; r += row_stride) {
      int queue_idx = (r / row_stride) % handle.num_of_queues;
      T * slice = dwork + r * lddwork1 * lddwork2;
      int ldslice = lddwork1;

      col_stride = Cstride;
      fib_stride = stride;
      ret = pow2p1_to_cpt(nc,         nf,
                          col_stride, fib_stride,
                          slice,      ldslice,
                          dcwork_2d_cf[queue_idx], lddcwork_2d_cf[queue_idx],
                          B, handle, queue_idx, local_profiling);
      pow2p1_to_cpt_time += ret.time;

      col_stride = 1;
      fib_stride = 1;
      ret = mgard_2d::mgard_gen::mass_mult_l_row_cuda_sm(nc_l[l+1],    nf_l[l],
                                                        col_stride, fib_stride,
                                                        dcwork_2d_cf[queue_idx], lddcwork_2d_cf[queue_idx],
                                                        ddist_f_l[l],
                                                        B, B, handle, queue_idx, local_profiling);
      mass_mult_l_fib_cuda_sm_time += ret.time;

      col_stride = 1;
      fib_stride = 1;
      ret = mgard_2d::mgard_gen::restriction_l_row_cuda_sm(nc_l[l+1],    nf_l[l],
                                                           col_stride, fib_stride,
                                                           dcwork_2d_cf[queue_idx], lddcwork_2d_cf[queue_idx],
                                                           ddist_f_l[l],
                                                           B, B, handle, queue_idx, local_profiling);
      restriction_l_fib_cuda_sm_time += ret.time;

      col_stride = 1;
      fib_stride = 2;
      ret = mgard_2d::mgard_gen::solve_tridiag_M_l_row_cuda_sm(nc_l[l+1],    nf_l[l],
                                                               col_stride, fib_stride,
                                                               dcwork_2d_cf[queue_idx], lddcwork_2d_cf[queue_idx],
                                                               ddist_f_l[l+1],
                                                               am_fib[queue_idx], bm_fib[queue_idx],
                                                               B, B, handle, queue_idx, local_profiling);
      solve_tridiag_M_l_fib_cuda_sm_time += ret.time;



      col_stride = Cstride;
      fib_stride = stride;
      ret = cpt_to_pow2p1(nc,         nf,
                          col_stride, fib_stride,
                          dcwork_2d_cf[queue_idx], lddcwork_2d_cf[queue_idx],
                          slice,      ldslice,
                          B, handle, queue_idx, local_profiling);
      cpt_to_pow2p1_time += ret.time;


    }

    handle.sync_all();

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    correction_calculation_fused_time = elapsed.count();

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
                               dwork,      lddwork1,   lddwork2,
                               B, handle, 0, profile);
    add_level_l_cuda_cpt_time = ret.time;

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

    if (profile) {
      timing_results << l << ",pow2p1_to_cpt," << pow2p1_to_cpt_time << std::endl;
      timing_results << l << ",cpt_to_pow2p1," << cpt_to_pow2p1_time << std::endl;

      timing_results << l << ",pi_Ql," << pi_Ql_cuda_cpt_sm_time << std::endl;
      timing_results << l << ",copy_level_l," << copy_level_l_cuda_cpt_time << std::endl;
      timing_results << l << ",assign_num_level_l," << assign_num_level_l_cuda_cpt_time << std::endl;

      timing_results << l << ",mass_mult_l_row," << mass_mult_l_row_cuda_sm_time << std::endl;
      timing_results << l << ",restriction_l_row," << restriction_l_row_cuda_sm_time << std::endl;
      timing_results << l << ",solve_tridiag_M_l_row," << solve_tridiag_M_l_row_cuda_sm_time << std::endl;

      timing_results << l << ",mass_mult_l_col," << mass_mult_l_col_cuda_sm_time << std::endl;
      timing_results << l << ",restriction_l_col," << restriction_l_col_cuda_sm_time << std::endl;
      timing_results << l << ",solve_tridiag_M_l_col," << solve_tridiag_M_l_col_cuda_sm_time << std::endl;

      timing_results << l << ",mass_mult_l_fib," << mass_mult_l_fib_cuda_sm_time << std::endl;
      timing_results << l << ",restriction_l_fib," << restriction_l_fib_cuda_sm_time << std::endl;
      timing_results << l << ",solve_tridiag_M_l_fib," << solve_tridiag_M_l_fib_cuda_sm_time << std::endl;

      timing_results << l << ",correction_calculation_fused," << correction_calculation_fused_time << std::endl;

      timing_results << l << ",add_level_l," << add_level_l_cuda_cpt_time << std::endl;

      total_time += pow2p1_to_cpt_time;
      total_time += cpt_to_pow2p1_time;

      total_time += pi_Ql_cuda_cpt_sm_time;
      total_time += copy_level_l_cuda_cpt_time;
      total_time += assign_num_level_l_cuda_cpt_time;

      total_time += mass_mult_l_row_cuda_sm_time;
      total_time += restriction_l_row_cuda_sm_time;
      total_time += solve_tridiag_M_l_row_cuda_sm_time;

      total_time += mass_mult_l_col_cuda_sm_time;
      total_time += restriction_l_col_cuda_sm_time;
      total_time += solve_tridiag_M_l_col_cuda_sm_time;

      total_time += mass_mult_l_fib_cuda_sm_time;
      total_time += restriction_l_fib_cuda_sm_time;
      total_time += solve_tridiag_M_l_fib_cuda_sm_time;

      total_time += correction_calculation_fused_time;

      total_time += add_level_l_cuda_cpt_time;

    }

  } //end of loop

  // handle.sync_all();
  // compare_matrix_cuda(nr,         nc, nf,      
  //                     dcv,        lddcv1,     lddcv2,  nf,
  //                     dcwork2, lddcwork21,  lddcwork22,  nf);
  // handle.sync_all();
  // compare_matrix_cuda(nrow,   ncol,   nfib,  
  //                     dv,    lddv1, lddv2, nfib,
  //                     dv2,    lddv21, lddv22, nfib);


  // ret = pow2p1_to_org(nrow,   ncol,   nfib,  
  //                     nr,     nc,     nf,
  //                     dirow,  dicol,  difib,
  //                     dcwork2, lddcwork21,  lddcwork22,
  //                     dv2,     lddv21,  lddv22,
  //                     B, handle, 0, profile);

  ret = pow2p1_to_org(nrow,   ncol,   nfib,  
                      nr,     nc,     nf,
                      dirow,  dicol,  difib,
                      dcv,    lddcv1, lddcv2,
                      dv,     lddv1,  lddv2,
                      B, handle, 0, profile);
  pow2p1_to_org_time = ret.time;
  // handle.sync_all();
  // compare_matrix_cuda(nrow,   ncol,   nfib,  
  //                     dv,    lddv1, lddv2, nfib,
  //                     dv2,    lddv21, lddv22, nfib);

  if (profile) {
    timing_results << 0 << ",org_to_pow2p1," << org_to_pow2p1_time << std::endl;
    timing_results << 0 << ",pow2p1_to_org," << pow2p1_to_org_time << std::endl;
    timing_results.close();

    total_time += org_to_pow2p1_time;
    total_time += pow2p1_to_org_time;
  }

  for (int l = 0; l < l_target+1; l++) {
    cudaFreeHelper(ddist_r_l[l]);
    cudaFreeHelper(ddist_c_l[l]);
    cudaFreeHelper(ddist_f_l[l]);
  }

  cudaFreeHelper(dcv);
  cudaFreeHelper(dcwork);
  cudaFreeHelper(dccoords_r);
  cudaFreeHelper(dccoords_c);
  cudaFreeHelper(dccoords_f);

  for (int i = 0; i < handle.num_of_queues; i++) {
    cudaFreeHelper(am_row[i]);
    cudaFreeHelper(bm_row[i]);
    cudaFreeHelper(am_col[i]);
    cudaFreeHelper(bm_col[i]);
    cudaFreeHelper(am_fib[i]);
    cudaFreeHelper(bm_fib[i]);
    cudaFreeHelper(dcwork_2d_rc[i]);
    cudaFreeHelper(dcwork_2d_cf[i]);
  }

  return mgard_cuda_ret(0, total_time);

}

template mgard_cuda_ret 
refactor_3D_cuda_cpt_l2_sm<double>(int l_target,
                           int nrow,           int ncol,           int nfib,
                           int nr,             int nc,             int nf,             
                           int * dirow,        int * dicol,        int * difib,
                           double * dcoords_r, double * dcoords_c, double * dcoords_f,
                           double * dv,        int lddv1,          int lddv2,
                           double * dwork,     int lddwork1,       int lddwork2,
                           int B, mgard_cuda_handle & handle, bool profile);
template mgard_cuda_ret 
refactor_3D_cuda_cpt_l2_sm<float>(int l_target,
                           int nrow,           int ncol,           int nfib,
                           int nr,             int nc,             int nf,             
                           int * dirow,        int * dicol,        int * difib,
                           float * dcoords_r, float * dcoords_c, float * dcoords_f,
                           float * dv,        int lddv1,          int lddv2,
                           float * dwork,     int lddwork1,       int lddwork2,
                           int B, mgard_cuda_handle & handle, bool profile);

template <typename T> mgard_cuda_ret
recompose_3D_cuda_cpt_l2_sm(const int l_target,
                            const int nrow, const int ncol, const int nfib, 
                            const int nr, const int nc, const int nf,
                            int * dirow,        int * dicol,        int * difib,
                            T * dcoords_r, T * dcoords_c, T * dcoords_f,
                            T * dv,        int lddv1,          int lddv2,
                            T * dwork,     int lddwork1,       int lddwork2,
                            int B, mgard_cuda_handle & handle, bool profile){
                                 // double *v, 
                                 // std::vector<double> &work, std::vector<double> &work2d,
                                 // std::vector<double> &coords_x, std::vector<double> &coords_y,
                                 // std::vector<double> &coords_z) {
  // printf("recompose_3D_cuda_cpt_l2_sm\n");
  T * dcv;
  size_t dcv_pitch;
  cudaMalloc3DHelper((void**)&dcv, &dcv_pitch, nf * sizeof(T), nc, nr);
  int lddcv1 = dcv_pitch / sizeof(T);
  int lddcv2 = nc;

  T * dcwork;
  size_t dcwork_pitch;
  cudaMalloc3DHelper((void**)&dcwork, &dcwork_pitch, nf * sizeof(T), nc, nr);
  int lddcwork1 = dcwork_pitch / sizeof(T);
  int lddcwork2 = nc;

  T ** dcwork_2d_rc = new T*[handle.num_of_queues];
  int * lddcwork_2d_rc = new int[handle.num_of_queues];
  for (int i = 0; i < handle.num_of_queues; i++) {
    size_t dcwork_2d_rc_pitch;
    cudaMallocPitchHelper((void**)&dcwork_2d_rc[i], &dcwork_2d_rc_pitch, nc * sizeof(T), nr);
    lddcwork_2d_rc[i] = dcwork_2d_rc_pitch / sizeof(T);
  }

  T ** dcwork_2d_cf = new T*[handle.num_of_queues];
  int * lddcwork_2d_cf = new int[handle.num_of_queues];
  for (int i = 0; i < handle.num_of_queues; i++) {
    size_t dcwork_2d_cf_pitch;
    cudaMallocPitchHelper((void**)&dcwork_2d_cf[i], &dcwork_2d_cf_pitch, nf * sizeof(T), nc);
    lddcwork_2d_cf[i] = dcwork_2d_cf_pitch / sizeof(T);
  }

  // T * dwork2;
  // size_t dwork2_pitch;
  // cudaMalloc3DHelper((void**)&dwork2, &dwork2_pitch, nfib * sizeof(T), ncol, nrow);
  // int lddwork21 = dwork2_pitch / sizeof(T);
  // int lddwork22 = ncol;

  T * dccoords_r;
  cudaMallocHelper((void**)&dccoords_r, nr * sizeof(T));
  T * dccoords_c;
  cudaMallocHelper((void**)&dccoords_c, nc * sizeof(T));
  T * dccoords_f;
  cudaMallocHelper((void**)&dccoords_f, nf * sizeof(T));

  org_to_pow2p1(nrow, nr, dirow, dcoords_r, dccoords_r,
                B, handle, 0, profile);
  org_to_pow2p1(ncol, nc, dicol, dcoords_c, dccoords_c,
                B, handle, 0, profile);
  org_to_pow2p1(nfib, nf, difib, dcoords_f, dccoords_f,
                B, handle, 0, profile);
  

  int * nr_l = new int[l_target+1];
  int * nc_l = new int[l_target+1];
  int * nf_l = new int[l_target+1];

  T ** ddist_r_l = new T*[l_target+1];
  T ** ddist_c_l = new T*[l_target+1];
  T ** ddist_f_l = new T*[l_target+1];
  
  for (int l = 0; l < l_target+1; l++) {
    int stride = std::pow(2, l);

    nr_l[l] = ceil((float)nr/std::pow(2, l));
    nc_l[l] = ceil((float)nc/std::pow(2, l));
    nf_l[l] = ceil((float)nf/std::pow(2, l));

    cudaMallocHelper((void**)&ddist_r_l[l], nr_l[l] * sizeof(T));
    calc_cpt_dist(nr, stride, dccoords_r, ddist_r_l[l],
                  B, handle, 0, profile);

    cudaMallocHelper((void**)&ddist_c_l[l], nc_l[l] * sizeof(T));
    calc_cpt_dist(nc, stride, dccoords_c, ddist_c_l[l],
                  B, handle, 0, profile);

    cudaMallocHelper((void**)&ddist_f_l[l], nf_l[l] * sizeof(T));
    calc_cpt_dist(nf, stride, dccoords_f, ddist_f_l[l],
                  B, handle, 0, profile);
  }

  T ** am_row = new T*[handle.num_of_queues];
  T ** bm_row = new T*[handle.num_of_queues];
  T ** am_col = new T*[handle.num_of_queues];
  T ** bm_col = new T*[handle.num_of_queues];
  T ** am_fib = new T*[handle.num_of_queues];
  T ** bm_fib = new T*[handle.num_of_queues];
  for (int i = 0; i < handle.num_of_queues; i++) {
    cudaMallocHelper((void**)&am_row[i], nr*sizeof(T));
    cudaMallocHelper((void**)&bm_row[i], nr*sizeof(T));
    cudaMallocHelper((void**)&am_col[i], nc*sizeof(T));
    cudaMallocHelper((void**)&bm_col[i], nc*sizeof(T));
    cudaMallocHelper((void**)&am_fib[i], nf*sizeof(T));
    cudaMallocHelper((void**)&bm_fib[i], nf*sizeof(T));
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::chrono::duration<double> elapsed;

  int row_stride;
  int col_stride;
  int fib_stride;
  
  mgard_cuda_ret ret;
  double total_time = 0.0;
  std::ofstream timing_results;
  if (profile) {
    timing_results.open (handle.csv_prefix + "recompose_3D_cuda_cpt_l2_sm.csv");
  }

  double org_to_pow2p1_time = 0.0;
  double pow2p1_to_org_time = 0.0;

  double pow2p1_to_cpt_time = 0.0;
  double cpt_to_pow2p1_time = 0.0;

  double copy_level_l_cuda_cpt_time = 0.0;
  double assign_num_level_l_cuda_cpt_time = 0.0;

  double mass_mult_l_row_cuda_sm_time = 0.0;
  double restriction_l_row_cuda_sm_time = 0.0;
  double solve_tridiag_M_l_row_cuda_sm_time = 0.0;

  double mass_mult_l_col_cuda_sm_time = 0.0;
  double restriction_l_col_cuda_sm_time = 0.0;
  double solve_tridiag_M_l_col_cuda_sm_time = 0.0;

  double mass_mult_l_fib_cuda_sm_time = 0.0;
  double restriction_l_fib_cuda_sm_time = 0.0;
  double solve_tridiag_M_l_fib_cuda_sm_time = 0.0;

  double correction_calculation_fused_time = 0.0;

  double subtract_level_l_cuda_cpt_time = 0.0;

  double prolongate_l_row_cuda_sm_time = 0.0;
  double prolongate_l_col_cuda_sm_time = 0.0;
  double prolongate_l_fib_cuda_sm_time = 0.0;

  double prolongate_caculation_fused_time = 0.0;

  double assign_num_level_l_cuda_cpt_time2 = 0.0;
  double subtract_level_l_cuda_cpt_time2 = 0.0;
  

  // recompose

  // std::vector<double> v2d(nrow * ncol), fib_vec(nfib);
  // std::vector<double> row_vec(ncol);
  // std::vector<double> col_vec(nrow);

  ret = org_to_pow2p1(nrow,  ncol,   nfib,  
                      nr,    nc,     nf,    
                      dirow, dicol,  difib, 
                      dv,    lddv1,  lddv2,
                      dcv,   lddcv1, lddcv2,
                      B, handle, 0, profile);
  org_to_pow2p1_time += ret.time;


  //std::cout  << "recomposing" << "\n";
  for (int l = l_target; l > 0; --l) {

    int stride = std::pow(2, l); // current stride
    int Pstride = stride / 2;

    pow2p1_to_cpt_time = 0.0;
    cpt_to_pow2p1_time = 0.0;


    mass_mult_l_row_cuda_sm_time = 0.0;
    restriction_l_row_cuda_sm_time = 0.0;
    solve_tridiag_M_l_row_cuda_sm_time = 0.0;

    mass_mult_l_col_cuda_sm_time = 0.0;
    restriction_l_col_cuda_sm_time = 0.0;
    solve_tridiag_M_l_col_cuda_sm_time = 0.0;

    mass_mult_l_fib_cuda_sm_time = 0.0;
    restriction_l_fib_cuda_sm_time = 0.0;
    solve_tridiag_M_l_fib_cuda_sm_time = 0.0;

    prolongate_l_row_cuda_sm_time = 0.0;
    prolongate_l_col_cuda_sm_time = 0.0;
    prolongate_l_fib_cuda_sm_time = 0.0;

    //printf("l = %d, stride = %d, Pstride = %d\n", l, stride, Pstride);


    // cudaMemcpy3DAsyncHelper(dv, lddv1 * sizeof(T), nfib * sizeof(T), ncol,
    //                      v, nfib  * sizeof(T), nfib * sizeof(T), ncol,
    //                      nfib * sizeof(T), ncol, nrow,
    //                      H2D, handle, 0, profile);

    // org_to_pow2p1(nrow,  ncol,   nfib,  
    //             nr,    nc,     nf,    
    //             dirow, dicol,  difib, 
    //             dv,    lddv1,  lddv2,
    //             dcv,   lddcv1, lddcv2,
    //             B, handle, 0, profile);


    // mgard_gen::copy3_level_l(l - 1, v, work.data(), nr, nc, nf, nrow, ncol,
    //                          nfib);
    fib_stride = Pstride;
    row_stride = Pstride;
    col_stride = Pstride;
    ret = copy_level_l_cuda_cpt(nr,         nc,         nf,
                          row_stride, col_stride, fib_stride,
                          dcv,        lddcv1,     lddcv2,
                          dwork,      lddwork1,   lddwork2,
                          B, handle, 0, profile);
    copy_level_l_cuda_cpt_time = ret.time;

    // fib_stride = stride;
    // row_stride = stride;
    // col_stride = stride;
    // ret = pow2p1_to_cpt(nr,         nc,         nf,
    //                     row_stride, col_stride, fib_stride,
    //                     dwork,      lddwork1,   lddwork2,
    //                     dcwork,     lddcwork1,  lddcwork2,
    //                     B, handle, 0, profile);
    // pow2p1_to_cpt_time += ret.time;

    fib_stride = stride;
    row_stride = stride;
    col_stride = stride;
    ret = assign_num_level_l_cuda_cpt(nr_l[0],  nc_l[0],  nf_l[0],
                                      row_stride, col_stride, fib_stride,
                                      dwork,     lddwork1,  lddwork2,
                                      (T)0.0, B, handle, 0, profile);
    assign_num_level_l_cuda_cpt_time = ret.time;

    // fib_stride = stride;
    // row_stride = stride;
    // col_stride = stride;
    // ret = cpt_to_pow2p1(nr,         nc,         nf,
    //                     row_stride, col_stride, fib_stride,
    //                     dcwork,     lddcwork1,  lddcwork2,
    //                     dwork,      lddwork1,   lddwork2,
    //                     B, handle, 0, profile);
    // cpt_to_pow2p1_time += ret.time;



    // pow2p1_to_org(nrow,  ncol,   nfib,  
    //             nr,    nc,     nf,    
    //             dirow, dicol,  difib, 
    //             dwork,      lddwork1,   lddwork2,
    //             dwork2,      lddwork21,   lddwork22,
    //             B, handle, 0, profile);

    // cudaMemcpy3DAsyncHelper(work.data(), nfib  * sizeof(double), nfib * sizeof(double), ncol,
    //                         dwork2, lddwork21 * sizeof(double), nfib * sizeof(double), ncol, 
    //                         nfib * sizeof(double), ncol, nrow,
    //                         D2H, handle, 0, profile);


    // // mgard_gen::assign3_level_l(l, work.data(), 0.0, nr, nc, nf, nrow, ncol,
    // //                            nfib);
    


    // cudaMemcpy3DAsyncHelper(dwork2, lddwork21 * sizeof(double), nfib * sizeof(double), ncol,
    //                         work.data(), nfib  * sizeof(double), nfib * sizeof(double), ncol,
    //                         nfib * sizeof(double), ncol, nrow,
    //                         H2D, handle, 0, profile);


    // org_to_pow2p1(nrow,   ncol,  nfib,  
    //               nr,     nc,    nf,    
    //               dirow,  dicol, difib, 
    //               dwork2, lddwork21,  lddwork22,
    //               dwork, lddwork1,  lddwork2,
    //               B, handle, 0, profile);

    // print_matrix_cuda(nr,           nc,    nf,      
    //                   dwork,        lddwork1,     lddwork2,  
    //                   nf); 
    bool local_profiling;
    if (handle.num_of_queues > 1) {
      local_profiling = false;
    } else {
      local_profiling = true;
    }
    start = std::chrono::high_resolution_clock::now();

    fib_stride = Pstride;
    for (int f = 0; f < nf; f += fib_stride) {
      int queue_idx = (f / fib_stride) % handle.num_of_queues;
      T * slice = dwork + f;
      int ldslice = lddwork2;

      row_stride = Pstride * lddwork1;
      col_stride = Pstride * lddwork1;
      ret = pow2p1_to_cpt(nr*lddwork1,    nc*lddwork1,
                          row_stride, col_stride,
                          slice,      ldslice,
                          dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                          B, handle, queue_idx, local_profiling);
      pow2p1_to_cpt_time += ret.time;

      // std::cout << "f  = " << f << "\n";
      // print_matrix_cuda(nr_l[0],    nc_l[l-1],
      //                   dcwork,     lddcwork1); 


    //   // std::cout << "f before = " << f << "\n";
    //   // print_matrix_cuda(nr_l[l],    nc_l[l],
    //   //                   dcwork,     lddcwork1); 

      row_stride = 1;
      col_stride = 1;
      ret = mgard_2d::mgard_gen::mass_mult_l_row_cuda_sm(nr_l[l-1],    nc_l[l-1],
                                                        row_stride, col_stride,
                                                        dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                                                        ddist_c_l[l-1],
                                                        B, B, handle, queue_idx, local_profiling);
      mass_mult_l_row_cuda_sm_time += ret.time;

      // // std::cout << "f after= " << f << "\n";
      // // print_matrix_cuda(nr_l[l],    nc_l[l],
      // //                   dcwork,     lddcwork1); 

      row_stride = 1;
      col_stride = 1;
      ret = mgard_2d::mgard_gen::restriction_l_row_cuda_sm(nr_l[0],     nc_l[l-1],
                                      row_stride,  col_stride,
                                      dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                                      ddist_c_l[l-1],
                                      B, B, handle, queue_idx, local_profiling);
      restriction_l_row_cuda_sm_time += ret.time;

      row_stride = 1;
      col_stride = 2;//1;
      ret = mgard_2d::mgard_gen::solve_tridiag_M_l_row_cuda_sm(nr_l[0],    nc_l[l-1],
                                          row_stride, col_stride,
                                          dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                                          ddist_c_l[l],
                                          am_row[queue_idx], bm_row[queue_idx],
                                          B, B, handle, queue_idx, local_profiling);
      solve_tridiag_M_l_row_cuda_sm_time += ret.time;


      // std::cout << "f before = " << f << "\n";
      // print_matrix_cuda(nr_l[0],     nc_l[l],
      //                   dcwork,     lddcwork1); 


      row_stride = 1;
      col_stride = 2;
      ret = mgard_2d::mgard_gen::mass_mult_l_col_cuda_sm(nr_l[l-1],     nc_l[l-1],
                                   row_stride,  col_stride,
                                   dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                                   ddist_r_l[l-1],
                                   B, B, handle, queue_idx, local_profiling);
      mass_mult_l_col_cuda_sm_time += ret.time;

      // std::cout << "f after= " << f << "\n";
      // print_matrix_cuda(nr_l[0],     nc_l[l],
      //                   dcwork,     lddcwork1); 

      row_stride = 1;
      col_stride = 2;
      ret = mgard_2d::mgard_gen::restriction_l_col_cuda_sm(nr_l[l-1],         nc_l[l-1],
                                      row_stride, col_stride,
                                      dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                                      ddist_r_l[l-1],
                                      B, B, handle, queue_idx, local_profiling);
      restriction_l_col_cuda_sm_time += ret.time;

      row_stride = 2;
      col_stride = 2;
      ret = mgard_2d::mgard_gen::solve_tridiag_M_l_col_cuda_sm(nr_l[l-1],     nc_l[l-1],
                                          row_stride,    col_stride,
                                          dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                                          ddist_r_l[l],
                                          am_col[queue_idx], bm_col[queue_idx],
                                          B, B, handle, queue_idx, local_profiling);
      solve_tridiag_M_l_col_cuda_sm_time += ret.time;

      // row_stride = 1 * lddwork1;
      // col_stride = Pstride * lddwork1;
      // ret = cpt_to_pow2p1(nr*lddwork1,    nc*lddwork1,
      //                     row_stride, col_stride,
      //                     dcwork,     lddcwork1,
      //                     slice,      ldslice,
      //                     B, handle, 0, profile);
      // cpt_to_pow2p1_time += ret.time;

      row_stride = Pstride * lddwork1;
      col_stride = Pstride * lddwork1;
      ret = cpt_to_pow2p1(nr*lddwork1,    nc*lddwork1,
                          row_stride, col_stride,
                          dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                          slice,      ldslice,
                          B, handle, queue_idx, local_profiling);
      cpt_to_pow2p1_time += ret.time;


    }

    handle.sync_all();

    row_stride = stride;
    for (int r = 0; r < nr; r += row_stride) {
      int queue_idx = (r / row_stride) % handle.num_of_queues;
      T * slice = dwork + r * lddwork1 * lddwork2;
      int ldslice = lddwork1;

      col_stride = stride;
      fib_stride = Pstride;
      ret = pow2p1_to_cpt(nc,         nf,
                          col_stride, fib_stride,
                          slice,      ldslice,
                          dcwork_2d_cf[queue_idx], lddcwork_2d_cf[queue_idx],
                          B, handle, queue_idx, local_profiling);
      pow2p1_to_cpt_time += ret.time;

      col_stride = 1;
      fib_stride = 1;
      ret = mgard_2d::mgard_gen::mass_mult_l_row_cuda_sm(nc_l[l],    nf_l[l-1],
                                                        col_stride, fib_stride,
                                                        dcwork_2d_cf[queue_idx], lddcwork_2d_cf[queue_idx],
                                                        ddist_f_l[l-1],
                                                        B, B, handle, queue_idx, local_profiling);
      mass_mult_l_fib_cuda_sm_time += ret.time;

      col_stride = 1;
      fib_stride = 1;
      ret = mgard_2d::mgard_gen::restriction_l_row_cuda_sm(nc_l[l],    nf_l[l-1],
                                                           col_stride, fib_stride,
                                                           dcwork_2d_cf[queue_idx], lddcwork_2d_cf[queue_idx],
                                                           ddist_f_l[l-1],
                                                           B, B, handle, queue_idx, local_profiling);
      restriction_l_fib_cuda_sm_time += ret.time;

      col_stride = 1;
      fib_stride = 2;
      ret = mgard_2d::mgard_gen::solve_tridiag_M_l_row_cuda_sm(nc_l[l],    nf_l[l-1],
                                                               col_stride, fib_stride,
                                                               dcwork_2d_cf[queue_idx], lddcwork_2d_cf[queue_idx],
                                                               ddist_f_l[l],
                                                               am_fib[queue_idx], bm_fib[queue_idx],
                                                               B, B, handle, queue_idx, local_profiling);
      solve_tridiag_M_l_fib_cuda_sm_time += ret.time;



      col_stride = stride;
      fib_stride = Pstride;
      ret = cpt_to_pow2p1(nc,         nf,
                          col_stride, fib_stride,
                          dcwork_2d_cf[queue_idx], lddcwork_2d_cf[queue_idx],
                          slice,      ldslice,
                          B, handle, queue_idx, local_profiling);
      cpt_to_pow2p1_time += ret.time;


    }


    handle.sync_all();

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    correction_calculation_fused_time = elapsed.count();

    // print_matrix_cuda(nr,           nc,    nf,      
    //                   dwork,        lddwork1,     lddwork2,  
    //                   nf); 


    // pow2p1_to_org(nrow,   ncol,   nfib,  
    //                   nr,     nc,     nf,
    //                   dirow,  dicol,  difib,
    //                   dwork,  lddwork1,  lddwork2,
    //                   dwork2, lddwork21,  lddwork22,
    //                   B, handle, 0, profile);

    // cudaMemcpy3DAsyncHelper(work.data(), nfib  * sizeof(double), nfib * sizeof(double), ncol,
    //                         dwork2, lddwork21 * sizeof(double), nfib * sizeof(double), ncol, 
    //                         nfib * sizeof(double), ncol, nrow,
    //                         D2H, handle, 0, profile);
    // handle.sync_all();


    // for (int kfib = 0; kfib < nf; kfib += Pstride) {
    //   //    int kf =kfib;
    //   int kf = get_lindex(nf, nfib, kfib);
    //   mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kf);
    //   //            mgard_gen::compute_zl(nr, nc, nrow, ncol, l,  work2d,
    //   //            coords_x, coords_y, row_vec, col_vec);
    //   // mgard_gen::refactor_2D(nr, nc, nrow, ncol, l - 1, v2d.data(), work2d,
    //   //                        coords_x, coords_y, row_vec, col_vec);
    //   mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kf);
    // }

    // for (int irow = 0; irow < nr; irow += stride) {
    //   int ir = get_lindex(nr, nrow, irow);
    //   for (int jcol = 0; jcol < nc; jcol += stride) {
    //     int jc = get_lindex(nc, ncol, jcol);
    //     for (int kfib = 0; kfib < nfib; ++kfib) {
    //       fib_vec[kfib] =
    //           work[mgard_common::get_index3(ncol, nfib, ir, jc, kfib)];
    //     }

    //     // mgard_gen::mass_mult_l(l - 1, fib_vec, coords_z, nf, nfib);

    //     // mgard_gen::restriction_l(l, fib_vec, coords_z, nf, nfib);

    //     // mgard_gen::solve_tridiag_M_l(l, fib_vec, coords_z, nf, nfib);

    //     for (int kfib = 0; kfib < nfib; ++kfib) {
    //       work[mgard_common::get_index3(ncol, nfib, ir, jc, kfib)] =
    //           fib_vec[kfib];
    //     }
    //   }
    // }

    //- computed zl -//

    // sub3_level_l(l, work.data(), v, nr, nc, nf, nrow, ncol,
    //              nfib); // do -(Qu - zl)

    fib_stride = stride;
    row_stride = stride;
    col_stride = stride;
    ret = subtract_level_l_cuda_cpt(nr,         nc,         nf,
                                   row_stride, col_stride, fib_stride,
                                   dwork,      lddwork1,   lddwork2,
                                   dcv,        lddcv1,     lddcv2,
                                   B, handle, 0, profile);
    subtract_level_l_cuda_cpt_time = ret.time;


    // cudaMemcpy3DAsyncHelper(dwork2, lddwork21 * sizeof(double), nfib * sizeof(double), ncol,
    //                         work.data(), nfib  * sizeof(double), nfib * sizeof(double), ncol,
    //                         nfib * sizeof(double), ncol, nrow,
    //                         H2D, handle, 0, profile);


    // org_to_pow2p1(nrow,   ncol,  nfib,  
    //               nr,     nc,    nf,    
    //               dirow,  dicol, difib, 
    //               dwork2, lddwork21,  lddwork22,
    //               dwork, lddwork1,  lddwork2,
    //               B, handle, 0, profile);
    handle.sync_all();
    start = std::chrono::high_resolution_clock::now();
    fib_stride = stride;
    for (int f = 0; f < nf; f += fib_stride) {
      int queue_idx = (f / fib_stride) % handle.num_of_queues;
      T * slice = dwork + f;
      int ldslice = lddwork2;

      row_stride = Pstride * lddwork1;
      col_stride = Pstride * lddwork1;
      ret = pow2p1_to_cpt(nr*lddwork1,    nc*lddwork1,
                          row_stride, col_stride,
                          slice,      ldslice,
                          dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                          B, handle, queue_idx, local_profiling);
      pow2p1_to_cpt_time += ret.time;

      row_stride = 2;
      col_stride = 1;
      ret = mgard_2d::mgard_gen::prolongate_l_row_cuda_sm(nr_l[l-1],    nc_l[l-1],
                                     row_stride, col_stride,
                                     dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                                     ddist_c_l[l-1],
                                     B,
                                     handle, queue_idx, local_profiling);
      prolongate_l_row_cuda_sm_time += ret.time;

      row_stride = 1;
      col_stride = 1;
      ret = mgard_2d::mgard_gen::prolongate_l_col_cuda_sm(nr_l[l-1],  nc_l[l-1],
                                     row_stride, col_stride,
                                     dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                                     ddist_r_l[l-1],
                                     B,
                                     handle, queue_idx, local_profiling);
      prolongate_l_col_cuda_sm_time += ret.time;


      row_stride = Pstride * lddwork1;
      col_stride = Pstride * lddwork1;
      ret = cpt_to_pow2p1(nr*lddwork1,    nc*lddwork1,
                          row_stride, col_stride,
                          dcwork_2d_rc[queue_idx], lddcwork_2d_rc[queue_idx],
                          slice,      ldslice,
                          B, handle, queue_idx, local_profiling);
      cpt_to_pow2p1_time += ret.time;


    }

    handle.sync_all();

    row_stride = Pstride;
    for (int r = 0; r < nr; r += row_stride) {
      int queue_idx = (r / row_stride) % handle.num_of_queues;
      T * slice = dwork + r * lddwork1 * lddwork2;
      int ldslice = lddwork1;

      col_stride = Pstride;
      fib_stride = Pstride;
      ret = pow2p1_to_cpt(nc,         nf,
                          col_stride, fib_stride,
                          slice,      ldslice,
                          dcwork_2d_cf[queue_idx], lddcwork_2d_cf[queue_idx],
                          B, handle, queue_idx, local_profiling);
      pow2p1_to_cpt_time += ret.time;


      col_stride = 1;
      fib_stride = 1;
      ret = mgard_2d::mgard_gen::prolongate_l_row_cuda_sm(nc_l[l-1],    nf_l[l-1],
                                                           col_stride, fib_stride,
                                                           dcwork_2d_cf[queue_idx], lddcwork_2d_cf[queue_idx],
                                                           ddist_f_l[l-1],
                                                           B,
                                                           handle, queue_idx, local_profiling);
      prolongate_l_fib_cuda_sm_time += ret.time;

      col_stride = Pstride;
      fib_stride = Pstride;
      ret = cpt_to_pow2p1(nc,         nf,
                          col_stride, fib_stride,
                          dcwork_2d_cf[queue_idx], lddcwork_2d_cf[queue_idx],
                          slice,      ldslice,
                          B, handle, queue_idx, local_profiling);
      cpt_to_pow2p1_time += ret.time;


    }

    handle.sync_all();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    prolongate_caculation_fused_time = elapsed.count();

    // pow2p1_to_org(nrow,   ncol,   nfib,  
    //               nr,     nc,     nf,
    //               dirow,  dicol,  difib,
    //               dwork,  lddwork1,  lddwork2,
    //               dwork2, lddwork21,  lddwork22,
    //               B, handle, 0, profile);

    // cudaMemcpy3DAsyncHelper(work.data(), nfib  * sizeof(double), nfib * sizeof(double), ncol,
    //                         dwork2, lddwork21 * sizeof(double), nfib * sizeof(double), ncol, 
    //                         nfib * sizeof(double), ncol, nrow,
    //                         D2H, handle, 0, profile);
    // handle.sync_all();



    // for (int kfib = 0; kfib < nf; kfib += stride) {
    //   int kf = get_lindex(nf, nfib, kfib);
    //   //            int kf = kfib;
    //   mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kf);
    //   // mgard_gen::prolong_add_2D(nr, nc, nrow, ncol, l, work2d, coords_x,
    //   //                           coords_y, row_vec, col_vec);
    //   for (int irow = 0; irow < nr; irow += stride) {
    //     int ir = get_lindex(nr, nrow, irow);
    //     for (int jcol = 0; jcol < ncol; ++jcol) {
    //       row_vec[jcol] = work2d[mgard_common::get_index(ncol, ir, jcol)];
    //     }

    //     // mgard_gen::prolongate_l(l, row_vec, coords_x, nc, ncol);

    //     for (int jcol = 0; jcol < ncol; ++jcol) {
    //       work2d[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol];
    //     }
    //   }

    //   //   //std::cout  << "recomposing-colsweep2" << "\n";
    //   // column-sweep, this is the slow one! Need something like column_copy
    //   if (nrow > 1) {
    //     for (int jcol = 0; jcol < nc; jcol += Pstride) {
    //       int jr = get_lindex(nc, ncol, jcol);
    //       for (int irow = 0; irow < nrow; ++irow) // copy all rows
    //       {
    //         col_vec[irow] = work2d[mgard_common::get_index(ncol, irow, jr)];
    //       }

    //       // mgard_gen::prolongate_l(l, col_vec, coords_y, nr, nrow);

    //       for (int irow = 0; irow < nrow; ++irow) {
    //         work2d[mgard_common::get_index(ncol, irow, jr)] = col_vec[irow];
    //       }
    //     }
    //   }
  
    //   mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kf);
    // }

    // for (int irow = 0; irow < nr; irow += Pstride) {
    //   int ir = get_lindex(nr, nrow, irow);
    //   for (int jcol = 0; jcol < nc; jcol += Pstride) {
    //     int jc = get_lindex(nc, ncol, jcol);
    //     for (int kfib = 0; kfib < nfib; ++kfib) {
    //       fib_vec[kfib] =
    //           work[mgard_common::get_index3(ncol, nfib, ir, jc, kfib)];
    //     }

    //     // mgard_gen::prolongate_l(l, fib_vec, coords_z, nf, nfib);

    //     for (int kfib = 0; kfib < nfib; ++kfib) {
    //       work[mgard_common::get_index3(ncol, nfib, ir, jc, kfib)] =
    //           fib_vec[kfib];
    //     }
    //   }
    // }

    //mgard_gen::assign3_level_l(l, v, 0.0, nr, nc, nf, nrow, ncol, nfib);

    // fib_stride = stride;
    // row_stride = stride;
    // col_stride = stride;
    // ret = pow2p1_to_cpt(nr,         nc,         nf,
    //                     row_stride, col_stride, fib_stride,
    //                     dwork,      lddwork1,   lddwork2,
    //                     dcwork,     lddcwork1,  lddcwork2,
    //                     B, handle, 0, profile);
    // pow2p1_to_cpt_time += ret.time;

    fib_stride = stride;
    row_stride = stride;
    col_stride = stride;
    ret = assign_num_level_l_cuda_cpt(nr,  nc,  nf,
                                      row_stride, col_stride, fib_stride,
                                      dcv,     lddcv1,  lddcv2,
                                      (T)0.0, B, handle, 0, profile);
    assign_num_level_l_cuda_cpt_time2 = ret.time;
    // fib_stride = stride;
    // row_stride = stride;
    // col_stride = stride;
    // ret = cpt_to_pow2p1(nr,         nc,         nf,
    //                     row_stride, col_stride, fib_stride,
    //                     dcwork,     lddcwork1,  lddcwork2,
    //                     dwork,      lddwork1,   lddwork2,
    //                     B, handle, 0, profile);
    // cpt_to_pow2p1_time += ret.time;

    // mgard_gen::sub3_level_l(l - 1, v, work.data(), nr, nc, nf, nrow, ncol,
    //                         nfib);
    fib_stride = Pstride;
    row_stride = Pstride;
    col_stride = Pstride;
    ret = subtract_level_l_cuda_cpt(nr,         nc,         nf,
                                   row_stride, col_stride, fib_stride,
                                   dcv,        lddcv1,     lddcv2,
                                   dwork,      lddwork1,   lddwork2,
                                   B, handle, 0, profile);
    subtract_level_l_cuda_cpt_time2 = ret.time;

    // pow2p1_to_org(nrow,  ncol,   nfib,  
    //             nr,    nc,     nf,    
    //             dirow, dicol,  difib, 
    //             dcv,   lddcv1, lddcv2,
    //             dv,    lddv1,  lddv2,
    //             B, handle, 0, profile);

    // cudaMemcpy3DAsyncHelper(
    //                      v, nfib  * sizeof(T), nfib * sizeof(T), ncol,
    //                      dv, lddv1 * sizeof(T), nfib * sizeof(T), ncol,
    //                      nfib * sizeof(T), ncol, nrow,
    //                      D2H, handle, 0, profile);

    if (profile) {
      timing_results << l << ",pow2p1_to_cpt," << pow2p1_to_cpt_time << std::endl;
      timing_results << l << ",cpt_to_pow2p1," << cpt_to_pow2p1_time << std::endl;

      timing_results << l << ",copy_level_l," << copy_level_l_cuda_cpt_time << std::endl;
      timing_results << l << ",assign_num_level_l," << assign_num_level_l_cuda_cpt_time << std::endl;

      timing_results << l << ",mass_mult_l_row," << mass_mult_l_row_cuda_sm_time << std::endl;
      timing_results << l << ",restriction_l_row," << restriction_l_row_cuda_sm_time << std::endl;
      timing_results << l << ",solve_tridiag_M_l_row," << solve_tridiag_M_l_row_cuda_sm_time << std::endl;

      timing_results << l << ",mass_mult_l_col," << mass_mult_l_col_cuda_sm_time << std::endl;
      timing_results << l << ",restriction_l_col," << restriction_l_col_cuda_sm_time << std::endl;
      timing_results << l << ",solve_tridiag_M_l_col," << solve_tridiag_M_l_col_cuda_sm_time << std::endl;

      timing_results << l << ",mass_mult_l_fib," << mass_mult_l_fib_cuda_sm_time << std::endl;
      timing_results << l << ",restriction_l_fib," << restriction_l_fib_cuda_sm_time << std::endl;
      timing_results << l << ",solve_tridiag_M_l_fib," << solve_tridiag_M_l_fib_cuda_sm_time << std::endl;

      timing_results << l << ",correction_calculation_fused," << correction_calculation_fused_time << std::endl;

      timing_results << l << ",subtract_level_l," << subtract_level_l_cuda_cpt_time << std::endl;

      timing_results << l << ",prolongate_l_row," << prolongate_l_row_cuda_sm_time << std::endl;
      timing_results << l << ",prolongate_l_col," << prolongate_l_col_cuda_sm_time << std::endl;
      timing_results << l << ",prolongate_l_fib," << prolongate_l_fib_cuda_sm_time << std::endl;

      timing_results << l << ",prolongate_caculation_fused," << prolongate_caculation_fused_time << std::endl;


      timing_results << l << ",assign_num_level_l," << assign_num_level_l_cuda_cpt_time2 << std::endl;
      timing_results << l << ",subtract_level_l," << subtract_level_l_cuda_cpt_time2 << std::endl;

      total_time += pow2p1_to_cpt_time;
      total_time += cpt_to_pow2p1_time;

      total_time += copy_level_l_cuda_cpt_time;
      total_time += assign_num_level_l_cuda_cpt_time;

      total_time += mass_mult_l_row_cuda_sm_time;
      total_time += restriction_l_row_cuda_sm_time;
      total_time += solve_tridiag_M_l_row_cuda_sm_time;

      total_time += mass_mult_l_col_cuda_sm_time;
      total_time += restriction_l_col_cuda_sm_time;
      total_time += solve_tridiag_M_l_col_cuda_sm_time;

      total_time += mass_mult_l_fib_cuda_sm_time;
      total_time += restriction_l_fib_cuda_sm_time;
      total_time += solve_tridiag_M_l_fib_cuda_sm_time;

      total_time += correction_calculation_fused_time;

      total_time += subtract_level_l_cuda_cpt_time;

      total_time += prolongate_l_row_cuda_sm_time;
      total_time += prolongate_l_col_cuda_sm_time;
      total_time += prolongate_l_fib_cuda_sm_time;

      total_time += prolongate_caculation_fused_time;

      total_time += assign_num_level_l_cuda_cpt_time2;
      total_time += subtract_level_l_cuda_cpt_time2;

    }

  } //end of loop

  ret = pow2p1_to_org(nrow,   ncol,   nfib,  
                      nr,     nc,     nf,
                      dirow,  dicol,  difib,
                      dcv,    lddcv1, lddcv2,
                      dv,     lddv1,  lddv2,
                      B, handle, 0, profile);
  pow2p1_to_org_time = ret.time;

  if (profile) {
    timing_results << 0 << ",org_to_pow2p1," << org_to_pow2p1_time << std::endl;
    timing_results << 0 << ",pow2p1_to_org," << pow2p1_to_org_time << std::endl;
    timing_results.close();

    total_time += org_to_pow2p1_time;
    total_time += pow2p1_to_org_time;
  }

  for (int l = 0; l < l_target+1; l++) {
    cudaFreeHelper(ddist_r_l[l]);
    cudaFreeHelper(ddist_c_l[l]);
    cudaFreeHelper(ddist_f_l[l]);
  }

  cudaFreeHelper(dcv);
  cudaFreeHelper(dcwork);
  cudaFreeHelper(dccoords_r);
  cudaFreeHelper(dccoords_c);
  cudaFreeHelper(dccoords_f);

  for (int i = 0; i < handle.num_of_queues; i++) {
    cudaFreeHelper(am_row[i]);
    cudaFreeHelper(bm_row[i]);
    cudaFreeHelper(am_col[i]);
    cudaFreeHelper(bm_col[i]);
    cudaFreeHelper(am_fib[i]);
    cudaFreeHelper(bm_fib[i]);
  }

  return mgard_cuda_ret(0, total_time);

}

template mgard_cuda_ret 
recompose_3D_cuda_cpt_l2_sm<double>(const int l_target,
                                     const int nrow, const int ncol, const int nfib, 
                                     const int nr, const int nc, const int nf,
                                     int * dirow,        int * dicol,        int * difib,
                                     double * dcoords_r, double * dcoords_c, double * dcoords_f,
                                     double * dv,        int lddv1,          int lddv2,
                                     double * dwork,     int lddwork1,       int lddwork2,
                                     int B, mgard_cuda_handle & handle, bool profile);
                                     // double *v, 
                                     // std::vector<double> &work, std::vector<double> &work2d,
                                     // std::vector<double> &coords_x, std::vector<double> &coords_y,
                                     // std::vector<double> &coords_z);
template mgard_cuda_ret 
recompose_3D_cuda_cpt_l2_sm<float>(const int l_target,
                                     const int nrow, const int ncol, const int nfib, 
                                     const int nr, const int nc, const int nf,
                                     int * dirow,        int * dicol,        int * difib,
                                     float * dcoords_r, float * dcoords_c, float * dcoords_f,
                                     float * dv,        int lddv1,          int lddv2,
                                     float * dwork,     int lddwork1,       int lddwork2,
                                     int B, mgard_cuda_handle & handle, bool profile);

                                     // double *v, 
                                     // std::vector<double> &work, std::vector<double> &work2d,
                                     // std::vector<double> &coords_x, std::vector<double> &coords_y,
                                     // std::vector<double> &coords_z);

// template <typename T>
// void postp_3D_cuda(const int nr, const int nc, const int nf, const int nrow,
//               const int ncol, const int nfib, const int l_target, double *v,
//               std::vector<double> &work, std::vector<double> &coords_x,
//               std::vector<double> &coords_y, std::vector<double> &coords_z,
//               int * dirow, int * dicol, int * difib,
//              int * dirowP, int * dicolP, int * difibP,
//              int * dirowA, int * dicolA, int * difibA,
//              T * ddist_r, T * ddist_c, T * ddist_f,
//              T * dcoords_r, T * dcoords_c, T * dcoords_f, 
//              int B, mgard_cuda_handle & handle, bool profile ) {

template <typename T>
void postp_3D_cuda(const int nrow, const int ncol, const int nfib,
                   const int nr, const int nc, const int nf,
                   int * dirow, int * dicol, int * difib,
                   int * dirowP, int * dicolP, int * difibP,
                   int * dirowA, int * dicolA, int * difibA,
                   T * ddist_r, T * ddist_c, T * ddist_f,
                   T * dcoords_r, T * dcoords_c, T * dcoords_f, 
                   T * dv, int lddv1, int lddv2,
                   T * dwork, int lddwork1, int lddwork2,
                   int B, mgard_cuda_handle & handle, bool profile) {


  // std::vector<double> work2d(nrow * ncol), fib_vec(nfib), v2d(nrow * ncol);
  // std::vector<double> row_vec(ncol);
  // std::vector<double> col_vec(nrow);

  mgard_cuda_ret ret;
  // T * dv;
  // size_t dv_pitch;
  // cudaMalloc3DHelper((void**)&dv, &dv_pitch, nfib * sizeof(T), ncol, nrow);
  // int lddv1 = dv_pitch / sizeof(T);
  // int lddv2 = ncol;

  // T * dwork;
  // size_t dwork_pitch;
  // cudaMalloc3DHelper((void**)&dwork, &dwork_pitch, nfib * sizeof(T), ncol, nrow);
  // int lddwork1 = dwork_pitch / sizeof(T);
  // int lddwork2 = ncol;

  // cudaMemcpy3DAsyncHelper(dv, lddv1 * sizeof(T), nfib * sizeof(T), ncol,
  //                         v,  nfib  * sizeof(T), nfib * sizeof(T), ncol,
  //                         nfib * sizeof(T), ncol, nrow,
  //                         H2D, handle, 0, profile);

  mgard_gen::copy_level_l_cuda_cpt(nrow, ncol, nfib,
                        1, 1, 1,
                        dv, lddv1, lddv2,
                        dwork, lddwork1, lddwork2,
                        B, handle, 0, profile);

  assign_num_level_l_cuda(nrow,        ncol,         nfib,
                          nr,          nc,           nf, 
                          1, 1, 1,
                          dirow,       dicol,        difib,
                          dwork, lddwork1, lddwork2,
                          (T)0.0,
                          B, handle, 0, profile);

  for (int r = 0; r < nrow; r += 1) {
    T * dwork_local = dwork + r * lddwork1*lddwork2;
    mgard_2d::mgard_cannon::mass_matrix_multiply_col_cuda(ncol,  nfib, 
                                                          1, 1,
                                                          dwork_local,   lddwork1,
                                                          dcoords_c,
                                                          B, handle, 0, profile);
    mgard_2d::mgard_gen::restriction_first_col_cuda(ncol,  nfib,
                                                    nc,  nfib, 
                                                    1, 1, 
                                                    dicolP, difibA,
                                                    dwork_local,   lddwork1,
                                                    dcoords_c,
                                                    B, handle, 0, profile);
  }
  // print_matrix_cuda(6, 6, 6, dwork, lddwork1, lddwork2, nfib);

  for (int r = 0; r < nr; r += 1) {
    int ir = get_lindex(nr, nrow, r);
    T * dwork_local = dwork + ir * lddwork1*lddwork2;
    mgard_2d::mgard_gen::solve_tridiag_M_l_col_cuda(ncol,  nfib,
                                                    nc,    nfib, //nf,
                                                    1, 1,
                                                    dicol,      difibA,
                                                    dwork_local,   lddwork1,
                                                    dcoords_c,
                                                    B, handle, 0, profile);
  }
  // print_matrix_cuda(6, 6, 6, dwork, lddwork1, lddwork2, nfib);

  for (int c = 0; c < ncol; c += 1) {
    T * dwork_local = dwork + c * lddwork1;
    mgard_2d::mgard_cannon::mass_matrix_multiply_col_cuda(nrow,  nfib, 
                                                          1, 1,
                                                          dwork_local,   lddwork1*lddwork2,
                                                          dcoords_r,
                                                          B, handle, 0, profile);
    mgard_2d::mgard_gen::restriction_first_col_cuda(nrow,  nfib, 
                                                    nr,    nfib,
                                                    1, 1,
                                                    dirowP, difibA,
                                                    dwork_local,   lddwork1*lddwork2,
                                                    dcoords_r,
                                                    B, handle, 0, profile);
  }

  for (int c = 0; c < nc; c += 1) {
    int ic = get_lindex(nc, ncol, c);
    T * dwork_local = dwork + ic * lddwork1;
    mgard_2d::mgard_gen::solve_tridiag_M_l_col_cuda(nrow,  nfib,
                                                    nr,    nfib, //nf,
                                                    1, 1,
                                                    dirow,      difibA,
                                                    dwork_local,   lddwork1*lddwork2,
                                                    dcoords_r,
                                                    B, handle, 0, profile);
  }



  for (int r = 0; r < nr; r += 1) {
    int ir = get_lindex(nr, nrow, r);
    T * dwork_local = dwork + ir * lddwork1*lddwork2;
    mgard_2d::mgard_gen::mass_mult_l_row_cuda(ncol,       nfib,
                                             nc,         nfib,
                                             1, 1,
                                             dicol,    difibA,
                                             dwork_local,   lddwork1,
                                             dcoords_f,
                                             B, handle, 0, profile);
  }

  // std::cout << "after cuda mass matrix:\n";
  // print_matrix_cuda(6, 6, 6, dwork, lddwork1, lddwork2, nfib);

  // cudaMemcpy3DAsyncHelper(work.data(), nfib * sizeof(T), nfib * sizeof(T), ncol,
  //                         dwork, lddwork1 * sizeof(T), nfib * sizeof(T), ncol,
  //                         nfib * sizeof(T), ncol, nrow,
  //                         D2H, handle, 0, profile);


   for (int r = 0; r < nr; r += 1) {
    int ir = get_lindex(nr, nrow, r);
    T * dwork_local = dwork + ir * lddwork1*lddwork2;
    mgard_2d::mgard_gen::restriction_first_row_cuda(ncol,  nfib, 
                                                    nc,    nf,
                                                    1, 1,
                                                    dicol, difibP,
                                                    dwork_local,   lddwork1,
                                                    dcoords_f,
                                                    B, handle, 0, profile);
    mgard_2d::mgard_gen::solve_tridiag_M_l_row_cuda(ncol,  nfib, 
                                                    nc,    nf,
                                                    1, 1,
                                                    dicol,      difib,
                                                    dwork_local,   lddwork1,
                                                    dcoords_f,
                                                    B, handle, 0, profile); 
  }


  mgard_gen::subtract_level_l_cuda(nrow,       ncol, nfib,
                                   nr,         nc,         nf,
                                   1, 1, 1,
                                   dirow,    dicol, difib,
                                   dwork, lddwork1,   lddwork2,
                                   dv,    lddv1,      lddv2,
                                   B, handle, 0, profile); 

  ret = prolongate_last_fib(nrow,        ncol,         nfib,
                          nr,          nc,           nf, 
                          dirow,       dicol,        difibP,
                          dwork, lddwork1,   lddwork2,
                          ddist_r,     ddist_c,      ddist_f,
                          B, handle, 
                          0, profile);

  ret = prolongate_last_col(nrow,        ncol,         nfib,
                          nr,          nc,           nf, 
                          dirow,       dicolP,        difib,
                          dwork, lddwork1,   lddwork2,
                          ddist_r,     ddist_c,      ddist_f,
                          B, handle, 
                          0, profile);

  ret = prolongate_last_row(nrow,        ncol,         nfib,
                          nr,          nc,           nf, 
                          dirowP,      dicol,        difib,
                          dwork, lddwork1,   lddwork2,
                          ddist_r,     ddist_c,      ddist_f,
                          B, handle, 
                          0, profile);

  ret = prolongate_last_fib_col(nrow,        ncol,         nfib,
                              nr,          nc,           nf, 
                              dirow,       dicolP,        difibP,
                              dwork, lddwork1,   lddwork2,
                              ddist_r,     ddist_c,      ddist_f,
                              B, handle, 
                              0, profile);

  ret = prolongate_last_fib_row(nrow,        ncol,         nfib,
                              nr,          nc,           nf, 
                              dirowP,      dicol,        difibP,
                              dwork, lddwork1,   lddwork2,
                              ddist_r,     ddist_c,      ddist_f,
                              B, handle, 
                              0, profile);

  ret = prolongate_last_col_row(nrow,        ncol,         nfib,
                              nr,          nc,           nf, 
                              dirowP,      dicolP,       difib,
                              dwork, lddwork1,   lddwork2,
                              ddist_r,     ddist_c,      ddist_f,
                              B, handle, 
                              0, profile);

  ret = prolongate_last_fib_col_row(nrow,        ncol,         nfib,
                              nr,          nc,           nf, 
                              dirowP,      dicolP,       difibP,
                              dwork, lddwork1,   lddwork2,
                              ddist_r,     ddist_c,      ddist_f,
                              B, handle, 
                              0, profile);

  assign_num_level_l_cuda(nrow,        ncol,         nfib,
                          nr,          nc,           nf, 
                          1, 1, 1,
                          dirow,       dicol,        difib,
                          dv, lddv1, lddv2,
                          (T)0.0,
                          B, handle, 0, profile);

  // mgard_gen::subtract_level_l_cuda(nrow,       ncol, nfib,
  //                                  nr,         nc,         nf,
  //                                  1, 1, 1,
  //                                  dirow,    dicol, difib,
  //                                  dv,    lddv1,      lddv2,
  //                                  dwork, lddwork1,   lddwork2,
  //                                  B, handle, 0, profile); 

  subtract_level_l_cuda_cpt(nrow,       ncol, nfib,
                            1, 1, 1,
                            dv,    lddv1,      lddv2,
                            dwork, lddwork1,   lddwork2,
                            B, handle, 0, profile); 

  // cudaMemcpy3DAsyncHelper(work.data(), nfib * sizeof(T), nfib * sizeof(T), ncol,
  //                         dwork, lddwork1 * sizeof(T), nfib * sizeof(T), ncol,
  //                         nfib * sizeof(T), ncol, nrow,
  //                         D2H, handle, 0, profile);

  // cudaMemcpy3DAsyncHelper(v,  nfib  * sizeof(T), nfib * sizeof(T), ncol,
  //                         dv, lddv1 * sizeof(T), nfib * sizeof(T), ncol,
  //                         nfib * sizeof(T), ncol, nrow,
  //                         D2H, handle, 0, profile);

  int l = 0;
  int stride = 1; // current stride
  int Pstride = stride / 2;

  // mgard_gen::copy3_level_l(l,  v,  work.data(),  nrow,  ncol, nfib,  nrow,
  // ncol, nfib);
  // mgard_gen::copy3_level(l, v, work.data(), nrow, ncol, nfib);
  // mgard_gen::assign3_level_l(l, work.data(), 0.0, nr, nc, nf, nrow, ncol, nfib);


  // for (int kfib = 0; kfib < nfib; kfib += stride) {
  //   mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kfib);
  //   mgard_gen::refactor_2D_first(nr, nc, nrow, ncol, l, v2d.data(), work2d,
  //                                coords_x, coords_y, row_vec, col_vec);
  //   mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kfib);
  // }

  // for (int irow = 0; irow < nr; irow += stride) {
  //   int ir = get_lindex(nr, nrow, irow);
  //   for (int jcol = 0; jcol < nc; jcol += stride) {
  //     int jc = get_lindex(nc, ncol, jcol);
  //     for (int kfib = 0; kfib < nfib; ++kfib) {
  //       fib_vec[kfib] =
  //           work[mgard_common::get_index3(ncol, nfib, ir, jc, kfib)];
  //     }
  //     mgard_cannon::mass_matrix_multiply(l, fib_vec, coords_z);
  //     mgard_gen::restriction_first(fib_vec, coords_z, nf, nfib);
  //     mgard_gen::solve_tridiag_M_l(l, fib_vec, coords_z, nf, nfib);
  //     for (int kfib = 0; kfib < nfib; ++kfib) {
  //       work[mgard_common::get_index3(ncol, nfib, ir, jc, kfib)] =
  //           fib_vec[kfib];
  //     }
  //   }
  // }

  //- computed zl -//

  // sub3_level_l(0, work.data(), v, nr, nc, nf, nrow, ncol, nfib); // do -(Qu -
                                                                 // zl)

  //    for (int kf = 0; kf < nfib; ++kf)
  // for (int kfib = 0; kfib < nf; kfib += stride) {
  //   int kf = get_lindex(nf, nfib, kfib);

  //   mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kf);
  //   mgard_gen::prolong_add_2D_last(nr, nc, nrow, ncol, l, work2d, coords_x,
  //                                  coords_y, row_vec, col_vec);
  //   mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kf);
  // }

  // for (int irow = 0; irow < nrow; irow += stride) {
  //   for (int jcol = 0; jcol < ncol; jcol += stride) {

  //     for (int kfib = 0; kfib < nfib; ++kfib) {
  //       fib_vec[kfib] =
  //           work[mgard_common::get_index3(ncol, nfib, irow, jcol, kfib)];
  //     }

  //     mgard_gen::prolongate_last(fib_vec, coords_z, nf, nfib);

  //     for (int kfib = 0; kfib < nfib; ++kfib) {
  //       work[mgard_common::get_index3(ncol, nfib, irow, jcol, kfib)] =
  //           fib_vec[kfib];
  //     }
  //   }
  // }

  // mgard_gen::assign3_level_l(0, v, 0.0, nr, nc, nf, nrow, ncol, nfib);
  // mgard_gen::sub3_level(0, v, work.data(), nrow, ncol, nfib);
  //    mgard_gen::sub3_level(l, v, work.data(), nrow,  ncol,  nfib);
}


template void 
postp_3D_cuda<double>(const int nrow, const int ncol, const int nfib,
                   const int nr, const int nc, const int nf,
                   int * dirow, int * dicol, int * difib,
                   int * dirowP, int * dicolP, int * difibP,
                   int * dirowA, int * dicolA, int * difibA,
                   double * ddist_r, double * ddist_c, double * ddist_f,
                   double * dcoords_r, double * dcoords_c, double * dcoords_f, 
                   double * dv, int lddv1, int lddv2,
                   double * dwork, int lddwork1, int lddwork2,
                   int B, mgard_cuda_handle & handle, bool profile);

template void 
postp_3D_cuda<float>(const int nrow, const int ncol, const int nfib,
                   const int nr, const int nc, const int nf,
                   int * dirow, int * dicol, int * difib,
                   int * dirowP, int * dicolP, int * difibP,
                   int * dirowA, int * dicolA, int * difibA,
                   float * ddist_r, float * ddist_c, float * ddist_f,
                   float * dcoords_r, float * dcoords_c, float * dcoords_f, 
                   float * dv, int lddv1, int lddv2,
                   float * dwork, int lddwork1, int lddwork2,
                   int B, mgard_cuda_handle & handle, bool profile);

}