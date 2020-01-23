#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"
#include "mgard_nuni_2d_cuda_o1.h"
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

__global__ void
_compact_to_2k_plus_1(int nrow,     int ncol,
                      int nr,       int nc,
                      int * irow,   int * icol,
                      double * dv,  int lddv,
                      double * dcv, int lddcv) {
  
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;

  for (int y = y0; y < nr; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x) {
      dcv[get_idx(lddcv, y, x)] = dv[get_idx(lddv, irow[y], icol[x])];
    }
  }
}


mgard_cuda_ret 
compact_to_2k_plus_1(int nrow,     int ncol,
                     int nr,       int nc,
                     int * dirow,  int * dicol,
                     double * dv,  int lddv,
                     double * dcv, int lddcv) {

  int B = 16;
  int total_thread_y = nr;
  int total_thread_x = nc;
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "_copy_level_l_cuda" << std::endl;
  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _compact_to_2k_plus_1<<<blockPerGrid, threadsPerBlock>>>(nrow,  ncol,
                                                           nr,    nc,
                                                           dirow, dicol,
                                                           dv,    lddv,
                                                           dcv,   lddcv);

  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}


__global__ void
_restore_from_2k_plus_1(int nrow,     int ncol,
                        int nr,       int nc,
                        int * irow,   int * icol,
                        double * dcv,  int lddcv,
                        double * dv, int lddv) {
  
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;

  for (int y = y0; y < nr; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x) {
       dv[get_idx(lddv, irow[y], icol[x])] = dcv[get_idx(lddcv, y, x)];
    }
  }
}

mgard_cuda_ret 
restore_from_2k_plus_1(int nrow,     int ncol,
                       int nr,       int nc,
                       int * dirow,  int * dicol,
                       double * dcv,  int lddcv,
                       double * dv, int lddv) {

  int B = 16;
  int total_thread_y = nr;
  int total_thread_x = nc;
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "_copy_level_l_cuda" << std::endl;
  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _restore_from_2k_plus_1<<<blockPerGrid, threadsPerBlock>>>(nrow,  ncol,
                                                             nr,    nc,
                                                             dirow, dicol,
                                                             dcv,   lddcv,
                                                             dv,    lddv);

  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}


__global__ void 
_original_to_compacted_cuda(int nrow,           int ncol, 
                           int row_stride,      int col_stride,
                           double * dv,         int lddv, 
                           double * dcv,        int lddcv) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    int y0 = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("x = %d, y = %d, stride = %d \n", x,y,stride);
    for (int y = y0; y * row_stride < nrow; y += blockDim.y * gridDim.y) {
        for (int x = x0; x * col_stride < ncol; x += blockDim.x * gridDim.x) {
            int x_strided = x * col_stride;
            int y_strided = y * row_stride;
            //printf("thread1(%d, %d): v(%d, %d)\n", x0, y0, x_strided, y_strided);
            dcv[get_idx(lddcv, y, x)] = dv[get_idx(lddv, y_strided, x_strided)];
            //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
            //y += blockDim.y * gridDim.y * stride;
        }
        //x += blockDim.x * gridDim.x * stride;
    }
}

mgard_cuda_ret 
original_to_compacted_cuda(int nrow,      int ncol, 
                          int row_stride, int col_stride,
                          double * dv,    int lddv, 
                          double * dcv,   int lddcv) {
    int B = 16;
    int total_thread_y = ceil((float)nrow/row_stride);
    int total_thread_x = ceil((float)ncol/col_stride);
    int tby = min(B, total_thread_y);
    int tbx = min(B, total_thread_x);
    int gridy = ceil((float)total_thread_y/tby);
    int gridx = ceil((float)total_thread_x/tbx);
    dim3 threadsPerBlock(tbx, tby);
    dim3 blockPerGrid(gridx, gridy);

    //std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
    //std::cout << "grid: " << gridx << ", " << gridy <<std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    _original_to_compacted_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,      ncol,
                                                                  row_stride, col_stride,
                                                                  dv,         lddv, 
                                                                  dcv,        lddcv);
    gpuErrchk(cudaGetLastError ());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return mgard_cuda_ret(0, milliseconds/1000.0);

}

__global__ void 
_compacted_to_original_cuda(int nrow,     int ncol,
                          int row_stride, int col_stride,
                          double * dcv,   int lddcv,
                          double * dv,    int lddv) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    int y0 = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("x = %d, y = %d, stride = %d \n", x,y,stride);
    for (int y = y0; y * row_stride < nrow; y += blockDim.y * gridDim.y) {
        for (int x = x0; x * col_stride < ncol; x += blockDim.x * gridDim.x) {
            int x_strided = x * col_stride;
            int y_strided = y * row_stride;
            //printf("thread2(%d, %d): v(%d, %d)\n", x0, y0, x_strided, y_strided);
            dv[get_idx(lddv, y_strided, x_strided)] = dcv[get_idx(lddcv, y, x)];
            //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
            //y += blockDim.y * gridDim.y * stride;
        }
        //x += blockDim.x * gridDim.x * stride;
    }
}


mgard_cuda_ret
compacted_to_original_cuda(int nrow, int ncol, 
                              int row_stride, int col_stride, 
                              double * dcv, int lddcv,
                              double * dv, int lddv) {
    

    int B = 16;
    int total_thread_x = ceil((float)nrow/row_stride);
    int total_thread_y = ceil((float)ncol/col_stride);
    int tbx = min(B, total_thread_x);
    int tby = min(B, total_thread_y);
    int gridx = ceil((float)total_thread_x/tbx);
    int gridy = ceil((float)total_thread_y/tby);
    dim3 threadsPerBlock(tbx, tby);
    dim3 blockPerGrid(gridx, gridy);

    //std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
    //std::cout << "grid: " << gridx << ", " << gridy <<std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    _compacted_to_original_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow, ncol,
                                                                  row_stride, col_stride, 
                                                                  dcv, lddcv,
                                                                  dv, lddv);
    
    gpuErrchk(cudaGetLastError ());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return mgard_cuda_ret(0, milliseconds/1000.0);

}



void 
refactor_2D_cuda_o1(const int l_target,
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



  mgard_cuda_ret ret;

  double compact_to_2k_plus_1_time = 0.0;
  double restore_from_2k_plus_1_time = 0.0;

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

  // ret = compact_to_2k_plus_1(nrow,  ncol,
  //                      nr,    nc,
  //                      dirow, dicol,
  //                      dv,    lddv,
  //                      dcv,   lddcv);
  // compact_to_2k_plus_1_time = ret.time;


  

  // std::cout << "dv:" << std::endl;
  // print_matrix_cuda(nrow, ncol, dv,  lddv );
  
  // std::cout << "dcv:" << std::endl;
  // print_matrix_cuda(nr,   nc,   dcv, lddcv);

  ret = compact_to_2k_plus_1(nrow,  ncol,
                       nr,    nc,
                       dirow, dicol,
                       dv,    lddv,
                       dcv,   lddcv);
  compact_to_2k_plus_1_time = ret.time;



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
                     dccoords_x,      dccoords_y);
    pi_Ql_cuda_time += ret.time;

    

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
                            dcwork,      lddcwork);
    copy_level_l_cuda_time += ret.time;

    // assign_num_level_l(l + 1, work.data(), 0.0, nr, nc, nrow, ncol);
    row_stride = Cstride;
    col_stride = Cstride;
    ret = assign_num_level_l_cuda(nr,         nc,
                                  nr,         nc,
                                  row_stride, col_stride,
                                  dcirow,     dcicol,
                                  dcwork,      lddcwork, 
                                  0.0);
    assign_num_level_l_cuda_time += ret.time;

    
    row_stride = 1;
    col_stride = stride;
    ret = mass_mult_l_row_cuda(nr,         nc,
                               nr,         nc,
                               row_stride, col_stride,
                               dcirow,     dcicol,
                               dcwork,     lddcwork,
                               dccoords_x);
    mass_mult_l_row_cuda_time += ret.time;

    row_stride = 1;
    col_stride = stride;
    ret = restriction_l_row_cuda(nr,         nc,
                                 nr,         nc,
                                 row_stride, col_stride,
                                 dcirow,     dcicol,
                                 dcwork,      lddcwork,
                                 dccoords_x);
    restriction_l_row_cuda_time += ret.time;

    row_stride = 1;
    col_stride = Cstride;
    ret = solve_tridiag_M_l_row_cuda(nr,         nc,
                                     nr,         nc,
                                     row_stride, col_stride,
                                     dcirow,     dcicol,
                                     dcwork,     lddcwork,
                                     dccoords_x);
    solve_tridiag_M_l_row_cuda_time += ret.time;


    if (nrow > 1) // do this if we have an 2-dimensional array
    {

      row_stride = stride;
      col_stride = Cstride;
      ret = mass_mult_l_col_cuda(nr,         nc,
                                 nr,         nc,
                                 row_stride, col_stride,
                                 dcirow,     dcicol,
                                 dcwork,     lddcwork,
                                 dccoords_y);
      mass_mult_l_col_cuda_time += ret.time;

      row_stride = stride;
      col_stride = Cstride;
      ret = restriction_l_col_cuda(nr,         nc,
                                   nr,         nc,
                                   row_stride, col_stride,
                                   dcirow,     dcicol,
                                   dcwork, lddcwork,
                                   dccoords_y);
      restriction_l_col_cuda_time += ret.time;

      row_stride = Cstride;
      col_stride = Cstride;
      ret = solve_tridiag_M_l_col_cuda(nr,         nc,
                                       nr,         nc,
                                       row_stride, col_stride,
                                       dcirow,     dcicol,
                                       dcwork, lddcwork,
                                       dccoords_y);
      solve_tridiag_M_l_col_cuda_time += ret.time;

    }

    // // Solved for (z_l, phi_l) = (c_{l+1}, vl)
    // // add_level_l(l + 1, v, work.data(), nr, nc, nrow, ncol);
    ret = add_level_l_cuda(nr,       nc,
                           nr,         nc,
                           row_stride, col_stride,
                           dcirow,     dcicol,
                           dcv,        lddcv, 
                           dcwork,      lddcwork);
    add_level_cuda_time += ret.time;
  }

  ret = restore_from_2k_plus_1(nrow,  ncol,
                               nr,    nc,
                               dirow, dicol,
                               dcv,   lddcv,
                               dv,    lddv);
  restore_from_2k_plus_1_time = ret.time;


  




  std::ofstream timing_results;
  timing_results.open ("refactor_2D_cuda_o1.csv");
  timing_results << "compact_to_2k_plus_1_time," << compact_to_2k_plus_1_time << std::endl;
  timing_results << "restore_from_2k_plus_1_time," << restore_from_2k_plus_1_time << std::endl;

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

//__constant__ double dcoords_x_const[8192];

__global__ void
_mass_mult_l_row_cuda_o1(int nrow,       int ncol,
                         int nr,         int nc,
                         int row_stride, int col_stride,
                         int * __restrict__ dirow,    int * __restrict__ dicol,
                         double * __restrict__ dv,    int lddv,
                         double * __restrict__ dcoords_x,
                         int ghost_col) {

  //int ghost_col = 2;
  register int total_row = ceil((double)nr/(row_stride));
  register int total_col = ceil((double)nc/(col_stride));


  // index on dirow and dicol
  register int r0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  register int c0 = threadIdx.x * col_stride;

  // index on sm
  register int r0_sm = threadIdx.y;
  register int c0_sm = threadIdx.x;

  extern __shared__ double sm[]; // row = blockDim.y; col = blockDim.x + ghost_col;
  register int ldsm = blockDim.x + ghost_col;
  // printf("ldsm = %d\n", ldsm);
  
  double * vec_sm = sm + r0_sm * ldsm;
  double * dcoords_x_sm = sm + blockDim.y * ldsm;
  
  register double result = 1;
  register double h1 = 1;
  register double h2 = 1;
  
  register int rest_col;
  register int real_ghost_col;
  register int real_main_col;

  register double prev_vec_sm;
  register double prev_dicol;
  register double prev_dcoord_x;
  
  for (int r = r0; r < nr; r += gridDim.y * blockDim.y * row_stride) {
    
    double * vec = dv + dirow[r] * lddv;

    prev_vec_sm = 0.0;
    prev_dicol = dicol[c0];
    prev_dcoord_x = dcoords_x[dicol[c0]];
    
    rest_col = total_col;    
    real_ghost_col = min(ghost_col, rest_col);

    // load first ghost
    if (c0_sm < real_ghost_col) {
      vec_sm[c0_sm] = vec[dicol[c0]];
      if (r0_sm == 0) {
        dcoords_x_sm[c0_sm] = dcoords_x[dicol[c0]];
      }
    }
    rest_col -= real_ghost_col;
    __syncthreads();

    while (rest_col > blockDim.x - real_ghost_col) {
      //load main column
      real_main_col = min(blockDim.x, rest_col);
      if (c0_sm < real_main_col) {
        vec_sm[c0_sm + real_ghost_col] = vec[dicol[c0 + real_ghost_col * col_stride]];
        if (r0_sm == 0) {
          dcoords_x_sm[c0_sm + real_ghost_col] = dcoords_x[dicol[c0 + real_ghost_col * col_stride]];
        }
      }
      __syncthreads();

      //computation
      if (c0_sm == 0) {
        //h1 = mgard_common::_get_dist_o1(dcoords_x,  prev_dicol, dicol[c0]);
        h1 = dcoords_x_sm[c0_sm] - prev_dcoord_x;
        //h2 = mgard_common::_get_dist_o1(dcoords_x,  dicol[c0], dicol[c0 + col_stride]);
        h2 = mgard_common::_get_dist_o1(dcoords_x_sm,  c0_sm, c0_sm + 1);
        // register double tmp1 = h1 * prev_vec_sm;
        // register double tmp2 = 2 * (h1 + h2) * vec_sm[c0_sm];
        // register double tmp3 = h2 * vec_sm[c0_sm + 1];
        // tmp1 += tmp2;
        // result += tmp3;
        // result += tmp1;
        //result = h1 + h2;
        result = h1 * prev_vec_sm + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 * vec_sm[c0_sm + 1];
      } else {
        //h1 = mgard_common::_get_dist_o1(dcoords_x, dicol[c0 - col_stride], dicol[c0]);
        h1 = mgard_common::_get_dist_o1(dcoords_x_sm, c0_sm - 1, c0_sm);
        //h2 = mgard_common::_get_dist_o1(dcoords_x, dicol[c0], dicol[c0 + col_stride]);
        h2 = mgard_common::_get_dist_o1(dcoords_x_sm, c0_sm, c0_sm + 1);
        // register double tmp1 = h1 * vec_sm[c0_sm - 1];
        // register double tmp2 = 2 * (h1 + h2) * vec_sm[c0_sm];
        // register double tmp3 = h2 * vec_sm[c0_sm + 1];
        // tmp1 += tmp2;
        // result += tmp3;
        // result += tmp1;
        //result = h1 + h2;
        result = h1 * vec_sm[c0_sm - 1] + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 * vec_sm[c0_sm + 1];
      }
      vec[dicol[c0]] = result;
      __syncthreads();
      

      // store last column
      if (c0_sm == 0) {
        prev_vec_sm = vec_sm[blockDim.x - 1];
        prev_dicol = dicol[c0 + (blockDim.x - 1) * col_stride];
        prev_dcoord_x = dcoords_x[dicol[c0 + (blockDim.x - 1) * col_stride]];//dcoords_x_sm[blockDim.x - 1];
      }

      // advance c0
      c0 += blockDim.x * col_stride;

      // copy ghost to main
      real_ghost_col = min(ghost_col, real_main_col - (blockDim.x - ghost_col));
      if (c0_sm < real_ghost_col) {
        vec_sm[c0_sm] = vec_sm[c0_sm + blockDim.x];
        if (r0_sm == 0) {
          dcoords_x_sm[c0_sm] = dcoords_x_sm[c0_sm + blockDim.x];
        }
      }
      __syncthreads();
      rest_col -= real_main_col;
    } //end while

    if (c0_sm < rest_col) {
       vec_sm[c0_sm + real_ghost_col] = vec[dicol[c0 + real_ghost_col * col_stride]];
       dcoords_x_sm[c0_sm + real_ghost_col] = dcoords_x[dicol[c0 + real_ghost_col * col_stride]];
    }
    __syncthreads();

    if (real_ghost_col + rest_col == 1) {
      if (c0_sm == 0) {
        //h1 = mgard_common::_get_dist_o1(dcoords_x,  prev_dicol, dicol[c0]);
        h1 = dcoords_x_sm[c0_sm] - prev_dcoord_x;
        result = h1 * prev_vec_sm + 2 * h1 * vec_sm[c0_sm];
        vec[dicol[c0]] = result;
      }

    } else {

    if (c0_sm < real_ghost_col + rest_col) {
        
      if (c0_sm == 0) {
        //h1 = mgard_common::_get_dist_o1(dcoords_x,  prev_dicol, dicol[c0]);
        //h2 = mgard_common::_get_dist_o1(dcoords_x,  dicol[c0], dicol[c0 + col_stride]);

        h1 = dcoords_x_sm[c0_sm] - prev_dcoord_x;
        h2 = mgard_common::_get_dist_o1(dcoords_x_sm,  c0_sm, c0_sm + 1);

        result = h1 * prev_vec_sm + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 * vec_sm[c0_sm + 1];
      } else if (c0_sm == real_ghost_col + rest_col - 1) {
        //h1 = mgard_common::_get_dist_o1(dcoords_x, dicol[c0 - col_stride], dicol[c0]);
        h1 = mgard_common::_get_dist_o1(dcoords_x_sm, c0_sm - 1, c0_sm);
        result = h1 * vec_sm[c0_sm - 1] + 2 * h1 * vec_sm[c0_sm];
      } else {
        // h1 = mgard_common::_get_dist_o1(dcoords_x, dicol[c0 - col_stride], dicol[c0]);
        // h2 = mgard_common::_get_dist_o1(dcoords_x, dicol[c0], dicol[c0 + col_stride]);

        h1 = mgard_common::_get_dist_o1(dcoords_x_sm, c0_sm - 1, c0_sm);
        h2 = mgard_common::_get_dist_o1(dcoords_x_sm, c0_sm, c0_sm + 1);
        result = h1 * vec_sm[c0_sm - 1] + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 * vec_sm[c0_sm + 1];
      }
      __syncthreads();
      vec[dicol[c0]] = result;
    }
  }
    

  }
}


mgard_cuda_ret 
mass_mult_l_row_cuda_o1_config(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     double * dv,    int lddv,
                     double * dcoords_x,
                     int B, int ghost_col) {
 

  //cudaMemcpyToSymbol (dcoords_x_const, dcoords_x, sizeof(double)*nc );


  // int B = 4;
  // int ghost_col = 2;
  int total_row = ceil((double)nr/(row_stride));
  int total_col = ceil((double)nc/(col_stride));
  int total_thread_y = ceil((double)nr/(row_stride));
  int total_thread_x = min(B, total_col);

  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);


  size_t sm_size = ((tbx + ghost_col) * (tby + 1)) * sizeof(double);

  int gridy = ceil((float)total_thread_y/tby);
  int gridx = 1; //ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx<< std::endl;



  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _mass_mult_l_row_cuda_o1<<<blockPerGrid, threadsPerBlock, sm_size>>>(nrow,       ncol,
                                                                       nr,         nc,
                                                                       row_stride, col_stride,
                                                                       dirow,      dicol,
                                                                       dv,         lddv,
                                                                       dcoords_x,
                                                                       ghost_col);
  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}


mgard_cuda_ret 
mass_mult_l_row_cuda_o1(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     double * dv,    int lddv,
                     double * dcoords_x) {
 
  int B = 4;
  int ghost_col = 2;
  int total_row = ceil((double)nr/(row_stride));
  int total_col = ceil((double)nc/(col_stride));
  int total_thread_y = ceil((double)nr/(row_stride));
  int total_thread_x = min(B, total_col);

  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);


  size_t sm_size = ((tbx + ghost_col) * tby) * sizeof(double);

  int gridy = ceil((float)total_thread_y/tby);
  int gridx = 1; //ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx<< std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _mass_mult_l_row_cuda_o1<<<blockPerGrid, threadsPerBlock, sm_size>>>(nrow,       ncol,
                                                                       nr,         nc,
                                                                       row_stride, col_stride,
                                                                       dirow,      dicol,
                                                                       dv,         lddv,
                                                                       dcoords_x,
                                                                       ghost_col);
  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}


// sm+data prefetch

__global__ void
_mass_mult_l_row_cuda_o2(int nrow,       int ncol,
                         int nr,         int nc,
                         int row_stride, int col_stride,
                         int * __restrict__ dirow,    int * __restrict__ dicol,
                         double * __restrict__ dv,    int lddv,
                         double * __restrict__ dcoords_x,
                         int ghost_col) {

  //int ghost_col = 2;
  register int total_row = ceil((double)nr/(row_stride));
  register int total_col = ceil((double)nc/(col_stride));


  // index on dirow and dicol
  register int r0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  register int c0 = threadIdx.x * col_stride;

  // index on sm
  register int r0_sm = threadIdx.y;
  register int c0_sm = threadIdx.x;

  extern __shared__ double sm[]; // row = blockDim.y; col = blockDim.x + ghost_col;
  register int ldsm = blockDim.x + ghost_col;
  // printf("ldsm = %d\n", ldsm);
  
  double * vec_sm = sm + r0_sm * ldsm;
  double * dcoords_x_sm = sm + blockDim.y * ldsm;
  
  register double result = 1;
  register double h1 = 1;
  register double h2 = 1;
  
  register int main_col = blockDim.x;
  register int rest_load_col;
  register int rest_comp_col;
  register int curr_ghost_col;
  register int curr_main_col;
  register int next_ghost_col;
  register int next_main_col;

  register double prev_vec_sm;
  register double prev_dicol;
  register double prev_dcoord_x;

  register double next_dv;
  register double next_dcoords_x;
  
  for (int r = r0; r < nr; r += gridDim.y * blockDim.y * row_stride) {
    
    double * vec = dv + dirow[r] * lddv;

    prev_vec_sm = 0.0;
    prev_dicol = dicol[c0];
    prev_dcoord_x = dcoords_x[dicol[c0]];
    
    rest_load_col = total_col;
    rest_comp_col = total_col;
    curr_ghost_col = min(ghost_col, rest_load_col);

    // load first ghost
    if (c0_sm < curr_ghost_col) {
      vec_sm[c0_sm] = vec[dicol[c0]];
      if (r0_sm == 0) {
        dcoords_x_sm[c0_sm] = dcoords_x[dicol[c0]];
      }
    }
    rest_load_col -= curr_ghost_col;
    //load main column
    curr_main_col = min(blockDim.x, rest_load_col);
    if (c0_sm < curr_main_col) {
      vec_sm[c0_sm + curr_ghost_col] = vec[dicol[c0 + curr_ghost_col * col_stride]];
      if (r0_sm == 0) {
        dcoords_x_sm[c0_sm + curr_ghost_col] = dcoords_x[dicol[c0 + curr_ghost_col * col_stride]];
      }
    }
    rest_load_col -= curr_main_col;
    __syncthreads();



    while (rest_comp_col > main_col) {
      //load next main column
      next_main_col = min(blockDim.x, rest_load_col);
      int next_c0 = c0 + (curr_main_col + curr_ghost_col) * col_stride;
      if (c0_sm < next_main_col) {
        next_dv = vec[dicol[next_c0]];
        if (r0_sm == 0) {
          next_dcoords_x = dcoords_x[dicol[next_c0]];
        }
      }
      __syncthreads();

      //computation
      if (c0_sm == 0) {
        //h1 = mgard_common::_get_dist_o1(dcoords_x,  prev_dicol, dicol[c0]);
        h1 = dcoords_x_sm[c0_sm] - prev_dcoord_x;
        //h2 = mgard_common::_get_dist_o1(dcoords_x,  dicol[c0], dicol[c0 + col_stride]);
        h2 = mgard_common::_get_dist_o1(dcoords_x_sm,  c0_sm, c0_sm + 1);

        result = h1 * prev_vec_sm + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 * vec_sm[c0_sm + 1];
      } else {
        //h1 = mgard_common::_get_dist_o1(dcoords_x, dicol[c0 - col_stride], dicol[c0]);
        h1 = mgard_common::_get_dist_o1(dcoords_x_sm, c0_sm - 1, c0_sm);
        //h2 = mgard_common::_get_dist_o1(dcoords_x, dicol[c0], dicol[c0 + col_stride]);
        h2 = mgard_common::_get_dist_o1(dcoords_x_sm, c0_sm, c0_sm + 1);

        result = h1 * vec_sm[c0_sm - 1] + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 * vec_sm[c0_sm + 1];
      }
      vec[dicol[c0]] = result;
      
      rest_comp_col -= main_col;

      // store last column
      if (c0_sm == 0) {
        prev_vec_sm = vec_sm[blockDim.x - 1];
        prev_dicol = dicol[c0 + (blockDim.x - 1) * col_stride];
        prev_dcoord_x = dcoords_x[dicol[c0 + (blockDim.x - 1) * col_stride]];//dcoords_x_sm[blockDim.x - 1];
      }

      __syncthreads();
      


      // advance c0
      c0 += blockDim.x * col_stride;

      // copy ghost to main
      next_ghost_col = curr_main_col + curr_ghost_col - main_col;
      if (c0_sm < next_ghost_col) {
        vec_sm[c0_sm] = vec_sm[c0_sm + main_col];
        if (r0_sm == 0) {
          dcoords_x_sm[c0_sm] = dcoords_x_sm[c0_sm + main_col];
        }
      }
      __syncthreads();
      // copy next main to main
      if (c0_sm < next_main_col) {
        vec_sm[c0_sm + next_ghost_col] = next_dv;
        if (r0_sm == 0) {
          dcoords_x_sm[c0_sm + next_ghost_col] = next_dcoords_x;
        }
      }
      rest_load_col -= next_main_col;

      curr_ghost_col = next_ghost_col;
      curr_main_col = next_main_col;
      //rest_col -= real_main_col;
    } //end while

    // if (c0_sm < col) {
    //    vec_sm[c0_sm + real_ghost_col] = vec[dicol[c0 + real_ghost_col * col_stride]];
    //    dcoords_x_sm[c0_sm + real_ghost_col] = dcoords_x[dicol[c0 + real_ghost_col * col_stride]];
    // }
    // __syncthreads();

    if (rest_comp_col == 1) {
      if (c0_sm == 0) {
        //h1 = mgard_common::_get_dist_o1(dcoords_x,  prev_dicol, dicol[c0]);
        h1 = dcoords_x_sm[c0_sm] - prev_dcoord_x;
        result = h1 * prev_vec_sm + 2 * h1 * vec_sm[c0_sm];
        vec[dicol[c0]] = result;
      }

    } else {

    if (c0_sm < rest_comp_col) {
        
      if (c0_sm == 0) {
        //h1 = mgard_common::_get_dist_o1(dcoords_x,  prev_dicol, dicol[c0]);
        //h2 = mgard_common::_get_dist_o1(dcoords_x,  dicol[c0], dicol[c0 + col_stride]);

        h1 = dcoords_x_sm[c0_sm] - prev_dcoord_x;
        h2 = mgard_common::_get_dist_o1(dcoords_x_sm,  c0_sm, c0_sm + 1);

        result = h1 * prev_vec_sm + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 * vec_sm[c0_sm + 1];
      } else if (c0_sm == rest_comp_col - 1) {
        //h1 = mgard_common::_get_dist_o1(dcoords_x, dicol[c0 - col_stride], dicol[c0]);
        h1 = mgard_common::_get_dist_o1(dcoords_x_sm, c0_sm - 1, c0_sm);
        result = h1 * vec_sm[c0_sm - 1] + 2 * h1 * vec_sm[c0_sm];
      } else {
        // h1 = mgard_common::_get_dist_o1(dcoords_x, dicol[c0 - col_stride], dicol[c0]);
        // h2 = mgard_common::_get_dist_o1(dcoords_x, dicol[c0], dicol[c0 + col_stride]);

        h1 = mgard_common::_get_dist_o1(dcoords_x_sm, c0_sm - 1, c0_sm);
        h2 = mgard_common::_get_dist_o1(dcoords_x_sm, c0_sm, c0_sm + 1);
        result = h1 * vec_sm[c0_sm - 1] + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 * vec_sm[c0_sm + 1];
      }
      __syncthreads();
      vec[dicol[c0]] = result;
    }
  }
    

  }
}


mgard_cuda_ret 
mass_mult_l_row_cuda_o2_config(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     double * dv,    int lddv,
                     double * dcoords_x,
                     int B, int ghost_col) {
 

  //cudaMemcpyToSymbol (dcoords_x_const, dcoords_x, sizeof(double)*nc );


  // int B = 4;
  // int ghost_col = 2;
  int total_row = ceil((double)nr/(row_stride));
  int total_col = ceil((double)nc/(col_stride));
  int total_thread_y = ceil((double)nr/(row_stride));
  int total_thread_x = min(B, total_col);

  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);


  size_t sm_size = ((tbx + ghost_col) * (tby + 1)) * sizeof(double);

  int gridy = ceil((float)total_thread_y/tby);
  int gridx = 1; //ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx<< std::endl;



  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _mass_mult_l_row_cuda_o2<<<blockPerGrid, threadsPerBlock, sm_size>>>(nrow,       ncol,
                                                                       nr,         nc,
                                                                       row_stride, col_stride,
                                                                       dirow,      dicol,
                                                                       dv,         lddv,
                                                                       dcoords_x,
                                                                       ghost_col);
  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}



// __global__ void
// _mass_mult_l_row_cuda_o2(int nrow,       int ncol,
//                          int nr,         int nc,
//                          int row_stride, int col_stride,
//                          int * __restrict__ dirow,    int * __restrict__ dicol,
//                          double * __restrict__ dv,    int lddv,
//                          double * __restrict__ dcoords_x,
//                          int ghost_col) {

//   //int ghost_col = 2;
//   register int total_row = ceil((double)nr/(row_stride));
//   register int total_col = ceil((double)nc/(col_stride));


//   // index on dirow and dicol
//   register int r0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
//   register int c0 = threadIdx.x * col_stride;

//   // index on sm
//   register int r0_sm = threadIdx.y;
//   register int c0_sm = threadIdx.x;

//   extern __shared__ double sm[]; // row = blockDim.y; col = blockDim.x;
//   register int ldsm = blockDim.x;
//   // printf("ldsm = %d\n", ldsm);
  
//   double * vec_sm = sm + r0_sm * ldsm;
//   double * dcoords_x_sm = sm + blockDim.y * ldsm;
  
//   register double result;
//   register double h1;
//   register double h2;
  
//   register int main_col = blockDim.x;
//   register int rest_col;
//   register int real_ghost_col;
//   register int curr_main_col;
//   register int next_main_col;

//   register double next_dv;
//   register double next_dcoords_x

//   register double prev_vec_sm;
//   register double prev_dicol;
//   register double prev_dcoord_x;
  
//   for (int r = r0; r < nr; r += gridDim.y * blockDim.y * row_stride) {
    
//     double * vec = dv + dirow[r] * lddv;

//     prev_vec_sm = 0.0;
//     prev_dicol = dicol[c0];
//     prev_dcoord_x = dcoords_x[dicol[c0]];
    
//     rest_col = total_col;    
//     real_main_col = min(main_col, rest_col);

//     // load first main
//     if (c0_sm < real_main_col) {
//       vec_sm[c0_sm] = vec[dicol[c0]];
//       if (r0_sm == 0) {
//         dcoords_x_sm[c0_sm] = dcoords_x[dicol[c0]];
//       }
//     }
//     //rest_col -= real_main_col;
//     __syncthreads();

//     while (rest_col > 0) {
//       //load main column
//       next_main_col = min(main_col, rest_col - curr_main_col);
//       if (c0_sm < next_main_col) {
//         next_dv = vec[dicol[c0 + (curr_main_col + next_main_col) * col_stride]];
//         if (r0_sm == 0) {
//           next_dcoords_x = dcoords_x[dicol[c0 + (curr_main_col + real_ghost_col) * col_stride]];
//         }
//       }
//       //__syncthreads();

//       //computation
//       if (c0_sm == 0) {
//         //h1 = mgard_common::_get_dist_o1(dcoords_x,  prev_dicol, dicol[c0]);
//         h1 = dcoords_x_sm[c0_sm] - prev_dcoord_x;
//         //h2 = mgard_common::_get_dist_o1(dcoords_x,  dicol[c0], dicol[c0 + col_stride]);
//         h2 = mgard_common::_get_dist_o1(dcoords_x_sm,  c0_sm, c0_sm + 1);
//         result = h1 * prev_vec_sm + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 * vec_sm[c0_sm + 1];
//       } else if (c0_sm < curr_main_col - 1){
//         //h1 = mgard_common::_get_dist_o1(dcoords_x, dicol[c0 - col_stride], dicol[c0]);
//         h1 = mgard_common::_get_dist_o1(dcoords_x_sm, c0_sm - 1, c0_sm);
//         //h2 = mgard_common::_get_dist_o1(dcoords_x, dicol[c0], dicol[c0 + col_stride]);
//         h2 = mgard_common::_get_dist_o1(dcoords_x_sm, c0_sm, c0_sm + 1);
//         result = h1 * vec_sm[c0_sm - 1] + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 * vec_sm[c0_sm + 1];
//       } 

//       if (c0_sm == 0) {
//         h1 = mgard_common::_get_dist_o1(dcoords_x_sm, curr_main_col - 2, curr_main_col - 1);
//         h2 = next_dcoords_x - dcoords_x_sm[curr_main_col - 1];

//       }

//       __syncthreads();



//       vec[dicol[c0]] = result;
//       __syncthreads();
      

//       // store last column
//       if (c0_sm == 0) {
//         prev_vec_sm = vec_sm[blockDim.x - 1];
//         prev_dicol = dicol[c0 + (blockDim.x - 1) * col_stride];
//         prev_dcoord_x = dcoords_x[dicol[c0 + (blockDim.x - 1) * col_stride]];//dcoords_x_sm[blockDim.x - 1];
//       }

//       // advance c0
//       c0 += blockDim.x * col_stride;

//       // copy ghost to main
//       real_ghost_col = min(ghost_col, real_main_col - (blockDim.x - ghost_col));
//       if (c0_sm < real_ghost_col) {
//         vec_sm[c0_sm] = vec_sm[c0_sm + blockDim.x];
//         if (r0_sm == 0) {
//           dcoords_x_sm[c0_sm] = dcoords_x_sm[c0_sm + blockDim.x];
//         }
//       }
//       __syncthreads();
//       rest_col -= real_main_col;
//     } //end while

//     if (c0_sm < rest_col) {
//        vec_sm[c0_sm + real_ghost_col] = vec[dicol[c0 + real_ghost_col * col_stride]];
//        dcoords_x_sm[c0_sm + real_ghost_col] = dcoords_x[dicol[c0 + real_ghost_col * col_stride]];
//     }
//     __syncthreads();

//     if (real_ghost_col + rest_col == 1) {
//       if (c0_sm == 0) {
//         //h1 = mgard_common::_get_dist_o1(dcoords_x,  prev_dicol, dicol[c0]);
//         h1 = dcoords_x_sm[c0_sm] - prev_dcoord_x;
//         result = h1 * prev_vec_sm + 2 * h1 * vec_sm[c0_sm];
//         vec[dicol[c0]] = result;
//       }

//     } else {

//     if (c0_sm < real_ghost_col + rest_col) {
        
//       if (c0_sm == 0) {
//         //h1 = mgard_common::_get_dist_o1(dcoords_x,  prev_dicol, dicol[c0]);
//         //h2 = mgard_common::_get_dist_o1(dcoords_x,  dicol[c0], dicol[c0 + col_stride]);

//         h1 = dcoords_x_sm[c0_sm] - prev_dcoord_x;
//         h2 = mgard_common::_get_dist_o1(dcoords_x_sm,  c0_sm, c0_sm + 1);

//         result = h1 * prev_vec_sm + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 * vec_sm[c0_sm + 1];
//       } else if (c0_sm == real_ghost_col + rest_col - 1) {
//         //h1 = mgard_common::_get_dist_o1(dcoords_x, dicol[c0 - col_stride], dicol[c0]);
//         h1 = mgard_common::_get_dist_o1(dcoords_x_sm, c0_sm - 1, c0_sm);
//         result = h1 * vec_sm[c0_sm - 1] + 2 * h1 * vec_sm[c0_sm];
//       } else {
//         // h1 = mgard_common::_get_dist_o1(dcoords_x, dicol[c0 - col_stride], dicol[c0]);
//         // h2 = mgard_common::_get_dist_o1(dcoords_x, dicol[c0], dicol[c0 + col_stride]);

//         h1 = mgard_common::_get_dist_o1(dcoords_x_sm, c0_sm - 1, c0_sm);
//         h2 = mgard_common::_get_dist_o1(dcoords_x_sm, c0_sm, c0_sm + 1);
//         result = h1 * vec_sm[c0_sm - 1] + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 * vec_sm[c0_sm + 1];
//       }
//       __syncthreads();
//       vec[dicol[c0]] = result;
//     }
//   }
    

//   }
// }

void 
refactor_2D_cuda_o2(const int l_target,
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

  double compact_to_2k_plus_1_time = 0.0;
  double restore_from_2k_plus_1_time = 0.0;

  double original_to_compacted_cuda_time = 0.0;
  double compacted_to_original_cuda_time = 0.0;

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

  // ret = compact_to_2k_plus_1(nrow,  ncol,
  //                      nr,    nc,
  //                      dirow, dicol,
  //                      dv,    lddv,
  //                      dcv,   lddcv);
  // compact_to_2k_plus_1_time = ret.time;


  

  // std::cout << "dv:" << std::endl;
  // print_matrix_cuda(nrow, ncol, dv,  lddv );
  
  // std::cout << "dcv:" << std::endl;
  // print_matrix_cuda(nr,   nc,   dcv, lddcv);

  ret = compact_to_2k_plus_1(nrow,  ncol,
                       nr,    nc,
                       dirow, dicol,
                       dv,    lddv,
                       dcv,   lddcv);
  compact_to_2k_plus_1_time = ret.time;



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
     
    ret = original_to_compacted_cuda(nr,    nc,
                                     row_stride, col_stride,
                                     dcv,        lddcv, 
                                     dcwork,     lddcwork);
    original_to_compacted_cuda_time += ret.time;

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
    ret = compacted_to_original_cuda(nr,         nc, 
                                     row_stride, col_stride,
                                     dcwork,     lddcwork,
                                     dcv,        lddcv);
    compacted_to_original_cuda_time += ret.time;

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
    ret = original_to_compacted_cuda(nr,    nc,
                                     row_stride, col_stride,
                                     dwork,      lddwork,
                                     dcwork,     lddcwork);
    original_to_compacted_cuda_time += ret.time;
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
    ret = compacted_to_original_cuda(nr,         nc, 
                                     row_stride, col_stride,
                                     dcwork,     lddcwork,
                                     dwork,      lddwork);
    compacted_to_original_cuda_time += ret.time;


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
    ret = original_to_compacted_cuda(nr,    nc,
                                     row_stride, col_stride,
                                     dwork,      lddwork,
                                     dcwork,     lddcwork);
    original_to_compacted_cuda_time += ret.time;


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
 
    ret = mass_mult_l_row_cuda_o1(nr_l[0],    nc_l[l],
                                  nr_l[0],    nc_l[l],
                                  row_stride, col_stride,
                                  dcirow_l[0], dcicol_l[l],
                                  dcwork,     lddcwork,
                                  dccoords_x_l[l]);

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
    ret = compacted_to_original_cuda(nr,         nc, 
                                     row_stride, col_stride,
                                     dcwork,     lddcwork,
                                     dwork,      lddwork);
    compacted_to_original_cuda_time += ret.time;

    
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
    ret = original_to_compacted_cuda(nr,    nc,
                                     row_stride, col_stride,
                                     dwork,      lddwork,
                                     dcwork,     lddcwork);
    original_to_compacted_cuda_time += ret.time;

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
    ret = compacted_to_original_cuda(nr,         nc, 
                                     row_stride, col_stride,
                                     dcwork,     lddcwork,
                                     dwork,      lddwork);
    compacted_to_original_cuda_time += ret.time;

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
    ret = original_to_compacted_cuda(nr,    nc,
                                     row_stride, col_stride,
                                     dwork,      lddwork,
                                     dcwork,     lddcwork);
    original_to_compacted_cuda_time += ret.time;

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
    ret = compacted_to_original_cuda(nr,         nc, 
                                     row_stride, col_stride,
                                     dcwork,     lddcwork,
                                     dwork,      lddwork);
    compacted_to_original_cuda_time += ret.time;


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
      ret = original_to_compacted_cuda(nr,    nc,
                                       row_stride, col_stride,
                                       dwork,      lddwork,
                                       dcwork,     lddcwork);
      original_to_compacted_cuda_time += ret.time;


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
      ret = compacted_to_original_cuda(nr,         nc, 
                                       row_stride, col_stride,
                                       dcwork,     lddcwork,
                                       dwork,      lddwork);
      compacted_to_original_cuda_time += ret.time;

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
      ret = original_to_compacted_cuda(nr,    nc,
                                       row_stride, col_stride,
                                       dwork,      lddwork,
                                       dcwork,     lddcwork);
      original_to_compacted_cuda_time += ret.time;

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
      ret = compacted_to_original_cuda(nr,         nc, 
                                       row_stride, col_stride,
                                       dcwork,     lddcwork,
                                       dwork,      lddwork);
      compacted_to_original_cuda_time += ret.time;


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
      ret = original_to_compacted_cuda(nr,    nc,
                                       row_stride, col_stride,
                                       dwork,      lddwork,
                                       dcwork,     lddcwork);
      original_to_compacted_cuda_time += ret.time;
      

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
      ret = compacted_to_original_cuda(nr,         nc, 
                                       row_stride, col_stride,
                                       dcwork,     lddcwork,
                                       dwork,      lddwork);
      compacted_to_original_cuda_time += ret.time;


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

  ret = restore_from_2k_plus_1(nrow,  ncol,
                               nr,    nc,
                               dirow, dicol,
                               dcv,   lddcv,
                               dv,    lddv);
  restore_from_2k_plus_1_time = ret.time;


  




  std::ofstream timing_results;
  timing_results.open ("refactor_2D_cuda_o2.csv");
  timing_results << "compact_to_2k_plus_1_time," << compact_to_2k_plus_1_time << std::endl;
  timing_results << "restore_from_2k_plus_1_time," << restore_from_2k_plus_1_time << std::endl;

  timing_results << "original_to_compacted_cuda_time," << original_to_compacted_cuda_time << std::endl;
  timing_results << "compacted_to_original_cuda_time," << compacted_to_original_cuda_time << std::endl;

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

