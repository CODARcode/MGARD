#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>
#include <cmath>
#include "mgard_cuda_detail/scan.cu"

namespace mgard_2d {
namespace mgard_gen {  

__device__ double
_dist_tradiag_M_l(double * dcoord, int x, int y) {
  return dcoord[y] - dcoord[x];
}

__global__ void
_calc_am_bm(int n,        
            double * am, double * bm,    
            double * ddist) {
  int c = threadIdx.x;
  int c_sm = threadIdx.x;
  extern __shared__ double sm[];
  double * ddist_sm = sm;
  double * am_sm = sm + blockDim.x;
  double * bm_sm = am_sm + blockDim.x;

  double prev_am = 1.0;
  double prev_dist = 0.0;
  int rest = n;

  while (rest > blockDim.x) {
    /* Load ddsist */
    ddist_sm[c_sm] = ddist[c];
    __syncthreads();
    /* Calculation on one thread*/
    if (c_sm == 0) {
      bm_sm[0] = prev_dist / prev_am;
      am_sm[0] = 2.0 * (ddist_sm[0] + prev_dist) - bm_sm[0] * prev_dist;
      for (int i = 1; i < blockDim.x; i++) {
        bm_sm[i] = ddist_sm[i-1] / am_sm[i-1];
        am_sm[i] = 2.0 * (ddist_sm[i-1] + ddist_sm[i]) - bm_sm[i] * ddist_sm[i-1];
      }
      prev_am = am_sm[blockDim.x-1];
      prev_dist = ddist_sm[blockDim.x-1];
    }
    __syncthreads();
    am[c] = am_sm[c_sm];
    bm[c] = bm_sm[c_sm];
    __syncthreads();
    c += blockDim.x;
    rest -= blockDim.x;
    __syncthreads();
  } // end of while


  if (c_sm < rest-1) {
    ddist_sm[c_sm] = ddist[c];
  }
  
  __syncthreads();
  if (c_sm == 0) {
    if (rest == 1) {
      bm_sm[rest-1] = prev_dist / prev_am;
      am_sm[rest-1] = 2.0 * prev_dist - bm_sm[rest-1] * prev_dist;
      // printf("bm = %f\n", bm_sm[rest-1]);
      // printf("am = %f\n", am_sm[rest-1]);
    } else {
      bm_sm[0] = prev_dist / prev_am;
      am_sm[0] = 2.0 * (ddist_sm[0] + prev_dist) - bm_sm[0] * prev_dist;
      for (int i = 1; i < rest-1; i++) {
        bm_sm[i] = ddist_sm[i-1] / am_sm[i-1];
        am_sm[i] = 2.0 * (ddist_sm[i-1] + ddist_sm[i]) - bm_sm[i] * ddist_sm[i-1];
      }
      bm_sm[rest-1] = ddist_sm[rest-2] / am_sm[rest-2];
      am_sm[rest-1] = 2.0 * ddist_sm[rest-2] - bm_sm[rest-1] * ddist_sm[rest-2];
    }
  }
  __syncthreads();
  if (c_sm < rest) {
    // printf("bm_sm = %f\n", bm_sm[c_sm]);
    // printf("am_sm = %f\n", am_sm[c_sm]);
    am[c] = am_sm[c_sm];
    bm[c] = bm_sm[c_sm];
    // printf("bm_sm = %f\n", bm_sm[c_sm]);
    // printf("am_sm = %f\n", am_sm[c_sm]);
    // printf("bm = %f\n", bm[c]);
    // printf("am = %f\n", am[c]);
  }
}

mgard_cuda_ret
calc_am_bm(int n,        
           double * am, double * bm,    
           double * ddist,
           int B) {
  int total_thread_y = 1;
  int total_thread_x = B;

  int tby = 1;
  int tbx = min(B, total_thread_x);


  size_t sm_size = B * 3 * sizeof(double);

  int gridy = 1;
  int gridx = 1;
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _calc_am_bm<<<blockPerGrid, threadsPerBlock, sm_size>>>(n, am, bm,
                                                          ddist);
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
_solve_tridiag_M_l_row_forward_cuda_sm(int nr,             int nc,
                                       int row_stride,     int col_stride,
                                       double * bm, 
                                       double * dv,        int lddv, 
                                       int ghost_col) {

  /* Global col idx */
  register int r0 = blockIdx.x * blockDim.x;
  register int r0_stride = r0 * row_stride;
  register int c = threadIdx.x;
  register int c_stride = c * col_stride;

  /* Local col idx */
  register int r_sm = threadIdx.x; // for computation
  register int c_sm = threadIdx.x; // for load data

  double * vec;

  /* SM allocation */
  extern __shared__ double sm[];
  register int ldsm = blockDim.x + ghost_col;
  double * vec_sm = sm;
  double * bm_sm = sm + (blockDim.x) * ldsm;

  // register double result;

  register double prev_vec_sm = 0.0;

  register int total_col = ceil((double)nc/(col_stride));
  register int rest_col;
  register int real_ghost_col;
  register int real_main_col;
  register int rest_row;

  for (int r = r0_stride; r < nr; r += gridDim.x * blockDim.x * row_stride) {
    rest_row = min(blockDim.x, (int)ceilf((nr - r)/row_stride));

    vec = dv + r * lddv;
    // if (r_sm == 0) printf("vec[0] = %f\n", vec[0]);

    /* Mark progress */
    rest_col = total_col;    
    real_ghost_col = min(ghost_col, rest_col);

    /* Load first ghost */
    if (c_sm < real_ghost_col) {
      for (int i = 0; i < rest_row; i++) {
        vec_sm[i * ldsm + c_sm] = vec[i * row_stride * lddv + c_stride];
        // if (r_sm == 0) printf("r0_stride = %d, vec_sm[%d] = %f\n", r0_stride, i, vec_sm[i * ldsm + c_sm]);
      }
      bm_sm[c_sm] = bm[c];
    }
    rest_col -= real_ghost_col;
    __syncthreads();

    /* Can still fill main col */
    // int j = 0;

    while (rest_col > blockDim.x - real_ghost_col) {
    // while (j<1) {
    //   j++;
      /* Fill main col + next ghost col */
      real_main_col = min(blockDim.x, rest_col);
      if (c_sm < real_main_col) {
        for (int i = 0; i < rest_row; i++) {
          vec_sm[i * ldsm + c_sm + real_ghost_col] = vec[i * row_stride * lddv + c_stride + real_ghost_col * col_stride];
          // printf("c_sm = %d, r0_stride = %d, vec_sm_gh[%d/%d](%d) = %f\n", c_sm, r0_stride, i,rest_row, i * row_stride * lddv + c_stride + real_ghost_col * col_stride, vec_sm[i * ldsm + c_sm + real_ghost_col]);
        }
        bm_sm[c_sm + real_ghost_col]  = bm[c + real_ghost_col];
      }
      __syncthreads();

      /* Computation of v in parallel*/
      if (r_sm < rest_row) {
        vec_sm[r_sm * ldsm + 0] -= prev_vec_sm * bm_sm[0];
        for (int i = 1; i < blockDim.x; i++) {
          vec_sm[r_sm * ldsm + i] -= vec_sm[r_sm * ldsm + i - 1] * bm_sm[i];
        }

        /* Store last v */
        prev_vec_sm = vec_sm[r_sm * ldsm + blockDim.x - 1];
      }
      __syncthreads();
      /* flush results to v */
      for (int i = 0; i < rest_row; i++) {
        vec[i * row_stride * lddv + c_stride] = vec_sm[i * ldsm + c_sm];
      }
      __syncthreads();

      /* Update unloaded col */
      rest_col -= real_main_col;

      // printf("c_stride in while before = %d\n", c_stride);
      //  printf("blockDim.x %d  in while before = %d\n", c_stride);
      /* Advance c */
      c += blockDim.x;
      c_stride += blockDim.x * col_stride;

      // printf("c_stride in while = %d\n", c_stride);
      /* Copy next ghost to main */
      real_ghost_col = min(ghost_col, real_main_col - (blockDim.x - ghost_col));
      if (c_sm < real_ghost_col) {
        for (int i = 0; i < rest_row; i++) {
          vec_sm[i * ldsm + c_sm] = vec_sm[i * ldsm + c_sm + blockDim.x];
        }
        bm_sm[c_sm] = bm_sm[c_sm + blockDim.x];
      }
      __syncthreads();
    } // end of while

    /* Load all rest col */
    if (c_sm < rest_col) {
      for (int i = 0; i < rest_row; i++) {
        vec_sm[i * ldsm + c_sm + real_ghost_col] = vec[i * row_stride * lddv + c_stride + real_ghost_col * col_stride];
      }
      bm_sm[c_sm + real_ghost_col] = bm[c + real_ghost_col];
    }
    __syncthreads();

    /* Only 1 col remain */
    if (real_ghost_col + rest_col == 1) {
      if (r_sm < rest_row) {
        vec_sm[r_sm * ldsm + 0] -= prev_vec_sm * bm_sm[0];
        // printf ("prev_vec_sm = %f\n", prev_vec_sm );
        // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
      }
      __syncthreads();

    } else {
      if (r_sm < rest_row) {
        vec_sm[r_sm * ldsm + 0] -= prev_vec_sm * bm_sm[0];
        for (int i = 1; i < real_ghost_col + rest_col; i++) {
          vec_sm[r_sm * ldsm + i] -= vec_sm[r_sm * ldsm + i - 1] * bm_sm[i];
        }
      }
    }
    __syncthreads();
    /* flush results to v */
    if (c_sm < real_ghost_col + rest_col) {
      for (int i = 0; i < rest_row; i++) {
        vec[i * row_stride * lddv + c_stride] = vec_sm[i * ldsm + c_sm];
        // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] = %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv + c_stride, vec[i * row_stride * lddv + c_stride]);
      }
    }
    __syncthreads();
    
  }
}


mgard_cuda_ret 
solve_tridiag_M_l_row_forward_cuda_sm(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      double * bm,
                                      double * dv,    int lddv,
                                      int B, int ghost_col) {
 

  int total_row = ceil((double)nr/(row_stride));
  int total_col = 1;
  int total_thread_y = 1;
  int total_thread_x = total_row;

  int tby = 1;
  int tbx = min(B, total_thread_x);


  size_t sm_size = (B+1)*(B+ghost_col) * sizeof(double);

  int gridy = 1;
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _solve_tridiag_M_l_row_forward_cuda_sm<<<blockPerGrid, threadsPerBlock, sm_size>>>(nr,         nc,
                                                                                     row_stride, col_stride,
                                                                                     bm,
                                                                                     dv,         lddv,
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



__global__ void
_solve_tridiag_M_l_row_backward_cuda_sm(int nr,             int nc,
                                       int row_stride,     int col_stride,
                                       double * am,        double * ddist_x,
                                       double * dv,        int lddv, 
                                       int ghost_col) {
  /* Global col idx */
  register int r0 = blockIdx.x * blockDim.x;
  register int r0_stride = r0 * row_stride;
  register int c = threadIdx.x;
  register int c_stride = threadIdx.x * col_stride;

  /* Local col idx */
  register int r_sm = threadIdx.x; // for computation
  register int c_sm = threadIdx.x; // for load data

  double * vec;

  /* SM allocation */
  extern __shared__ double sm[];
  register int ldsm = blockDim.x + ghost_col;
  double * vec_sm = sm;
  double * am_sm = sm + (blockDim.x) * ldsm;
  double * dist_x_sm = am_sm + ldsm;



  register double prev_vec_sm = 0.0;

  register int total_col = ceil((double)nc/(col_stride));
  register int rest_col;
  register int real_ghost_col;
  register int real_main_col;
  register int rest_row;

  for (int r = r0_stride; r < nr; r += gridDim.x * blockDim.x * row_stride) {
    rest_row = min(blockDim.x, (int)ceilf((nr - r)/row_stride));

    vec = dv + r * lddv;
    // if (r_sm == 0) printf("vec[0] = %f\n", vec[0]);

    /* Mark progress */
    rest_col = total_col;    
    real_ghost_col = min(ghost_col, rest_col);

    /* Load first ghost */
    if (c_sm < real_ghost_col) {
      for (int i = 0; i < rest_row; i++) {
        vec_sm[i * ldsm + c_sm] = vec[i * row_stride * lddv + (nc-1) - c_stride];
        // if (r_sm == 0) printf("r0_stride = %d, vec_sm[%d] = %f\n", r0_stride, i, vec_sm[i * ldsm + c_sm]);
      }
      am_sm[c_sm] = am[(total_col-1) - c];
      dist_x_sm[c_sm] = ddist_x[(total_col-1) - c];
      // if (c_sm == 0) printf("am_sm[%d] = %f\n",c_sm, am_sm[c_sm]);
      // if (c_sm == 0) printf("ddist_x[%d] = %f\n",(total_col-1) - c, ddist_x[(total_col-1) - c]);
    }
    rest_col -= real_ghost_col;
    __syncthreads();

    while (rest_col > blockDim.x - real_ghost_col) {
      /* Fill main col + next ghost col */
      real_main_col = min(blockDim.x, rest_col);
      if (c_sm < real_main_col) {
        for (int i = 0; i < rest_row; i++) {
          vec_sm[i * ldsm + c_sm + real_ghost_col] = vec[i * row_stride * lddv + (nc-1) - (c_stride + real_ghost_col * col_stride)];
          // printf("c_sm = %d, r0_stride = %d, vec_sm_gh[%d/%d](%d) = %f\n", c_sm, r0_stride, i,rest_row, i * row_stride * lddv + c_stride + real_ghost_col * col_stride, vec_sm[i * ldsm + c_sm + real_ghost_col]);
        }
        am_sm[c_sm + real_ghost_col] = am[(total_col-1) - (c + real_ghost_col)];
        dist_x_sm[c_sm + real_ghost_col] = ddist_x[(total_col-1) - (c + real_ghost_col)];

        // printf("am_sm[%d+ real_ghost_col] = %f\n",c_sm, am_sm[c_sm+ real_ghost_col]);
        // printf("ddist_x[%d] = %f\n",(total_col-1) - (c + real_ghost_col), ddist_x[(total_col-1) - (c + real_ghost_col)]);
        // printf("dist_x_sm[%d] =\n", c_sm + real_ghost_col);
      }
      __syncthreads();

      /* Computation of v in parallel*/
      if (r_sm < rest_row) {
        vec_sm[r_sm * ldsm + 0] = (vec_sm[r_sm * ldsm + 0] - dist_x_sm[0] * prev_vec_sm) / am_sm[0];
        for (int i = 1; i < blockDim.x; i++) {
          vec_sm[r_sm * ldsm + i] = (vec_sm[r_sm * ldsm + i] - dist_x_sm[i] * vec_sm[r_sm * ldsm + i - 1]) / am_sm[i];
        }
        /* Store last v */
        prev_vec_sm = vec_sm[r_sm * ldsm + blockDim.x - 1];
      }
      __syncthreads();

      /* flush results to v */
      for (int i = 0; i < rest_row; i++) {
        vec[i * row_stride * lddv + (nc-1) - c_stride] = vec_sm[i * ldsm + c_sm];
      }
      __syncthreads();

      /* Update unloaded col */
      rest_col -= real_main_col;

      /* Advance c */
      c += blockDim.x;
      c_stride += blockDim.x * col_stride;

      // /* Copy next ghost to main */
      real_ghost_col = min(ghost_col, real_main_col - (blockDim.x - ghost_col));
      if (c_sm < real_ghost_col) {
        for (int i = 0; i < rest_row; i++) {
          vec_sm[i * ldsm + c_sm] = vec_sm[i * ldsm + c_sm + blockDim.x];
        }
        am_sm[c_sm] = am_sm[c_sm + blockDim.x];
        dist_x_sm[c_sm] = dist_x_sm[c_sm + blockDim.x];
      }
      __syncthreads();
    } // end of while

    /* Load all rest col */
    if (c_sm < rest_col) {
      for (int i = 0; i < rest_row; i++) {
        vec_sm[i * ldsm + c_sm + real_ghost_col] = vec[i * row_stride * lddv + (nc-1) - (c_stride + real_ghost_col * col_stride)];
      }
      am_sm[c_sm + real_ghost_col] = am[(total_col-1) - (c + real_ghost_col)];
      dist_x_sm[c_sm + real_ghost_col] = ddist_x[(total_col-1) - (c + real_ghost_col)];
    }
    __syncthreads();

    /* Only 1 col remain */
    if (real_ghost_col + rest_col == 1) {
      if (r_sm < rest_row) {
        vec_sm[r_sm * ldsm + 0] = (vec_sm[r_sm * ldsm + 0] - dist_x_sm[0] * prev_vec_sm) / am_sm[0];
        // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
      }
      __syncthreads();

    } else {
      if (r_sm < rest_row) {
        vec_sm[r_sm * ldsm + 0] = (vec_sm[r_sm * ldsm + 0] - dist_x_sm[0] * prev_vec_sm) / am_sm[0];
        for (int i = 1; i < real_ghost_col + rest_col; i++) {
          vec_sm[r_sm * ldsm + i] = (vec_sm[r_sm * ldsm + i] - dist_x_sm[i] * vec_sm[r_sm * ldsm + i - 1]) / am_sm[i];
        }
      }
    }
    __syncthreads();
    /* flush results to v */
    if (c_sm < real_ghost_col + rest_col) {
      for (int i = 0; i < rest_row; i++) {
        vec[i * row_stride * lddv + (nc-1) - c_stride] = vec_sm[i * ldsm + c_sm];
        // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] = %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv + c_stride, vec[i * row_stride * lddv + c_stride]);
      }
    }
    __syncthreads();


  }  
}


mgard_cuda_ret 
solve_tridiag_M_l_row_backward_cuda_sm(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      double * am,    double * ddist_x,
                                      double * dv,    int lddv,
                                      int B, int ghost_col) {
 

  int total_row = ceil((double)nr/(row_stride));
  int total_col = 1;
  int total_thread_y = 1;
  int total_thread_x = total_row;

  int tby = 1;
  int tbx = min(B, total_thread_x);


  size_t sm_size = (B+2)*(B+ghost_col) * sizeof(double);

  int gridy = 1;
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _solve_tridiag_M_l_row_backward_cuda_sm<<<blockPerGrid, threadsPerBlock, sm_size>>>(nr,         nc,
                                                                                     row_stride, col_stride,
                                                                                     am,         ddist_x,
                                                                                     dv,         lddv,
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
solve_tridiag_M_l_row_cuda_sm(int nr,         int nc,
                              int row_stride, int col_stride,
                              double * dv,    int lddv,
                              double * ddist_x,
                              int B, int ghost_col) {
  // double * ddist_x;
  // //int len_ddist_x = ceil((float)nc/col_stride)-1;
  // int len_ddist_x = ceil((float)nc/col_stride); // add one for better consistance for backward
  // cudaMallocHelper((void**)&ddist_x, len_ddist_x*sizeof(double));
  // calc_cpt_dist(nc, col_stride, dcoords_x, ddist_x);
  // // printf("dcoords_x %d:\n", nc);
  // // print_matrix_cuda(1, nc, dcoords_x, nc);
  // // printf("ddist_x:\n");
  // // print_matrix_cuda(1, len_ddist_x, ddist_x, len_ddist_x);

  mgard_cuda_ret tmp(0, 0.0);
  mgard_cuda_ret ret(0, 0.0);
  double * am;
  double * bm;
  cudaMallocHelper((void**)&am, nc*sizeof(double));
  cudaMallocHelper((void**)&bm, nc*sizeof(double));
  tmp = calc_am_bm(ceil((float)nc/col_stride), am, bm, ddist_x, 16);
  ret.time += tmp.time;

  // printf("am:\n");
  // print_matrix_cuda(1, ceil((float)nc/col_stride), am, ceil((float)nc/col_stride));
  // printf("bm:\n");
  // print_matrix_cuda(1, ceil((float)nc/col_stride), bm, ceil((float)nc/col_stride));


  tmp = solve_tridiag_M_l_row_forward_cuda_sm(nr,         nc,
                                        row_stride, col_stride,
                                        bm,
                                        dv,    lddv,
                                        B,     ghost_col);
  ret.time += tmp.time;
  tmp = solve_tridiag_M_l_row_backward_cuda_sm(nr,         nc,
                                        row_stride, col_stride,
                                        am,     ddist_x,
                                        dv,    lddv,
                                        B,     ghost_col);
  ret.time += tmp.time;
  return ret;
}



__global__ void
_solve_tridiag_M_l_col_forward_cuda_sm(int nr,             int nc,
                                       int row_stride,     int col_stride,
                                       double * bm, 
                                       double * dv,        int lddv, 
                                       int ghost_row) {

  /* Global idx */
  register int c0 = blockIdx.x * blockDim.x;
  register int c0_stride = c0 * col_stride;
  register int r = 0;

  /* Local col idx */
  register int r_sm = threadIdx.x; // for computation
  register int c_sm = threadIdx.x; // for load data

  double * vec;

  /* SM allocation */
  extern __shared__ double sm[];
  register int ldsm = blockDim.x;
  double * vec_sm = sm + c_sm;
  double * bm_sm = sm + (blockDim.x + ghost_row) * ldsm;

  // register double result;

  register double prev_vec_sm = 0.0;

  register int total_row = ceil((double)nr/(row_stride));
  register int rest_row;
  register int real_ghost_row;
  register int real_main_row;
  //register int rest_row;
  // printf("c_sm = %d\n", c_sm);
  for (int c = c0_stride; c < nc; c += gridDim.x * blockDim.x * col_stride) {
    //rest_row = min(blockDim.x, (int)ceilf((nr - r)/row_stride));

    vec = dv + c + threadIdx.x * col_stride;
    // if (r_sm == 0) printf("vec[0] = %f\n", vec[0]);

    /* Mark progress */
    rest_row = total_row;    
    real_ghost_row = min(ghost_row, rest_row);

    /* Load first ghost */

    if (c + threadIdx.x * col_stride < nc) {
      for (int i = 0; i < real_ghost_row; i++) {
        vec_sm[i * ldsm] = vec[(i + r) * row_stride * lddv];
        // if (c + threadIdx.x * col_stride == 0) printf("vec_sm[%d] = %f\n", i, vec_sm[i * ldsm]);
      }
    }
    // printf("c_sm = %d\n", c_sm);
    if (c_sm < real_ghost_row) {
      bm_sm[c_sm] = bm[r + c_sm];
      // printf("load real_ghost_row = %d, bm_sm[%d] = %f\n", real_ghost_row, c_sm, bm_sm[c_sm]);
    }
    rest_row -= real_ghost_row;
    __syncthreads();

    /* Can still fill main col */
    // int j = 0;

    while (rest_row > blockDim.x - real_ghost_row) {
    // while (j<1) {
    //   j++;
      /* Fill main col + next ghost col */
      real_main_row = min(blockDim.x, rest_row);
      if (c + threadIdx.x * col_stride < nc) {
        for (int i = 0; i < real_main_row; i++) {
          vec_sm[(i + real_ghost_row) * ldsm] = vec[(i + r + real_ghost_row) * row_stride * lddv];
          // if (c + threadIdx.x * col_stride == 0) printf("vec_sm[%d] = %f, vec[%d] = %f\n", 
          //                   i + real_ghost_row, vec_sm[(i + real_ghost_row) * ldsm],
          //                   (i + r + real_ghost_row) * row_stride * lddv,
          //                   vec[(i + r + real_ghost_row) * row_stride * lddv]);      
        }
      }
      if (c_sm < real_main_row) {
        bm_sm[c_sm + real_ghost_row]  = bm[r + c_sm + real_ghost_row];
      }
      __syncthreads();
      if (c + threadIdx.x * col_stride < nc) {
        /* Computation of v in parallel*/
        vec_sm[0 * ldsm] -= prev_vec_sm * bm_sm[0];
        for (int i = 1; i < blockDim.x; i++) {
          vec_sm[i * ldsm] -= vec_sm[(i - 1) * ldsm] * bm_sm[i];
        }
        /* Store last v */
        prev_vec_sm = vec_sm[(blockDim.x - 1) * ldsm];
        /* flush results to v */
        for (int i = 0; i < blockDim.x; i++) {
          vec[(i + r) * row_stride * lddv] = vec_sm[i * ldsm];
        }
      }
      __syncthreads();

      /* Update unloaded row */
      rest_row -= real_main_row;

      // printf("c_stride in while before = %d\n", c_stride);
      //  printf("blockDim.x %d  in while before = %d\n", c_stride);
      /* Advance r */
      r += blockDim.x;

      // printf("c_stride in while = %d\n", c_stride);
      if (c + threadIdx.x * col_stride < nc) {
        /* Copy next ghost to main */
        real_ghost_row = min(ghost_row, real_main_row - (blockDim.x - ghost_row));
        for (int i = 0; i < real_ghost_row; i++) {
          vec_sm[i * ldsm] = vec_sm[(i + blockDim.x) * ldsm];
        }
      }
      if (c_sm < real_ghost_row) {
        bm_sm[c_sm] = bm_sm[c_sm + blockDim.x];
      }
      __syncthreads();
    } // end of while

    /* Load all rest row */
    if (c + threadIdx.x * col_stride < nc) {
      for (int i = 0; i < rest_row; i++) {
        vec_sm[(i + real_ghost_row) * ldsm] = vec[(i + r + real_ghost_row) * row_stride * lddv];
      }
    }
    if (c_sm < rest_row) {
      bm_sm[c_sm + real_ghost_row] = bm[r + c_sm + real_ghost_row];
    }
    __syncthreads();

    if (c + threadIdx.x * col_stride < nc) {
      /* Only 1 row remain */
      if (real_ghost_row + rest_row == 1) {
        vec_sm[0 * ldsm] -= prev_vec_sm * bm_sm[0];
        // printf ("prev_vec_sm = %f\n", prev_vec_sm );
        // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
      } else {
        vec_sm[0 * ldsm] -= prev_vec_sm * bm_sm[0];
        for (int i = 1; i < real_ghost_row + rest_row; i++) {
          if (c + threadIdx.x * col_stride == 0) {
            // printf("vec_sm[%d] (%f) -= vec_sm[%d] (%f) * bm_sm[%d] (%f);\n", 
            //     i * ldsm, 
            //     vec_sm[i * ldsm], 
            //     (i - 1) * ldsm, 
            //     vec_sm[(i - 1) * ldsm],
            //       i, 
            //       bm_sm[i]);
          }
          vec_sm[i * ldsm] -= vec_sm[(i - 1) * ldsm] * bm_sm[i];
        }
      }
      /* flush results to v */
      for (int i = 0; i < real_ghost_row + rest_row; i++) {
        vec[(r + i) * row_stride * lddv] = vec_sm[i * ldsm];
        // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] = %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv + c_stride, vec[i * row_stride * lddv + c_stride]);
      }
    }
    __syncthreads();
    
  }
}


mgard_cuda_ret 
solve_tridiag_M_l_col_forward_cuda_sm(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      double * bm,
                                      double * dv,    int lddv,
                                      int B, int ghost_row) {
 

  int total_row = 1;
  int total_col = ceil((double)nc/(col_stride));
  int total_thread_y = 1;
  int total_thread_x = total_col;

  int tby = 1;
  int tbx = min(B, total_thread_x);
  tbx = max(B, tbx);

  size_t sm_size = (B+1)*(B+ghost_row) * sizeof(double);

  int gridy = 1;
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _solve_tridiag_M_l_col_forward_cuda_sm<<<blockPerGrid, threadsPerBlock, sm_size>>>(nr,         nc,
                                                                                     row_stride, col_stride,
                                                                                     bm,
                                                                                     dv,         lddv,
                                                                                     ghost_row);
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
_solve_tridiag_M_l_col_backward_cuda_sm(int nr,             int nc,
                                       int row_stride,     int col_stride,
                                       double * am,        double * ddist_x,
                                       double * dv,        int lddv, 
                                       int ghost_row) {
  /* Global idx */
  register int c0 = blockIdx.x * blockDim.x;
  register int c0_stride = c0 * col_stride;
  register int r = 0;

  /* Local col idx */
  register int r_sm = threadIdx.x; // for computation
  register int c_sm = threadIdx.x; // for load data

  double * vec;

  /* SM allocation */
  extern __shared__ double sm[];
  register int ldsm = blockDim.x;
  double * vec_sm = sm + c_sm;
  double * am_sm = sm + (blockDim.x + ghost_row) * ldsm;
  double * dist_x_sm = am_sm + blockDim.x + ghost_row;



  register double prev_vec_sm = 0.0;

  register int total_row = ceil((double)nr/(row_stride));
  register int rest_row;
  register int real_ghost_row;
  register int real_main_row;
  // register int rest_row;

  for (int c = c0_stride; c < nc; c += gridDim.x * blockDim.x * col_stride) {
    //rest_row = min(blockDim.x, (int)ceilf((nr - r)/row_stride));

    vec = dv + c + threadIdx.x * col_stride;
    // if (r_sm == 0) printf("vec[0] = %f\n", vec[0]);

    /* Mark progress */
    rest_row = total_row;    
    real_ghost_row = min(ghost_row, rest_row);

    /* Load first ghost */
    if (c + threadIdx.x * col_stride < nc) {
      for (int i = 0; i < real_ghost_row; i++) {
        vec_sm[i * ldsm] = vec[((nr - 1) - (i + r)* row_stride) * lddv];
        // if (c_sm==0) printf("load %f from vec[%d]\n", vec_sm[i * ldsm]);
        // if (r_sm == 0) printf("r0_stride = %d, vec_sm[%d] = %f\n", r0_stride, i, vec_sm[i * ldsm + c_sm]);
      }
    }
    if (c_sm < real_ghost_row) {
      am_sm[c_sm] = am[(total_row - 1) - (r + c_sm)];
      dist_x_sm[c_sm] = ddist_x[(total_row - 1) - (r + c_sm)];
      // printf("load am_sm[%d] = %f\n",c_sm, am_sm[c_sm]);
      // printf("load dist_x_sm[%d] = %f\n",c_sm, dist_x_sm[c_sm]);
      // if (c_sm == 0) printf("ddist_x[%d] = %f\n",(total_col-1) - c, ddist_x[(total_col-1) - c]);
    }
    rest_row -= real_ghost_row;
    __syncthreads();
    while (rest_row > blockDim.x - real_ghost_row) {
      /* Fill main col + next ghost col */
      real_main_row = min(blockDim.x, rest_row);
      if (c + threadIdx.x * col_stride < nc) {
        for (int i = 0; i < real_main_row; i++) {
          vec_sm[(i + real_ghost_row) * ldsm] = vec[((nr - 1) - (i + r + real_ghost_row) * row_stride) * lddv];
          // printf("c_sm = %d, r0_stride = %d, vec_sm_gh[%d/%d](%d) = %f\n", c_sm, r0_stride, i,rest_row, i * row_stride * lddv + c_stride + real_ghost_col * col_stride, vec_sm[i * ldsm + c_sm + real_ghost_col]);
        }
      }
      if (c_sm < real_main_row) {
        am_sm[c_sm + real_ghost_row] = am[(total_row-1) - (r + c_sm + real_ghost_row)];
        dist_x_sm[c_sm + real_ghost_row] = ddist_x[(total_row-1) - (r + c_sm + real_ghost_row)];

        // printf("am_sm[%d+ real_ghost_col] = %f\n",c_sm, am_sm[c_sm+ real_ghost_col]);
        // printf("ddist_x[%d] = %f\n",(total_col-1) - (c + real_ghost_col), ddist_x[(total_col-1) - (c + real_ghost_col)]);
        // printf("dist_x_sm[%d] =\n", c_sm + real_ghost_col);
      }
      __syncthreads();

      /* Computation of v in parallel*/
      if (c + threadIdx.x * col_stride < nc) {
        // printf("before vec: %f, am: %f\n", vec_sm[0 * ldsm], am_sm[0]);
        vec_sm[0 * ldsm] = (vec_sm[0 * ldsm] - dist_x_sm[0] * prev_vec_sm) / am_sm[0];
        // printf("after vec: %f, am: %f\n", vec_sm[0 * ldsm], am_sm[0]);
        for (int i = 1; i < blockDim.x; i++) {
          vec_sm[i * ldsm] = (vec_sm[i * ldsm] - dist_x_sm[i] * vec_sm[(i - 1) * ldsm]) / am_sm[i];
        }
        /* Store last v */
        prev_vec_sm = vec_sm[(blockDim.x - 1) * ldsm];

        /* flush results to v */
        for (int i = 0; i < blockDim.x; i++) {
          vec[((nr - 1) - (i + r) * row_stride)  * lddv] = vec_sm[i * ldsm];
          // printf("flush: %f  to: vec[%d]\n", vec_sm[i * ldsm], ((nr - 1) - (i + r)) * row_stride * lddv);
        }
      }
      __syncthreads();

      /* Update unloaded row */
      rest_row -= real_main_row;

    //   /* Advance r */
      r += blockDim.x;

    //   /* Copy next ghost to main */
      real_ghost_row = min(ghost_row, real_main_row - (blockDim.x - ghost_row));
      if (c + threadIdx.x * col_stride < nc) {  
        for (int i = 0; i < real_ghost_row; i++) {
          vec_sm[i * ldsm] = vec_sm[(i + blockDim.x) * ldsm];
        }
      }
      if (c_sm < real_ghost_row) {
        am_sm[c_sm] = am_sm[c_sm + blockDim.x];
        dist_x_sm[c_sm] = dist_x_sm[c_sm + blockDim.x];
      }
      __syncthreads();
    } // end of while

    /* Load all rest col */
    if (c + threadIdx.x * col_stride< nc) {
      for (int i = 0; i < rest_row; i++) {
        vec_sm[(i + real_ghost_row) * ldsm] = vec[((nr - 1) - (i + r + real_ghost_row) * row_stride)  * lddv];
      }
    }
    if (c_sm < rest_row) {
      am_sm[c_sm + real_ghost_row] = am[(total_row - 1) - (r + c_sm + real_ghost_row)];
      dist_x_sm[c_sm + real_ghost_row] = ddist_x[(total_row - 1) - (r + c_sm + real_ghost_row)];

      // printf("am_sm[%d+ real_ghost_col] = %f\n",c_sm, am_sm[c_sm+ real_ghost_col]);
      // printf("ddist_x[%d] = %f\n",(total_col-1) - (c + real_ghost_col), ddist_x[(total_col-1) - (c + real_ghost_col)]);
      // printf("dist_x_sm[%d] =\n", c_sm + real_ghost_col);
    }
    __syncthreads();
    if (c + threadIdx.x * col_stride < nc) {
      /* Only 1 col remain */
      if (real_ghost_row + total_row == 1) {
        vec_sm[0 * ldsm] = (vec_sm[0 * ldsm] - dist_x_sm[0] * prev_vec_sm) / am_sm[0];
      } else {
        // if (c_sm==0) printf("compute: vec_sm[0 * ldsm] (%f) / am_sm[0] (%f) = %f\n", vec_sm[0 * ldsm], am_sm[0], (vec_sm[0 * ldsm] - dist_x_sm[0] * prev_vec_sm) / am_sm[0]);
        vec_sm[0 * ldsm] = (vec_sm[0 * ldsm] - dist_x_sm[0] * prev_vec_sm) / am_sm[0];
        // printf ("thread vec_sm[0 * ldsm]  = %f\n", vec_sm[0 * ldsm]  );
        for (int i = 1; i < real_ghost_row + rest_row; i++) {
          vec_sm[i * ldsm] = (vec_sm[i * ldsm] - dist_x_sm[i] * vec_sm[(i - 1) * ldsm]) / am_sm[i];
        }
      }
      /* flush results to v */
      for (int i = 0; i < real_ghost_row + rest_row; i++) {
        vec[((nr - 1) - (i + r) * row_stride) * lddv] = vec_sm[i * ldsm];
        // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] = %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv + c_stride, vec[i * row_stride * lddv + c_stride]);
      }
    }
    __syncthreads();
  }  
}


mgard_cuda_ret 
solve_tridiag_M_l_col_backward_cuda_sm(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      double * am,    double * ddist_y,
                                      double * dv,    int lddv,
                                      int B, int ghost_row) {
 

  int total_row = 1;
  int total_col = ceil((double)nc/(col_stride));
  int total_thread_y = 1;
  int total_thread_x = total_col;

  int tby = 1;
  int tbx = min(B, total_thread_x);
  tbx = max(B, tbx);

  size_t sm_size = (B+2)*(B+ghost_row) * sizeof(double);

  int gridy = 1;
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _solve_tridiag_M_l_col_backward_cuda_sm<<<blockPerGrid, threadsPerBlock, sm_size>>>(nr,         nc,
                                                                                     row_stride, col_stride,
                                                                                     am,         ddist_y,
                                                                                     dv,         lddv,
                                                                                     ghost_row);
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
solve_tridiag_M_l_col_cuda_sm(int nr,         int nc,
                              int row_stride, int col_stride,
                              double * dv,    int lddv,
                              double * ddist_y,
                              int B, int ghost_row) {
  // double * ddist_y;
  // //int len_ddist_x = ceil((float)nc/col_stride)-1;
  // int len_ddist_y = ceil((float)nr/row_stride); // add one for better consistance for backward
  // cudaMallocHelper((void**)&ddist_y, len_ddist_y*sizeof(double));
  // calc_cpt_dist(nr, row_stride, dcoords_y, ddist_y);
  // // printf("dcoords_y %d:\n", nc);
  // // print_matrix_cuda(1, nr, dcoords_y, nr);
  // // printf("ddist_y:\n");
  // // print_matrix_cuda(1, len_ddist_y, ddist_y, len_ddist_y);

  mgard_cuda_ret tmp(0, 0.0);
  mgard_cuda_ret ret(0, 0.0);
  double * am;
  double * bm;
  cudaMallocHelper((void**)&am, nr*sizeof(double));
  cudaMallocHelper((void**)&bm, nr*sizeof(double));
  tmp = calc_am_bm(ceil((float)nr/row_stride), am, bm, ddist_y, 16);
  ret.time += tmp.time;

  // printf("am:\n");
  // print_matrix_cuda(1, ceil((float)nr/row_stride), am, ceil((float)nr/row_stride));
  // printf("bm:\n");
  // print_matrix_cuda(1, ceil((float)nr/row_stride), bm, ceil((float)nr/row_stride));


  tmp = solve_tridiag_M_l_col_forward_cuda_sm(nr,         nc,
                                              row_stride, col_stride,
                                              bm,
                                              dv,         lddv,
                                              B,          ghost_row);
  ret.time += tmp.time;
  tmp = solve_tridiag_M_l_col_backward_cuda_sm(nr,         nc,
                                               row_stride, col_stride,
                                               am,         ddist_y,
                                               dv,         lddv,
                                               B,          ghost_row);
  ret.time += tmp.time;
  return ret;
}



}
}
