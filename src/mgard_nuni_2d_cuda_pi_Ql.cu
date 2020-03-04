#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>
#include <cmath>

namespace mgard_2d {
namespace mgard_gen {  

__global__ void 
_pi_Ql_cuda_sm(int nr,           int nc,
             int row_stride,   int col_stride,
             double * dv,      int lddv, 
             double * ddist_x, double * ddist_y) {

  register int c0 = blockIdx.x * blockDim.x;
  register int c0_stride = c0 * col_stride;
  register int r0 = blockIdx.y * blockDim.y;
  register int r0_stride = r0 * row_stride;

  register int total_row = ceil((double)nr/(row_stride));
  register int total_col = ceil((double)nc/(col_stride));

  register int c_sm = threadIdx.x;
  register int r_sm = threadIdx.y;

  extern __shared__ double sm[]; // size: (blockDim.x + 1) * (blockDim.y + 1)
  int ldsm = blockDim.x + 1;
  double * v_sm = sm;
  double * dist_x_sm = sm + (blockDim.x + 1) * (blockDim.y + 1);
  double * dist_y_sm = dist_x_sm + blockDim.x;

  for (int r = r0; r < total_row - 1; r += blockDim.y * gridDim.y) {
    for (int c = c0; c < total_col - 1; c += blockDim.x * gridDim.x) {
      /* Load v */
      if (c + c_sm < total_col && r + r_sm < total_row) {
        v_sm[r_sm * ldsm + c_sm] = dv[(r + r_sm) * row_stride * lddv + (c + c_sm) * col_stride];

        if (r_sm == 0 && r + blockDim.y < total_row) {
          v_sm[blockDim.y * ldsm + c_sm] = dv[(r + blockDim.y) * row_stride * lddv + (c + c_sm) * col_stride];
        }
        if (c_sm == 0 && c + blockDim.x < total_col) {
          v_sm[r_sm * ldsm + blockDim.x] = dv[(r + r_sm) * row_stride * lddv + (c + blockDim.x) * col_stride];
        }
        if (r_sm == 0 && c_sm == 0 && r + blockDim.y < total_row && c + blockDim.x < total_col) {
          v_sm[blockDim.y * ldsm + blockDim.x] = dv[(r + blockDim.y) * row_stride * lddv + (c + blockDim.x) * col_stride];
        }
      }

      /* Load dist_x */
      //if (c + c_sm < total_col) {
      if (r_sm == 0 && c + c_sm < total_col) {
        dist_x_sm[c_sm] = ddist_x[c + c_sm];
      }
      /* Load dist_y */
      //if (r + r_sm < total_row) {
      if (c_sm == 0 && r + r_sm < total_row) {  
        dist_y_sm[r_sm] = ddist_y[r + r_sm];
        // printf("load ddist_y[%d] %f\n", r_sm, dist_y_sm[r_sm]);
      }

      __syncthreads();

      /* Compute */
      if (r_sm % 2 == 0 && c_sm % 2 != 0) {
        double h1 = dist_x_sm[c_sm - 1];
        double h2 = dist_x_sm[c_sm];
        v_sm[r_sm * ldsm + c_sm] -= (h2 * v_sm[r_sm * ldsm + (c_sm - 1)] + 
                                     h1 * v_sm[r_sm * ldsm + (c_sm + 1)])/
                                    (h1 + h2);
        dv[(r + r_sm) * row_stride * lddv + (c + c_sm) * col_stride] = v_sm[r_sm * ldsm + c_sm];
      } 
      if (r_sm % 2 != 0 && c_sm % 2 == 0) {
        double h1 = dist_y_sm[r_sm - 1];
        double h2 = dist_y_sm[r_sm];
        v_sm[r_sm * ldsm + c_sm] -= (h2 * v_sm[(r_sm - 1) * ldsm + c_sm] +
                                     h1 * v_sm[(r_sm + 1) * ldsm + c_sm])/
                                    (h1 + h2);
        dv[(r + r_sm) * row_stride * lddv + (c + c_sm) * col_stride] = v_sm[r_sm * ldsm + c_sm];

        // if (r_sm == 5) {
        //   printf("dv %f h1 %f h2 %f dv-1 %f dv+1 %f\n", 
        //          v_sm[r_sm * ldsm + c_sm],
        //          dist_y_sm[r_sm - 1], dist_y_sm[r_sm], v_sm[(r_sm - 1) * ldsm + c_sm], v_sm[(r_sm + 1) * ldsm + c_sm]);
        // }
      } 
      if (r_sm % 2 != 0 && c_sm % 2 != 0) {
        double h1_col = dist_x_sm[c_sm - 1];
        double h2_col = dist_x_sm[c_sm];
        double h1_row = dist_y_sm[r_sm - 1];
        double h2_row = dist_y_sm[r_sm];
        v_sm[r_sm * ldsm + c_sm] -= (v_sm[(r_sm - 1) * ldsm + (c_sm - 1)] * h2_col * h2_row +
                                     v_sm[(r_sm - 1) * ldsm + (c_sm + 1)] * h1_col * h2_row + 
                                     v_sm[(r_sm + 1) * ldsm + (c_sm - 1)] * h2_col * h1_row + 
                                     v_sm[(r_sm + 1) * ldsm + (c_sm + 1)] * h1_col * h1_row)/
                                    ((h1_col + h2_col) * (h1_row + h2_row));
        dv[(r + r_sm) * row_stride * lddv + (c + c_sm) * col_stride] = v_sm[r_sm * ldsm + c_sm];
      }
      /* extra computaion for global boarder */
      if (c + blockDim.x == total_col - 1) {
        if (r_sm % 2 != 0 && c_sm == 0) {
          double h1 = dist_y_sm[r_sm - 1];
          double h2 = dist_y_sm[r_sm];
          v_sm[r_sm * ldsm + blockDim.x] -= (h2 * v_sm[(r_sm - 1) * ldsm + blockDim.x] +
                                             h1 * v_sm[(r_sm + 1) * ldsm + blockDim.x])/
                                            (h1 + h2);
          dv[(r + r_sm) * row_stride * lddv + (c + blockDim.x) * col_stride] = v_sm[r_sm * ldsm + blockDim.x];
        } 
      }
      if (r + blockDim.y == total_row - 1) {
        if (r_sm == 0 && c_sm % 2 != 0) {
          double h1 = dist_x_sm[c_sm - 1];
          double h2 = dist_x_sm[c_sm];
          v_sm[blockDim.y * ldsm + c_sm] -= (h2 * v_sm[blockDim.y * ldsm + (c_sm - 1)] + 
                                             h1 * v_sm[blockDim.y * ldsm + (c_sm + 1)])/
                                            (h1 + h2);
          dv[(r + blockDim.y) * row_stride * lddv + (c + c_sm) * col_stride] = v_sm[blockDim.y * ldsm + c_sm];
        }
      }
      __syncthreads();
    }
  }
}


mgard_cuda_ret 
pi_Ql_cuda_sm(int nr,         int nc,
              int row_stride, int col_stride,
              double * dv,    int lddv,
              double * ddist_x, double * ddist_y,
              int B) {
 

  //cudaMemcpyToSymbol (dcoords_x_const, dcoords_x, sizeof(double)*nc );
  // double * ddist_x;
  // //int len_ddist_x = ceil((float)nc/col_stride)-1;
  // int len_ddist_x = ceil((float)nc/col_stride); // add one for better consistance for backward
  // cudaMallocHelper((void**)&ddist_x, len_ddist_x*sizeof(double));
  // calc_cpt_dist(nc, col_stride, dcoords_x, ddist_x);
  // // printf("dcoords_x %d:\n", nc);
  // // print_matrix_cuda(1, nc, dcoords_x, nc);
  // printf("ddist_x:\n");
  // print_matrix_cuda(1, len_ddist_x, ddist_x, len_ddist_x);

  // double * ddist_y;
  // //int len_ddist_x = ceil((float)nc/col_stride)-1;
  // int len_ddist_y = ceil((float)nr/row_stride); // add one for better consistance for backward
  // cudaMallocHelper((void**)&ddist_y, len_ddist_y*sizeof(double));
  // calc_cpt_dist(nr, row_stride, dcoords_y, ddist_y);
  // // printf("dcoords_y %d:\n", nc);
  // // print_matrix_cuda(1, nr, dcoords_y, nr);
  // printf("ddist_y:\n");
  // print_matrix_cuda(1, len_ddist_y, ddist_y, len_ddist_y);



  // int B = 4;
  // int ghost_col = 2;
  int total_row = ceil((double)nr/(row_stride));
  int total_col = ceil((double)nc/(col_stride));
  int total_thread_y = total_row - 1;
  int total_thread_x = total_col - 1;

  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);


  size_t sm_size = ((B+1) * (B+1) + 2 * B) * sizeof(double);

  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx<< std::endl;



  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _pi_Ql_cuda_sm<<<blockPerGrid, threadsPerBlock, sm_size>>>(nr,         nc,
                                                             row_stride, col_stride,
                                                             dv,         lddv,
                                                             ddist_x,    ddist_y);


  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}





}
}