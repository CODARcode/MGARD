#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_nuni_2d_cuda_kernels.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>
#include <cmath>

namespace mgard_2d {
namespace mgard_gen {  

template <typename T>
__global__ void 
_pi_Ql_cuda(int nrow,           int ncol,
            int nr,             int nc,
            int row_stride,     int col_stride,
            int * irow,         int * icol,
            T * dv,        int lddv, 
            T * dcoords_x, T * dcoords_y) {

  int row_Cstride = row_stride * 2;
  int col_Cstride = col_stride * 2;
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_Cstride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_Cstride;
  
  // in most cases it only needs to iterate once unless the input is really large
  for (int y = y0; y + row_Cstride <= nr - 1; y += blockDim.y * gridDim.y * row_Cstride) {
    for (int x = x0; x + col_Cstride <= nc - 1; x += blockDim.x * gridDim.x * col_Cstride) {
      register T a00 = dv[get_idx(lddv, irow[y],             icol[x]             )];
      register T a01 = dv[get_idx(lddv, irow[y],             icol[x+col_stride]  )];
      register T a02 = dv[get_idx(lddv, irow[y],             icol[x+col_Cstride] )];
      register T a10 = dv[get_idx(lddv, irow[y+row_stride],  icol[x]             )];
      register T a11 = dv[get_idx(lddv, irow[y+row_stride],  icol[x+col_stride]  )];
      register T a12 = dv[get_idx(lddv, irow[y+row_stride],  icol[x+col_Cstride] )];
      register T a20 = dv[get_idx(lddv, irow[y+row_Cstride], icol[x]             )];
      register T a21 = dv[get_idx(lddv, irow[y+row_Cstride], icol[x+col_stride]  )];
      register T a22 = dv[get_idx(lddv, irow[y+row_Cstride], icol[x+col_Cstride] )];

      // printf("thread (%d, %d) working on v(%d, %d) = %f \n", y0, x0, irow[y], icol[x+col_stride], a01);

      int h1_col = mgard_common::_get_dist(dcoords_x, icol[x], icol[x + col_stride]);  //icol[x+col_stride]  - icol[x];
      int h2_col = mgard_common::_get_dist(dcoords_x, icol[x + col_stride], icol[x + col_Cstride]);  //icol[x+col_Cstride] - icol[x+col_stride];
      int hsum_col = h1_col + h2_col;
   
      int h1_row = mgard_common::_get_dist(dcoords_y, irow[y], irow[y + row_stride]);  //irow[y+row_stride]  - irow[y];
      int h2_row = mgard_common::_get_dist(dcoords_y, irow[y + row_stride], irow[y + row_Cstride]);  //irow[y+row_Cstride] - irow[y+row_stride];
      int hsum_row = h1_row + h2_row;
      //double ta01 = a01;
      a01 -= (h1_col * a02 + h2_col * a00) / hsum_col;
      //a21 -= (h1_col * a22 + h2_col * a20) / hsum_col;
       // printf("thread (%d, %d) working on v(%d, %d) = %f -> %f \n", y0, x0, irow[y], icol[x+col_stride], ta01, a01);
      
      a10 -= (h1_row * a20 + h2_row * a00) / hsum_row;
      //a12 -= (h1_row * a22 + h2_row * a02) / hsum_row;
     

      a11 -= 1.0 / (hsum_row * hsum_col) * (a00 * h2_col * h2_row + a02 * h1_col * h2_row + a20 * h2_col * h1_row + a22 * h1_col * h1_row);
      
      dv[get_idx(lddv, irow[y],             icol[x+col_stride]  )] = a01;
      dv[get_idx(lddv, irow[y+row_stride],  icol[x]             )] = a10;
      dv[get_idx(lddv, irow[y+row_stride],  icol[x+col_stride]  )] = a11;

      if (x + col_Cstride == nc - 1) {
        a12 -= (h1_row * a22 + h2_row * a02) / hsum_row;
        dv[get_idx(lddv, irow[y+row_stride],  icol[x+col_Cstride] )] = a12;
      }
      if (y + row_Cstride == nr - 1) {
        //double ta21=a21;
        a21 -= (h1_col * a22 + h2_col * a20) / hsum_col;
        dv[get_idx(lddv, irow[y+row_Cstride], icol[x+col_stride]  )] = a21;
         // printf("thread (%d, %d) working on v(%d, %d) = %f -> %f \n", y0, x0, irow[y+row_Cstride], icol[x+col_stride], ta21, a21);
      }
    }
  }

}

template <typename T>
mgard_cuda_ret 
pi_Ql_cuda(int nrow,           int ncol,
           int nr,             int nc,
           int row_stride,     int col_stride,
           int * dirow,        int * dicol,
           T * dv,        int lddv, 
           T * dcoords_x, T * dcoords_y,
           int B, mgard_cuda_handle & handle, 
           int queue_idx, bool profile) {

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_y = floor((double)nr/(row_stride * 2));
  int total_thread_x = floor((double)nc/(col_stride * 2));
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _pi_Ql_cuda<<<blockPerGrid, threadsPerBlock,
                0, stream>>>(nrow,       ncol,
                             nr,         nc,
                             row_stride, col_stride,
                             dirow,      dicol,
                             dv,         lddv,
                             dcoords_x,  dcoords_y);
  gpuErrchk(cudaGetLastError ());

  if (profile) {
    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
  }

  return mgard_cuda_ret(0, milliseconds/1000.0);
}


template <typename T>
__global__ void 
_pi_Ql_cuda_sm(int nr,           int nc,
               int row_stride,   int col_stride,
               T * dv,      int lddv, 
               T * ddist_x, T * ddist_y) {

  register int c0 = blockIdx.x * blockDim.x;
  register int c0_stride = c0 * col_stride;
  register int r0 = blockIdx.y * blockDim.y;
  register int r0_stride = r0 * row_stride;

  register int total_row = ceil((double)nr/(row_stride));
  register int total_col = ceil((double)nc/(col_stride));

  register int c_sm = threadIdx.x;
  register int r_sm = threadIdx.y;

  extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  T * sm = reinterpret_cast<T *>(smem);

  //extern __shared__ double sm[]; // size: (blockDim.x + 1) * (blockDim.y + 1)
  int ldsm = blockDim.x + 1;
  T * v_sm = sm;
  T * dist_x_sm = sm + (blockDim.x + 1) * (blockDim.y + 1);
  T * dist_y_sm = dist_x_sm + blockDim.x;

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
        T h1 = dist_x_sm[c_sm - 1];
        T h2 = dist_x_sm[c_sm];
        v_sm[r_sm * ldsm + c_sm] -= (h2 * v_sm[r_sm * ldsm + (c_sm - 1)] + 
                                     h1 * v_sm[r_sm * ldsm + (c_sm + 1)])/
                                    (h1 + h2);
        dv[(r + r_sm) * row_stride * lddv + (c + c_sm) * col_stride] = v_sm[r_sm * ldsm + c_sm];
      } 
      if (r_sm % 2 != 0 && c_sm % 2 == 0) {
        T h1 = dist_y_sm[r_sm - 1];
        T h2 = dist_y_sm[r_sm];
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
        T h1_col = dist_x_sm[c_sm - 1];
        T h2_col = dist_x_sm[c_sm];
        T h1_row = dist_y_sm[r_sm - 1];
        T h2_row = dist_y_sm[r_sm];
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
          T h1 = dist_y_sm[r_sm - 1];
          T h2 = dist_y_sm[r_sm];
          v_sm[r_sm * ldsm + blockDim.x] -= (h2 * v_sm[(r_sm - 1) * ldsm + blockDim.x] +
                                             h1 * v_sm[(r_sm + 1) * ldsm + blockDim.x])/
                                            (h1 + h2);
          dv[(r + r_sm) * row_stride * lddv + (c + blockDim.x) * col_stride] = v_sm[r_sm * ldsm + blockDim.x];
        } 
      }
      if (r + blockDim.y == total_row - 1) {
        if (r_sm == 0 && c_sm % 2 != 0) {
          T h1 = dist_x_sm[c_sm - 1];
          T h2 = dist_x_sm[c_sm];
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

template <typename T>
mgard_cuda_ret 
pi_Ql_cuda_sm(int nr,         int nc,
              int row_stride, int col_stride,
              T * dv,    int lddv,
              T * ddist_x, T * ddist_y,
              int B,
              mgard_cuda_handle & handle, 
              int queue_idx, bool profile) {
 
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_row = ceil((double)nr/(row_stride));
  int total_col = ceil((double)nc/(col_stride));
  int total_thread_y = total_row - 1;
  int total_thread_x = total_col - 1;
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  size_t sm_size = ((B+1) * (B+1) + 2 * B) * sizeof(T);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _pi_Ql_cuda_sm<<<blockPerGrid, threadsPerBlock, 
                   sm_size, stream>>>(nr,         nc,
                                      row_stride, col_stride,
                                      dv,         lddv,
                                      ddist_x,    ddist_y);
  gpuErrchk(cudaGetLastError ());

  if (profile) {
    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
  }

  return mgard_cuda_ret(0, milliseconds/1000.0);
}

template mgard_cuda_ret 
pi_Ql_cuda<double>(int nrow,           int ncol,
           int nr,             int nc,
           int row_stride,     int col_stride,
           int * dirow,        int * dicol,
           double * dv,        int lddv, 
           double * dcoords_x, double * dcoords_y,
           int B, mgard_cuda_handle & handle, 
           int queue_idx, bool profile);
template mgard_cuda_ret 
pi_Ql_cuda<float>(int nrow,           int ncol,
           int nr,             int nc,
           int row_stride,     int col_stride,
           int * dirow,        int * dicol,
           float * dv,        int lddv, 
           float * dcoords_x, float * dcoords_y,
           int B, mgard_cuda_handle & handle, 
           int queue_idx, bool profile);



template mgard_cuda_ret 
pi_Ql_cuda_sm<double>(int nr,         int nc,
                      int row_stride, int col_stride,
                      double * dv,    int lddv,
                      double * ddist_x, double * ddist_y,
                      int B,
                      mgard_cuda_handle & handle, 
                      int queue_idx, bool profile);
template mgard_cuda_ret 
pi_Ql_cuda_sm<float>(int nr,         int nc,
                      int row_stride, int col_stride,
                      float * dv,    int lddv,
                      float * ddist_x, float * ddist_y,
                      int B,
                      mgard_cuda_handle & handle, 
                      int queue_idx, bool profile);


}
}