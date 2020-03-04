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

// todo: to make a function maps icol/irow index either from memory or calculation
// todo: double ghost col to avoid thread divergence

__device__ double
_dist_restriction_l(double * dcoord, int x, int y) {
  return dcoord[y] - dcoord[x];
}

// assume number of main col and ghost col are even numbers
__global__ void
_restriction_l_row_cuda_sm(int nr,         int nc,
                         int row_stride, int col_stride,
                         //int * __restrict__ dirow,    int * __restrict__ dicol,
                         double * __restrict__ dv,    int lddv,
                         //double * __restrict__ dcoords_x,
                         double * ddist_x,
                         int ghost_col) {

  //int ghost_col = 2;
  register int total_row = ceil((double)nr/(row_stride));
  register int total_col = ceil((double)nc/(col_stride));


  // index on dirow and dicol
  register int r0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  register int c0 = threadIdx.x;
  register int c0_stride = threadIdx.x * col_stride;

  // index on sm
  register int r0_sm = threadIdx.y;
  register int c0_sm = threadIdx.x;

  extern __shared__ double sm[]; // row = blockDim.y; col = blockDim.x + ghost_col;
  register int ldsm = blockDim.x + ghost_col;
  
  double * vec_sm = sm + r0_sm * ldsm;
  double * dist_x_sm = sm + (blockDim.y) * ldsm;
  
  register double result = 1;
  register double h1 = 1;
  register double h2 = 1;
  register double h3;
  register double h4;
  
  register double result1;
  register double result2;
  register int rest_col;
  register int real_ghost_col;
  register int real_main_col;

  register double prev_vec_sm;
  register double prev_h1;
  register double prev_h2;
  
  for (int r = r0; r < nr; r += gridDim.y * blockDim.y * row_stride) {
    
    double * vec = dv + r * lddv;

    rest_col = total_col;    
    real_ghost_col = min(ghost_col, rest_col);

    // load first ghost
    if (c0_sm < real_ghost_col) {
      vec_sm[c0_sm] = vec[c0_stride];
      if (r0_sm == 0) {
        dist_x_sm[c0_sm] = ddist_x[c0];
      }
    }
    rest_col -= real_ghost_col;
    __syncthreads();

    prev_vec_sm = 0.0;
    prev_h1 = dist_x_sm[0];
    prev_h2 = dist_x_sm[1];

    while (rest_col > blockDim.x - real_ghost_col) {
      //load main column
      real_main_col = min(blockDim.x, rest_col);
      if (c0_sm < real_main_col) {
        vec_sm[c0_sm + real_ghost_col] = vec[c0_stride + real_ghost_col * col_stride];
        if (r0_sm == 0) {
          dist_x_sm[c0_sm + real_ghost_col] = ddist_x[c0 + real_ghost_col];
        }
      }
      __syncthreads();
      
      if (c0_sm % 2 == 0) {
        //computation
        result = vec_sm[c0_sm];
        if (c0_sm == 0) {
          //result = vec_sm[c0_sm];
          h1 = prev_h1;
          h2 = prev_h2;
          h3 = dist_x_sm[c0_sm];
          h4 = dist_x_sm[c0_sm+1];
          result1 = h1 * prev_vec_sm / (h1+h2);
          // h1 = dist_x_sm[c0_sm];
          // h2 = dist_x_sm[c0_sm+1];
          result2 = h4 * vec_sm[c0_sm+1] / (h3+h4);
          //result = result + result1 + result2;
        } else {
          //result = vec_sm[c0_sm];
          h1 = dist_x_sm[c0_sm-2];
          h2 = dist_x_sm[c0_sm-1];
          h3 = dist_x_sm[c0_sm];
          h4 = dist_x_sm[c0_sm+1];
          result1 = h1 * vec_sm[c0_sm-1] / (h1+h2);
          // h1 = dist_x_sm[c0_sm];
          // h2 = dist_x_sm[c0_sm+1];
          result2 = h4 * vec_sm[c0_sm+1] / (h3+h4);
        }
        result = result + result1 + result2;
        vec[c0_stride] = result;
      }
       __syncthreads();
      
      // store last column
      if (c0_sm == 0) {
        prev_vec_sm = vec_sm[blockDim.x - 1];
        prev_h1 = dist_x_sm[blockDim.x - 2]; //_dist_restriction_l(dcoords_x_sm, blockDim.x - 2, blockDim.x - 1);
        prev_h2 = dist_x_sm[blockDim.x - 1]; //_dist_restriction_l(dcoords_x_sm, blockDim.x - 1, blockDim.x);
      }

      // advance c0
      c0 += blockDim.x;
      c0_stride += blockDim.x * col_stride;

      // copy ghost to main
      real_ghost_col = min(ghost_col, real_main_col - (blockDim.x - ghost_col));
      if (c0_sm < real_ghost_col) {
        vec_sm[c0_sm] = vec_sm[c0_sm + blockDim.x];
        if (r0_sm == 0) {
          // dcoords_x_sm[c0_sm] = dcoords_x_sm[c0_sm + blockDim.x];
          dist_x_sm[c0_sm] = dist_x_sm[c0_sm + blockDim.x];
        }
      }
      __syncthreads();
      rest_col -= real_main_col;
    } //end while

    if (c0_sm < rest_col) {
       vec_sm[c0_sm + real_ghost_col] = vec[c0_stride + real_ghost_col * col_stride];
       // dcoords_x_sm[c0_sm + real_ghost_col] = dcoords_x[c0_stride + real_ghost_col * col_stride];
       dist_x_sm[c0_sm + real_ghost_col] = ddist_x[c0 + real_ghost_col];
    }
    __syncthreads();

    if (real_ghost_col + rest_col == 1) {
      if (c0_sm == 0) {
        result = vec_sm[c0_sm];
        h1 = prev_h1;
        h2 = prev_h2;
        result += h1 * prev_vec_sm / (h1+h2);
        vec[c0_stride] = result;
      }
    } else {
      if (c0_sm < real_ghost_col + rest_col) {
        if (c0_sm % 2 == 0) {
          if (c0_sm == 0) {
            result = vec_sm[c0_sm];
            h1 = prev_h1;
            h2 = prev_h2;
            result += h1 * prev_vec_sm / (h1+h2);
            h1 = dist_x_sm[c0_sm]; //_dist_restriction_l(dcoords_x_sm, c0_sm, c0_sm+1);
            h2 = dist_x_sm[c0_sm+1]; //_dist_restriction_l(dcoords_x_sm, c0_sm+1, c0_sm+2);
            result += h2 * vec_sm[c0_sm+1] / (h1+h2);
          } else if (c0_sm == real_ghost_col + rest_col - 1) {
            result = vec_sm[c0_sm];
            h1 = dist_x_sm[c0_sm-2]; //_dist_restriction_l(dcoords_x_sm, c0_sm-2, c0_sm-1);;
            h2 = dist_x_sm[c0_sm-1]; //_dist_restriction_l(dcoords_x_sm, c0_sm-1, c0_sm);;
            result += h1 * vec_sm[c0_sm-1] / (h1+h2);
          } else {
            result = vec_sm[c0_sm];
            h1 = dist_x_sm[c0_sm-2]; //_dist_restriction_l(dcoords_x_sm, c0_sm-2, c0_sm-1);
            h2 = dist_x_sm[c0_sm-1]; //_dist_restriction_l(dcoords_x_sm, c0_sm-1, c0_sm);
            result += h1 * vec_sm[c0_sm-1] / (h1+h2);
            h1 = dist_x_sm[c0_sm]; //_dist_restriction_l(dcoords_x_sm, c0_sm, c0_sm+1);
            h2 = dist_x_sm[c0_sm+1]; //_dist_restriction_l(dcoords_x_sm, c0_sm+1, c0_sm+2);
            result += h2 * vec_sm[c0_sm+1] / (h1+h2);
          }
          vec[c0_stride] = result;
        }
        __syncthreads();
      }
    }
  }

}


// __global__ void
// _restriction_l_row_cuda_sm(int nrow,       int ncol,
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

//   extern __shared__ double sm[]; // row = blockDim.y; col = blockDim.x + ghost_col;
//   register int ldsm = blockDim.x + ghost_col;
//   // printf("ldsm = %d\n", ldsm);
  
//   double * vec_sm = sm + r0_sm * ldsm;
//   double * dcoords_x_sm = sm + blockDim.y * ldsm;
  
//   register double result = 1;
//   register double h1 = 1;
//   register double h2 = 1;
  
//   register int rest_col;
//   register int real_ghost_col;
//   register int real_main_col;

//   register double prev_vec_sm;
//   register double prev_h1;
//   register double prev_h2;
  
//   for (int r = r0; r < nr; r += gridDim.y * blockDim.y * row_stride) {
    
//     double * vec = dv + dirow[r] * lddv;

//     rest_col = total_col;    
//     real_ghost_col = min(ghost_col, rest_col);

//     // load first ghost
//     if (c0_sm < real_ghost_col) {
//       vec_sm[c0_sm] = vec[dicol[c0]];
//       if (r0_sm == 0) {
//         dcoords_x_sm[c0_sm] = dcoords_x[dicol[c0]];
//       }
//     }
//     rest_col -= real_ghost_col;
//     __syncthreads();

//     prev_vec_sm = 0.0;
//     prev_h1 = _dist_restriction_l(dcoords_x_sm, 0, 1);
//     prev_h2 = _dist_restriction_l(dcoords_x_sm, 1, 2);

//     while (rest_col > blockDim.x - real_ghost_col) {
//       //load main column
//       real_main_col = min(blockDim.x, rest_col);
//       if (c0_sm < real_main_col) {
//         vec_sm[c0_sm + real_ghost_col] = vec[dicol[c0 + real_ghost_col * col_stride]];
//         if (r0_sm == 0) {
//           dcoords_x_sm[c0_sm + real_ghost_col] = dcoords_x[dicol[c0 + real_ghost_col * col_stride]];
//         }
//       }
//       __syncthreads();

//       if (c0_sm % 2 == 0) {
//         //computation
//         if (c0_sm == 0) {
//           result = vec_sm[c0_sm];
//           h1 = prev_h1;
//           h2 = prev_h2;
//           result += h1 * prev_vec_sm / (h1+h2);
//           h1 = _dist_restriction_l(dcoords_x_sm, c0_sm, c0_sm+1);
//           h2 = _dist_restriction_l(dcoords_x_sm, c0_sm+1, c0_sm+2);
//           result += h2 * vec_sm[c0_sm+1] / (h1+h2);
//         } else {
//           result = vec_sm[c0_sm];
//           h1 = _dist_restriction_l(dcoords_x_sm, c0_sm-2, c0_sm-1);
//           h2 = _dist_restriction_l(dcoords_x_sm, c0_sm-1, c0_sm);
//           result += h1 * vec_sm[c0_sm-1] / (h1+h2);
//           h1 = _dist_restriction_l(dcoords_x_sm, c0_sm, c0_sm+1);
//           h2 = _dist_restriction_l(dcoords_x_sm, c0_sm+1, c0_sm+2);
//           result += h2 * vec_sm[c0_sm+1] / (h1+h2);
//         }
//         vec[dicol[c0]] = result;
//       }
//        __syncthreads();
    
//       // store last column
//       if (c0_sm == 0) {
//         prev_vec_sm = vec_sm[blockDim.x - 1];
//         prev_h1 = _dist_restriction_l(dcoords_x_sm, blockDim.x - 2, blockDim.x - 1);
//         prev_h2 = _dist_restriction_l(dcoords_x_sm, blockDim.x - 1, blockDim.x);
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
//         result = vec_sm[c0_sm];
//         h1 = prev_h1;
//         h2 = prev_h2;
//         result += h1 * prev_vec_sm / (h1+h2);
//         vec[dicol[c0]] = result;
//       }
//     } else {
//       if (c0_sm < real_ghost_col + rest_col) {
//         if (c0_sm % 2 == 0) {
//           if (c0_sm == 0) {
//             result = vec_sm[c0_sm];
//             h1 = prev_h1;
//             h2 = prev_h2;
//             result += h1 * prev_vec_sm / (h1+h2);
//             h1 = _dist_restriction_l(dcoords_x_sm, c0_sm, c0_sm+1);
//             h2 = _dist_restriction_l(dcoords_x_sm, c0_sm+1, c0_sm+2);
//             result += h2 * vec_sm[c0_sm+1] / (h1+h2);
//           } else if (c0_sm == real_ghost_col + rest_col - 1) {
//             result = vec_sm[c0_sm];
//             h1 = _dist_restriction_l(dcoords_x_sm, c0_sm-2, c0_sm-1);;
//             h2 = _dist_restriction_l(dcoords_x_sm, c0_sm-1, c0_sm);;
//             result += h1 * vec_sm[c0_sm-1] / (h1+h2);
//           } else {
//             result = vec_sm[c0_sm];
//             h1 = _dist_restriction_l(dcoords_x_sm, c0_sm-2, c0_sm-1);
//             h2 = _dist_restriction_l(dcoords_x_sm, c0_sm-1, c0_sm);
//             result += h1 * vec_sm[c0_sm-1] / (h1+h2);
//             h1 = _dist_restriction_l(dcoords_x_sm, c0_sm, c0_sm+1);
//             h2 = _dist_restriction_l(dcoords_x_sm, c0_sm+1, c0_sm+2);
//             result += h2 * vec_sm[c0_sm+1] / (h1+h2);
//           }
//           vec[dicol[c0]] = result;
//         }
//         __syncthreads();
//       }
//     }
//   }
// }

mgard_cuda_ret 
restriction_l_row_cuda_sm(int nr,         int nc,
                          int row_stride, int col_stride,
                          double * dv,    int lddv,
                          double * ddist_x,
                          int B, int ghost_col) {
 

  // //cudaMemcpyToSymbol (dcoords_x_const, dcoords_x, sizeof(double)*nc );
  // double * ddist_x;
  // //int len_ddist_x = ceil((float)nc/col_stride)-1;
  // int len_ddist_x = ceil((float)nc/col_stride); // add one for better consistance for backward
  // cudaMallocHelper((void**)&ddist_x, len_ddist_x*sizeof(double));
  // calc_cpt_dist(nc, col_stride, dcoords_x, ddist_x);
  // // printf("dcoords_x %d:\n", nc);
  // // print_matrix_cuda(1, nc, dcoords_x, nc);
  // // printf("ddist_x:\n");
  // // print_matrix_cuda(1, len_ddist_x, ddist_x, len_ddist_x);

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

  _restriction_l_row_cuda_sm<<<blockPerGrid, threadsPerBlock, sm_size>>>(nr,         nc,
                                                                         row_stride, col_stride,
                                                                         dv,         lddv,
                                                                         ddist_x,
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

// assume number of main col and ghost col are even numbers
__global__ void
_restriction_l_col_cuda_sm(int nr,         int nc,
                           int row_stride, int col_stride,
                           double * __restrict__ dv,    int lddv,
                           double * ddist_y,
                           int ghost_row) {

  //int ghost_col = 2;
  register int total_row = ceil((double)nr/(row_stride));
  register int total_col = ceil((double)nc/(col_stride));


  // index on dirow and dicol
  register int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;
  register int r0 = threadIdx.y;
  register int r0_stride = threadIdx.y * row_stride;

  // index on sm
  register int r0_sm = threadIdx.y;
  register int c0_sm = threadIdx.x;

  extern __shared__ double sm[]; // row = blockDim.y; col = blockDim.x + ghost_col;
  register int ldsm = blockDim.x;
  
  double * vec_sm = sm + c0_sm;
  double * dist_y_sm = sm + (blockDim.y + ghost_row) * ldsm;
  
  register double result = 1;
  register double h1 = 1;
  register double h2 = 1;
  register double h3;
  register double h4;
  
  register double result1;
  register double result2;
  register int rest_row;
  register int real_ghost_row;
  register int real_main_row;

  register double prev_vec_sm;
  register double prev_h1;
  register double prev_h2;
  
  for (int c = c0; c < nc; c += gridDim.x * blockDim.x * col_stride) {
    
    double * vec = dv + c;

    rest_row = total_row;    
    real_ghost_row = min(ghost_row, rest_row);

    // load first ghost
    if (r0_sm < real_ghost_row) {
      vec_sm[r0_sm * ldsm] = vec[r0_stride * lddv];
    }
    if (c0_sm == 0 && r0_sm < real_ghost_row) {
        dist_y_sm[r0_sm] = ddist_y[r0];
        // printf("load dist[%d] = %f\n", c0_sm, dist_y_sm[c0_sm]);
    }
    rest_row -= real_ghost_row;
    __syncthreads();

    prev_vec_sm = 0.0;
    prev_h1 = dist_y_sm[0];
    prev_h2 = dist_y_sm[1];

    while (rest_row > blockDim.y - real_ghost_row) {
  //     //load main column
      real_main_row = min(blockDim.y, rest_row);
      if (r0_sm < real_main_row) {
        vec_sm[(r0_sm + real_ghost_row) * ldsm] = vec[(r0_stride + real_ghost_row * row_stride) * lddv];
      }
      if (c0_sm == 0 && r0_sm < real_main_row) {
          dist_y_sm[r0_sm + real_ghost_row] = ddist_y[r0 + real_ghost_row];
      }
      __syncthreads();
      
      if (r0_sm % 2 == 0) {
        //computation
        result = vec_sm[r0_sm * ldsm];
        if (r0_sm == 0) {
          //result = vec_sm[c0_sm];
          h1 = prev_h1;
          h2 = prev_h2;
          h3 = dist_y_sm[r0_sm];
          h4 = dist_y_sm[r0_sm+1];
          result1 = h1 * prev_vec_sm / (h1+h2);
          // h1 = dist_x_sm[c0_sm];
          // h2 = dist_x_sm[c0_sm+1];
          result2 = h4 * vec_sm[(r0_sm+1) * ldsm] / (h3+h4);
          //result = result + result1 + result2;
        } else {
          //result = vec_sm[c0_sm];
          h1 = dist_y_sm[r0_sm-2];
          h2 = dist_y_sm[r0_sm-1];
          h3 = dist_y_sm[r0_sm];
          h4 = dist_y_sm[r0_sm+1];
          result1 = h1 * vec_sm[(r0_sm-1) * ldsm] / (h1+h2);
          // h1 = dist_x_sm[c0_sm];
          // h2 = dist_x_sm[c0_sm+1];
          result2 = h4 * vec_sm[(r0_sm+1) * ldsm] / (h3+h4);
        }
        result = result + result1 + result2;
        vec[r0_stride * lddv] = result;
      }
       __syncthreads();
      
      // store last column
      if (r0_sm == 0) {
        prev_vec_sm = vec_sm[(blockDim.y - 1) * ldsm];
        prev_h1 = dist_y_sm[blockDim.y - 2]; //_dist_restriction_l(dcoords_x_sm, blockDim.x - 2, blockDim.x - 1);
        prev_h2 = dist_y_sm[blockDim.y - 1]; //_dist_restriction_l(dcoords_x_sm, blockDim.x - 1, blockDim.x);
      }

      // advance c0
      r0 += blockDim.y;
      r0_stride += blockDim.y * row_stride;

  //     // copy ghost to main
      real_ghost_row = min(ghost_row, real_main_row - (blockDim.y - ghost_row));
      if (r0_sm < real_ghost_row) {
        vec_sm[r0_sm * ldsm] = vec_sm[(r0_sm + blockDim.y) * ldsm];
      }
      if (c0_sm == 0 && r0_sm < real_ghost_row) {
        dist_y_sm[r0_sm] = dist_y_sm[r0_sm + blockDim.y];
      }
      __syncthreads();
      rest_row -= real_main_row;
     } //end while

    if (r0_sm < rest_row) {
      vec_sm[(r0_sm + real_ghost_row) * ldsm] = vec[(r0_stride + real_ghost_row * row_stride) * lddv]; 
    }
    if (c0_sm == 0 && r0_sm < rest_row) {
      dist_y_sm[r0_sm + real_ghost_row] = ddist_y[r0 + real_ghost_row];
    }
    __syncthreads();

    if (real_ghost_row + rest_row == 1) {
      if (r0_sm == 0) {
        result = vec_sm[r0_sm * ldsm];
        h1 = prev_h1;
        h2 = prev_h2;
        result += h1 * prev_vec_sm / (h1+h2);
        vec[r0_stride * lddv] = result;
      }
    } else {
      if (r0_sm < real_ghost_row + rest_row) {
        if (r0_sm % 2 == 0) {
          if (r0_sm == 0) {
            result = vec_sm[r0_sm * ldsm];
            h1 = prev_h1;
            h2 = prev_h2;
            result += h1 * prev_vec_sm / (h1+h2);
            h1 = dist_y_sm[r0_sm]; //_dist_restriction_l(dcoords_x_sm, c0_sm, c0_sm+1);
            h2 = dist_y_sm[r0_sm+1]; //_dist_restriction_l(dcoords_x_sm, c0_sm+1, c0_sm+2);
            result += h2 * vec_sm[(r0_sm+1) * ldsm] / (h1+h2);
          } else if (r0_sm == real_ghost_row + rest_row - 1) {
            result = vec_sm[r0_sm * ldsm];
            h1 = dist_y_sm[r0_sm-2]; //_dist_restriction_l(dcoords_x_sm, c0_sm-2, c0_sm-1);;
            h2 = dist_y_sm[r0_sm-1]; //_dist_restriction_l(dcoords_x_sm, c0_sm-1, c0_sm);;
            result += h1 * vec_sm[(r0_sm-1) * ldsm] / (h1+h2);
          } else {
            result = vec_sm[r0_sm * ldsm];
            h1 = dist_y_sm[r0_sm-2]; //_dist_restriction_l(dcoords_x_sm, c0_sm-2, c0_sm-1);
            h2 = dist_y_sm[r0_sm-1]; //_dist_restriction_l(dcoords_x_sm, c0_sm-1, c0_sm);
            result += h1 * vec_sm[(r0_sm-1) * ldsm] / (h1+h2);
            h1 = dist_y_sm[r0_sm]; //_dist_restriction_l(dcoords_x_sm, c0_sm, c0_sm+1);
            h2 = dist_y_sm[r0_sm+1]; //_dist_restriction_l(dcoords_x_sm, c0_sm+1, c0_sm+2);
            result += h2 * vec_sm[(r0_sm+1) * ldsm] / (h1+h2);
          }
          vec[r0_stride * lddv] = result;
        }
        __syncthreads();
      }
    }
  }

}


mgard_cuda_ret 
restriction_l_col_cuda_sm(int nr,         int nc,
                          int row_stride, int col_stride,
                          double * dv,    int lddv,
                          double * ddist_y,
                          int B, int ghost_row) {
 

  // //cudaMemcpyToSymbol (dcoords_x_const, dcoords_x, sizeof(double)*nc );
  // double * ddist_y;
  // //int len_ddist_x = ceil((float)nc/col_stride)-1;
  // int len_ddist_y = ceil((float)nr/row_stride); // add one for better consistance for backward
  // cudaMallocHelper((void**)&ddist_y, len_ddist_y*sizeof(double));
  // calc_cpt_dist(nr, row_stride, dcoords_y, ddist_y);
  // // printf("dcoords_y %d:\n", nc);
  // // print_matrix_cuda(1, nr, dcoords_y, nr);
  // // printf("ddist_y:\n");
  // // print_matrix_cuda(1, len_ddist_y, ddist_y, len_ddist_y);

  // int B = 4;
  // int ghost_col = 2;
  int total_row = ceil((double)nr/(row_stride));
  int total_col = ceil((double)nc/(col_stride));
  int total_thread_y = min(B, total_row);
  int total_thread_x = ceil((double)nc/(col_stride));

  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);


  size_t sm_size = ((tby + ghost_row) * (tbx + 1)) * sizeof(double);

  int gridy = 1;
  int gridx = ceil((float)total_thread_x/tbx); //ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx<< std::endl;



  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _restriction_l_col_cuda_sm<<<blockPerGrid, threadsPerBlock, sm_size>>>(nr,         nc,
                                                                       row_stride, col_stride,
                                                                       dv,         lddv,
                                                                       ddist_y,
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
_restriction_l_row_cuda_sm_pf(int nrow,       int ncol,
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
  register double prev_h1;
  register double prev_h2;

  register double next_dv;
  register double next_dcoords_x;
  
  for (int r = r0; r < nr; r += gridDim.y * blockDim.y * row_stride) {
    
    double * vec = dv + dirow[r] * lddv;
    
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

    prev_vec_sm = 0.0;
    prev_h1 = _dist_restriction_l(dcoords_x_sm, 0, 1);
    prev_h2 = _dist_restriction_l(dcoords_x_sm, 1, 2);

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

      if (c0_sm % 2 == 0) {
        //computation
        if (c0_sm == 0) {
          result = vec_sm[c0_sm];
          h1 = prev_h1;
          h2 = prev_h2;
          result += h1 * prev_vec_sm / (h1+h2);
          h1 = _dist_restriction_l(dcoords_x_sm, c0_sm, c0_sm+1);
          h2 = _dist_restriction_l(dcoords_x_sm, c0_sm+1, c0_sm+2);
          result += h2 * vec_sm[c0_sm+1] / (h1+h2);
        } else {
          result = vec_sm[c0_sm];
          h1 = _dist_restriction_l(dcoords_x_sm, c0_sm-2, c0_sm-1);
          h2 = _dist_restriction_l(dcoords_x_sm, c0_sm-1, c0_sm);
          result += h1 * vec_sm[c0_sm-1] / (h1+h2);
          h1 = _dist_restriction_l(dcoords_x_sm, c0_sm, c0_sm+1);
          h2 = _dist_restriction_l(dcoords_x_sm, c0_sm+1, c0_sm+2);
          result += h2 * vec_sm[c0_sm+1] / (h1+h2);
        }
        vec[dicol[c0]] = result;
      }
      
      rest_comp_col -= main_col;

      // store last column
      if (c0_sm == 0) {
        prev_vec_sm = vec_sm[blockDim.x - 1];
        prev_h1 = _dist_restriction_l(dcoords_x_sm, blockDim.x - 2, blockDim.x - 1);
        prev_h2 = _dist_restriction_l(dcoords_x_sm, blockDim.x - 1, blockDim.x);
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
        result = vec_sm[c0_sm];
        h1 = prev_h1;
        h2 = prev_h2;
        result += h1 * prev_vec_sm / (h1+h2);
        vec[dicol[c0]] = result;
      }
    } else {
      if (c0_sm < rest_comp_col) {
        if (c0_sm % 2 == 0) {
          if (c0_sm == 0) {
            result = vec_sm[c0_sm];
            h1 = prev_h1;
            h2 = prev_h2;
            result += h1 * prev_vec_sm / (h1+h2);
            h1 = _dist_restriction_l(dcoords_x_sm, c0_sm, c0_sm+1);
            h2 = _dist_restriction_l(dcoords_x_sm, c0_sm+1, c0_sm+2);
            result += h2 * vec_sm[c0_sm+1] / (h1+h2);
          } else if (c0_sm == rest_comp_col - 1) {
            result = vec_sm[c0_sm];
            h1 = _dist_restriction_l(dcoords_x_sm, c0_sm-2, c0_sm-1);;
            h2 = _dist_restriction_l(dcoords_x_sm, c0_sm-1, c0_sm);;
            result += h1 * vec_sm[c0_sm-1] / (h1+h2);
          } else {
            result = vec_sm[c0_sm];
            h1 = _dist_restriction_l(dcoords_x_sm, c0_sm-2, c0_sm-1);
            h2 = _dist_restriction_l(dcoords_x_sm, c0_sm-1, c0_sm);
            result += h1 * vec_sm[c0_sm-1] / (h1+h2);
            h1 = _dist_restriction_l(dcoords_x_sm, c0_sm, c0_sm+1);
            h2 = _dist_restriction_l(dcoords_x_sm, c0_sm+1, c0_sm+2);
            result += h2 * vec_sm[c0_sm+1] / (h1+h2);
          }
          vec[dicol[c0]] = result;
        }
        __syncthreads();      
      }
    }
  }
}


mgard_cuda_ret 
restriction_l_row_cuda_sm_pf(int nrow,       int ncol,
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

  _restriction_l_row_cuda_sm_pf<<<blockPerGrid, threadsPerBlock, sm_size>>>(nrow,       ncol,
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
}
}
