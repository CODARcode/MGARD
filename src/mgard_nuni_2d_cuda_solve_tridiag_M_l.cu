#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_nuni_2d_cuda_common.h"
#include "mgard_nuni_2d_cuda_gen.h"
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

template <typename T>
__global__ void
_solve_tridiag_M_l_row_cuda(int nrow,        int ncol,
                             int nr,         int nc,
                             int row_stride, int col_stride,
                             int * dirow,    int * dicol, 
                             T * dv,    int lddv, 
                             T * dcoords_x,
                             T * dcoeff,    int lddcoeff) {
  int idx0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  //printf("thread %d, nr = %d\n", idx0, nr);
  T am, bm, h1, h2;
  //T * coeff = new T[ncol];
  for (int idx = idx0; idx < nr; idx += (blockDim.x * gridDim.x) * row_stride) {
    //printf("thread %d, nr = %d, idx = %d\n", idx0, nr, idx);
    int r = dirow[idx];
    //printf("thread %d working on row %d \n", idx0, r);
    T * vec = dv + r * lddv;
    T * coeff = dcoeff + idx * lddcoeff;

    am = 2.0 * mgard_common::_get_dist(dcoords_x, dicol[0], dicol[col_stride]); //dicol[col_stride] - dicol[0]
    bm = mgard_common::_get_dist(dcoords_x, dicol[0], dicol[col_stride]) / am; //dicol[col_stride] - dicol[0]

    // if (idx == 0) {
    //   printf("true bm:\n");
    //   printf("%f, ", bm);
    // }

    int counter = 1;
    coeff[0] = am;
    for (int i = col_stride; i < nc - 1; i += col_stride) {

      h1 = mgard_common::_get_dist(dcoords_x, dicol[i - col_stride], dicol[i]);
      h2 = mgard_common::_get_dist(dcoords_x, dicol[i], dicol[i + col_stride]);

      vec[dicol[i]] -= vec[dicol[i - col_stride]] * bm;

      am = 2.0 * (h1 + h2) - bm * h1;
      bm = h2 / am;

      // if (idx == 0) {
      //   printf("%f, ", bm);
      // }

      coeff[counter] = am;
      ++counter;
    }
    h2 = mgard_common::_get_dist(dcoords_x, dicol[nc - 1 - col_stride], dicol[nc - 1]);
    // h2 = dicol[nc - 1] - dicol[nc - 1 - col_stride];


    am = 2.0 * h2 - bm * h2;

    vec[dicol[nc - 1]] -= vec[dicol[nc - 1 - col_stride]] * bm;
    coeff[counter] = am;

    // if (idx == 0) {
    //   // printf("h2 = %f\n", h2);
    //   printf("\ntrue am:\n");
    //   for (int i = 0; i < counter+1; i++) {
    //     printf("%f, ", coeff[i] );
    //   }
    //   printf("\n");
    // }

    /* Start of backward pass */
    vec[dicol[nc - 1]] /= am;
    --counter;

    for (int i = nc - 1 - col_stride; i >= 0; i -= col_stride) {
      h2 = mgard_common::_get_dist(dcoords_x, dicol[i], dicol[i + col_stride]);
      // h2 = dicol[i + col_stride] - dicol[i];
      vec[dicol[i]] = (vec[dicol[i]] - h2 * vec[dicol[i + col_stride]]) / coeff[counter];
      --counter;
    }
  }
  //delete[] coeff;
}

template <typename T>
mgard_cuda_ret
solve_tridiag_M_l_row_cuda(int nrow,       int ncol,
                           int nr,         int nc,
                           int row_stride, int col_stride,
                           int * dirow,    int * dicol,
                           T * dv,     int lddv, 
                           T * dcoords_x,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile) {
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);


  T * dcoeff;
  size_t dcoeff_pitch;
  cudaMallocPitchHelper((void**)&dcoeff, &dcoeff_pitch, nc * sizeof(T), nr);
  int lddcoeff = dcoeff_pitch / sizeof(T);
  cudaMemset2DHelper(dcoeff, dcoeff_pitch, 0, nc * sizeof(T), nr);


  int total_thread = ceil((float)nr / row_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _solve_tridiag_M_l_row_cuda<<<blockPerGrid, threadsPerBlock,
                                0, stream>>>(nrow,   ncol,
                                             nr,     nc,
                                             row_stride, col_stride,
                                             dirow,  dicol,
                                             dv,     lddv, 
                                             dcoords_x,
                                             dcoeff, lddcoeff);
  gpuErrchk(cudaGetLastError ()); 

  if (profile) {
    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
  }

  cudaFreeHelper(dcoeff);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}

template <typename T>
__global__ void
_solve_tridiag_M_l_col_cuda(int nrow,        int ncol,
                             int nr,         int nc,
                             int row_stride, int col_stride,
                             int * dirow,    int * dicol,
                             T * dv,    int lddv, 
                             T * dcoords_y,
                             T * dcoeff, int lddcoeff) {
  int idx0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  //printf("thread %d, nr = %d\n", idx0, nr);
  T am, bm, h1, h2;
  //T * coeff = new T[nrow];
  for (int idx = idx0; idx < nc; idx += (blockDim.x * gridDim.x) * col_stride) {
    // printf("thread %d, nc = %d, idx = %d\n", idx0, nc, idx);
    int c = dicol[idx];
    // printf("thread %d working on col %d \n", idx0, c);
    T * vec = dv + c;
    T * coeff = dcoeff + idx;
    am = 2.0 * mgard_common::_get_dist(dcoords_y, dirow[0], dirow[row_stride]); //dirow[row_stride] - dirow[0]
    bm = mgard_common::_get_dist(dcoords_y, dirow[0], dirow[row_stride]) / am; //dirow[row_stride] - dirow[0]

    // if (idx == 0) {
    //   printf("true bm:\n");
    //   printf("%f, ", bm);
    // }
    
    int counter = 1;
    coeff[0 * lddcoeff] = am;
    
    for (int i = row_stride; i < nr - 1; i += row_stride) {
      h1 = mgard_common::_get_dist(dcoords_y, dirow[i - row_stride], dirow[i]);
      h2 = mgard_common::_get_dist(dcoords_y, dirow[i], dirow[i + row_stride]);

      // h1 = dirow[i] - dirow[i - row_stride];
      // h2 = dirow[i + row_stride] - dirow[i];
      // printf("thread %d working on col %d, vec[%d] = %f \n", idx0, c, dirow[i],  vec[dirow[i] * lddv]);
      vec[dirow[i] * lddv] -= vec[dirow[i - row_stride] * lddv] * bm;

      am = 2.0 * (h1 + h2) - bm * h1;
      bm = h2 / am;

      // if (idx == 0) {
      //   printf("%f, ", bm);
      // }

      coeff[counter * lddcoeff] = am;
      ++counter;

    }
    h2 = mgard_common::_get_dist(dcoords_y, dirow[nr - 1 - row_stride], dirow[nr - 1]);
    // h2 = get_h_l_cuda(dcoords_y, nr, nrow, nr - 1 - row_stride, row_stride);
    am = 2.0 * h2 - bm * h2;

    vec[dirow[nr - 1] * lddv] -= vec[dirow[nr - 1 - row_stride] * lddv] * bm;
    coeff[counter * lddcoeff] = am;

    // if (idx == 0) {
    //   // printf("h2 = %f\n", h2);
    //   printf("\ntrue am:\n");
    //   for (int i = 0; i < counter+1; i++) {
    //     printf("%f, ", coeff[i] );
    //   }
    //   printf("\n");
    // }
    // start of backward pass
    // if(idx == 0) {
    //   printf("vec: %f, am: %f\n", vec[dirow[nr - 1] * lddv], am);
    // }
    vec[dirow[nr - 1] * lddv] /= am;
    --counter;

    for (int i = nr - 1 - row_stride; i >= 0; i -= row_stride) {
      h2 = mgard_common::_get_dist(dcoords_y, dirow[i], dirow[i + row_stride]);
      // h2 = get_h_l_cuda(dcoords_y, nr, nrow, i, row_stride);
      vec[dirow[i] * lddv] =
        (vec[dirow[i] * lddv] - h2 * vec[dirow[i + row_stride] * lddv]) /
        coeff[counter * lddcoeff];
      --counter;
    }
  }
  //delete[] coeff;


}

template <typename T>
mgard_cuda_ret
solve_tridiag_M_l_col_cuda(int nrow,       int ncol,
                           int nr,         int nc,
                           int row_stride, int col_stride,
                           int * dirow,    int * dicol,
                           T * dv,    int lddv, 
                           T * dcoords_y,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile) {
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  T * dcoeff;
  size_t dcoeff_pitch;
  cudaMallocPitchHelper((void**)&dcoeff, &dcoeff_pitch, nc * sizeof(T), nr);
  int lddcoeff = dcoeff_pitch / sizeof(T);
  cudaMemset2DHelper(dcoeff, dcoeff_pitch, 0, nc * sizeof(T), nr);


  int total_thread = ceil((float)nc / col_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _solve_tridiag_M_l_col_cuda<<<blockPerGrid, threadsPerBlock,
                                0, stream>>>(nrow,       ncol,
                                             nr,         nc,
                                             row_stride, col_stride,
                                             dirow,      dicol,
                                             dv,         lddv, 
                                             dcoords_y,
                                             dcoeff, lddcoeff);
  gpuErrchk(cudaGetLastError ()); 

  if (profile) {
    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
  }

  cudaFreeHelper(dcoeff);
  return mgard_cuda_ret(0, milliseconds/1000.0);
}

template <typename T>
__global__ void
_calc_am_bm(int n,        
            T * am, T * bm,    
            T * ddist) {
  int c = threadIdx.x;
  int c_sm = threadIdx.x;
  extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  T * sm = reinterpret_cast<T *>(smem);
  //extern __shared__ double sm[];
  T * ddist_sm = sm;
  T * am_sm = sm + blockDim.x;
  T * bm_sm = am_sm + blockDim.x;

  T prev_am = 1.0;
  T prev_dist = 0.0;
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

template <typename T>
mgard_cuda_ret
calc_am_bm(int n,  T * am, T * bm,    
           T * ddist, int B,
           mgard_cuda_handle & handle, 
           int queue_idx, bool profile) {

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_y = 1;
  int total_thread_x = B;

  int tby = 1;
  int tbx = min(B, total_thread_x);


  size_t sm_size = B * 3 * sizeof(T);

  int gridy = 1;
  int gridx = 1;
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _calc_am_bm<<<blockPerGrid, threadsPerBlock, 
                sm_size, stream>>>(n, am, bm,
                                   ddist);
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
_solve_tridiag_M_l_row_forward_cuda_sm(int nr,             int nc,
                                       int row_stride,     int col_stride,
                                       T * bm, 
                                       T * dv,        int lddv, 
                                       int ghost_col) {

  /* Global col idx */
  register int r0 = blockIdx.x * blockDim.x;
  register int r0_stride = r0 * row_stride;
  register int c = threadIdx.x;
  register int c_stride = c * col_stride;

  /* Local col idx */
  register int r_sm = threadIdx.x; // for computation
  register int c_sm = threadIdx.x; // for load data

  T * vec;

  /* SM allocation */
  extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  T * sm = reinterpret_cast<T *>(smem);
  //extern __shared__ double sm[];
  register int ldsm = blockDim.x + ghost_col;
  T * vec_sm = sm;
  T * bm_sm = sm + (blockDim.x) * ldsm;

  // register double result;

  register T prev_vec_sm = 0.0;

  register int total_col = ceil((double)nc/(col_stride));
  register int rest_col;
  register int real_ghost_col;
  register int real_main_col;
  register int rest_row;

  for (int r = r0_stride; r < nr; r += gridDim.x * blockDim.x * row_stride) {
    rest_row = min(blockDim.x, (int)ceilf((double)(nr - r)/row_stride));

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

template <typename T>
mgard_cuda_ret 
solve_tridiag_M_l_row_forward_cuda_sm(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      T * bm,
                                      T * dv,    int lddv,
                                      int B, int ghost_col,
                                      mgard_cuda_handle & handle, 
                                      int queue_idx, bool profile) {
  
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_row = ceil((double)nr/(row_stride));
  int total_col = 1;
  int total_thread_y = 1;
  int total_thread_x = total_row;

  int tby = 1;
  int tbx = max(B, min(B, total_thread_x));
  size_t sm_size = (B+1)*(B+ghost_col) * sizeof(T);

  int gridy = 1;
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  //printf("nr (%d), row_stride(%d), gridx(%d), gridy(%d), tbx(%d), tby(%d)\n", nr, row_stride, gridx, gridy, tbx, tby);
  
  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }
  
  _solve_tridiag_M_l_row_forward_cuda_sm<<<blockPerGrid, threadsPerBlock, 
                                           sm_size, stream>>>(nr,         nc,
                                                              row_stride, col_stride,
                                                              bm,
                                                              dv,         lddv,
                                                              ghost_col);
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
_solve_tridiag_M_l_row_backward_cuda_sm(int nr,             int nc,
                                       int row_stride,     int col_stride,
                                       T * am,        T * ddist_x,
                                       T * dv,        int lddv, 
                                       int ghost_col) {
  /* Global col idx */
  register int r0 = blockIdx.x * blockDim.x;
  register int r0_stride = r0 * row_stride;
  register int c = threadIdx.x;
  register int c_stride = threadIdx.x * col_stride;

  /* Local col idx */
  register int r_sm = threadIdx.x; // for computation
  register int c_sm = threadIdx.x; // for load data

  T * vec;

  /* SM allocation */
  extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  T * sm = reinterpret_cast<T *>(smem);
  //extern __shared__ double sm[];
  register int ldsm = blockDim.x + ghost_col;
  T * vec_sm = sm;
  T * am_sm = sm + (blockDim.x) * ldsm;
  T * dist_x_sm = am_sm + ldsm;



  register T prev_vec_sm = 0.0;

  register int total_col = ceil((double)nc/(col_stride));
  register int rest_col;
  register int real_ghost_col;
  register int real_main_col;
  register int rest_row;
  
  for (int r = r0_stride; r < nr; r += gridDim.x * blockDim.x * row_stride) {
    rest_row = min(blockDim.x, (int)ceilf((double)(nr - r)/row_stride));

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

template <typename T>
mgard_cuda_ret 
solve_tridiag_M_l_row_backward_cuda_sm(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      T * am,    T * ddist_x,
                                      T * dv,    int lddv,
                                      int B, int ghost_col,
                                      mgard_cuda_handle & handle, 
                                      int queue_idx, bool profile) {
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_row = ceil((double)nr/(row_stride));
  int total_col = 1;
  int total_thread_y = 1;
  int total_thread_x = total_row;

  int tby = 1;
  int tbx = max(B, min(B, total_thread_x));


  size_t sm_size = (B+2)*(B+ghost_col) * sizeof(T);

  int gridy = 1;
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);


  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _solve_tridiag_M_l_row_backward_cuda_sm<<<blockPerGrid, threadsPerBlock, 
                                            sm_size, stream>>>(nr,         nc,
                                                               row_stride, col_stride,
                                                               am,         ddist_x,
                                                               dv,         lddv,
                                                               ghost_col);
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
mgard_cuda_ret 
solve_tridiag_M_l_row_cuda_sm(int nr,         int nc,
                              int row_stride, int col_stride,
                              T * dv,    int lddv,
                              T * ddist_x,
                              T * am, T * bm, 
                              int B, int ghost_col,
                              mgard_cuda_handle & handle, 
                              int queue_idx, bool profile) {

  mgard_cuda_ret tmp(0, 0.0);
  mgard_cuda_ret ret(0, 0.0);
  // T * am;
  // T * bm;
  // cudaMallocHelper((void**)&am, nc*sizeof(T));
  // cudaMallocHelper((void**)&bm, nc*sizeof(T));
  tmp = calc_am_bm(ceil((float)nc/col_stride), am, bm, ddist_x, B,
                    handle, queue_idx, profile);
  ret.time += tmp.time;

  // printf("am:\n");
  // print_matrix_cuda(1, ceil((float)nc/col_stride), am, ceil((float)nc/col_stride));
  // printf("bm:\n");
  // print_matrix_cuda(1, ceil((float)nc/col_stride), bm, ceil((float)nc/col_stride));

  tmp = solve_tridiag_M_l_row_forward_cuda_sm(nr,         nc,
                                              row_stride, col_stride,
                                              bm,
                                              dv,    lddv,
                                              B,     ghost_col,
                                              handle, queue_idx, profile);
  ret.time += tmp.time;
  tmp = solve_tridiag_M_l_row_backward_cuda_sm(nr,         nc,
                                               row_stride, col_stride,
                                               am,     ddist_x,
                                               dv,    lddv,
                                               B,     ghost_col,
                                               handle, queue_idx, profile);
  ret.time += tmp.time;
  // cudaFreeHelper(am);
  // cudaFreeHelper(bm);
  return ret;
}


template <typename T>
__global__ void
_solve_tridiag_M_l_col_forward_cuda_sm(int nr,             int nc,
                                       int row_stride,     int col_stride,
                                       T * bm, 
                                       T * dv,        int lddv, 
                                       int ghost_row) {

  /* Global idx */
  register int c0 = blockIdx.x * blockDim.x;
  register int c0_stride = c0 * col_stride;
  register int r = 0;

  /* Local col idx */
  register int r_sm = threadIdx.x; // for computation
  register int c_sm = threadIdx.x; // for load data

  T * vec;

  /* SM allocation */
  extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  T * sm = reinterpret_cast<T *>(smem);
  //extern __shared__ double sm[];
  register int ldsm = blockDim.x;
  T * vec_sm = sm + c_sm;
  T * bm_sm = sm + (blockDim.x + ghost_row) * ldsm;

  // register double result;

  register T prev_vec_sm = 0.0;

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

template <typename T>
mgard_cuda_ret 
solve_tridiag_M_l_col_forward_cuda_sm(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      T * bm,
                                      T * dv,    int lddv,
                                      int B, int ghost_row,
                                      mgard_cuda_handle & handle, 
                                      int queue_idx, bool profile) {
 
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_row = 1;
  int total_col = ceil((double)nc/(col_stride));
  int total_thread_y = 1;
  int total_thread_x = total_col;

  int tby = 1;
  int tbx = min(B, total_thread_x);
  tbx = max(B, tbx);

  size_t sm_size = (B+1)*(B+ghost_row) * sizeof(T);

  int gridy = 1;
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);


  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _solve_tridiag_M_l_col_forward_cuda_sm<<<blockPerGrid, threadsPerBlock, 
                                           sm_size, stream>>>(nr,         nc,
                                                              row_stride, col_stride,
                                                              bm,
                                                              dv,         lddv,
                                                              ghost_row);
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
_solve_tridiag_M_l_col_backward_cuda_sm(int nr,             int nc,
                                       int row_stride,     int col_stride,
                                       T * am,        T * ddist_x,
                                       T * dv,        int lddv, 
                                       int ghost_row) {
  /* Global idx */
  register int c0 = blockIdx.x * blockDim.x;
  register int c0_stride = c0 * col_stride;
  register int r = 0;

  /* Local col idx */
  register int r_sm = threadIdx.x; // for computation
  register int c_sm = threadIdx.x; // for load data

  T * vec;

  /* SM allocation */
  extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  T * sm = reinterpret_cast<T *>(smem);
  //extern __shared__ double sm[];
  register int ldsm = blockDim.x;
  T * vec_sm = sm + c_sm;
  T * am_sm = sm + (blockDim.x + ghost_row) * ldsm;
  T * dist_x_sm = am_sm + blockDim.x + ghost_row;



  register T prev_vec_sm = 0.0;

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

template <typename T>
mgard_cuda_ret 
solve_tridiag_M_l_col_backward_cuda_sm(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      T * am,    T * ddist_y,
                                      T * dv,    int lddv,
                                      int B, int ghost_row,
                                      mgard_cuda_handle & handle, 
                                      int queue_idx, bool profile) {
 
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_row = 1;
  int total_col = ceil((double)nc/(col_stride));
  int total_thread_y = 1;
  int total_thread_x = total_col;

  int tby = 1;
  int tbx = min(B, total_thread_x);
  tbx = max(B, tbx);

  size_t sm_size = (B+2)*(B+ghost_row) * sizeof(T);

  int gridy = 1;
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);


  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _solve_tridiag_M_l_col_backward_cuda_sm<<<blockPerGrid, threadsPerBlock, 
                                            sm_size, stream>>>(nr,         nc,
                                                               row_stride, col_stride,
                                                               am,         ddist_y,
                                                               dv,         lddv,
                                                               ghost_row);
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
mgard_cuda_ret 
solve_tridiag_M_l_col_cuda_sm(int nr,         int nc,
                              int row_stride, int col_stride,
                              T * dv,    int lddv,
                              T * ddist_y,
                              T * am, T * bm, 
                              int B, int ghost_row,
                              mgard_cuda_handle & handle, 
                              int queue_idx, bool profile) {
  mgard_cuda_ret tmp(0, 0.0);
  mgard_cuda_ret ret(0, 0.0);
  // T * am;
  // T * bm;
  // cudaMallocHelper((void**)&am, nr*sizeof(T));
  // cudaMallocHelper((void**)&bm, nr*sizeof(T));
  tmp = calc_am_bm(ceil((float)nr/row_stride), am, bm, ddist_y, B,
                   handle, queue_idx, profile);
  ret.time += tmp.time;

  // printf("am:\n");
  // print_matrix_cuda(1, ceil((float)nr/row_stride), am, ceil((float)nr/row_stride));
  // printf("bm:\n");
  // print_matrix_cuda(1, ceil((float)nr/row_stride), bm, ceil((float)nr/row_stride));


  tmp = solve_tridiag_M_l_col_forward_cuda_sm(nr,         nc,
                                              row_stride, col_stride,
                                              bm,
                                              dv,         lddv,
                                              B,          ghost_row,
                                              handle, queue_idx, profile);
  ret.time += tmp.time;
  tmp = solve_tridiag_M_l_col_backward_cuda_sm(nr,         nc,
                                               row_stride, col_stride,
                                               am,         ddist_y,
                                               dv,         lddv,
                                               B,          ghost_row,
                                               handle, queue_idx, profile);
  ret.time += tmp.time;
  // cudaFreeHelper(am);
  // cudaFreeHelper(bm);
  return ret;
}


template mgard_cuda_ret
solve_tridiag_M_l_row_cuda<double>(int nrow,       int ncol,
                           int nr,         int nc,
                           int row_stride, int col_stride,
                           int * dirow,    int * dicol,
                           double * dv,     int lddv, 
                           double * dcoords_x,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile);
template mgard_cuda_ret
solve_tridiag_M_l_row_cuda<float>(int nrow,       int ncol,
                           int nr,         int nc,
                           int row_stride, int col_stride,
                           int * dirow,    int * dicol,
                           float * dv,     int lddv, 
                           float * dcoords_x,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile);
template mgard_cuda_ret
solve_tridiag_M_l_col_cuda<double>(int nrow,       int ncol,
                           int nr,         int nc,
                           int row_stride, int col_stride,
                           int * dirow,    int * dicol,
                           double * dv,    int lddv, 
                           double * dcoords_y,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile);
template mgard_cuda_ret
solve_tridiag_M_l_col_cuda<float>(int nrow,       int ncol,
                           int nr,         int nc,
                           int row_stride, int col_stride,
                           int * dirow,    int * dicol,
                           float * dv,    int lddv, 
                           float * dcoords_y,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile);




template mgard_cuda_ret
calc_am_bm<double>(int n,  double * am, double * bm,  double * ddist, int B,
                   mgard_cuda_handle & handle, int queue_idx, bool profile);
template mgard_cuda_ret
calc_am_bm<float>(int n,  float * am, float * bm,  float * ddist, int B,
                  mgard_cuda_handle & handle, int queue_idx, bool profile);

template mgard_cuda_ret 
solve_tridiag_M_l_row_forward_cuda_sm<double>(int nr,         int nc,
                                              int row_stride, int col_stride,
                                              double * bm,
                                              double * dv,    int lddv,
                                              int B, int ghost_col,
                                              mgard_cuda_handle & handle, 
                                              int queue_idx, bool profile);
template mgard_cuda_ret 
solve_tridiag_M_l_row_forward_cuda_sm<float>(int nr,         int nc,
                                              int row_stride, int col_stride,
                                              float * bm,
                                              float * dv,    int lddv,
                                              int B, int ghost_col,
                                              mgard_cuda_handle & handle, 
                                              int queue_idx, bool profile);

template mgard_cuda_ret 
solve_tridiag_M_l_row_backward_cuda_sm<double>(int nr,         int nc,
                                              int row_stride, int col_stride,
                                              double * am,    double * ddist_x,
                                              double * dv,    int lddv,
                                              int B, int ghost_col,
                                              mgard_cuda_handle & handle, 
                                              int queue_idx, bool profile);
template mgard_cuda_ret 
solve_tridiag_M_l_row_backward_cuda_sm<float>(int nr,         int nc,
                                              int row_stride, int col_stride,
                                              float * am,    float * ddist_x,
                                              float * dv,    int lddv,
                                              int B, int ghost_col,
                                              mgard_cuda_handle & handle, 
                                              int queue_idx, bool profile);

template mgard_cuda_ret 
solve_tridiag_M_l_row_cuda_sm<double>(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      double * dv,    int lddv,
                                      double * ddist_x,
                                      double * am, double * bm,
                                      int B, int ghost_col,
                                      mgard_cuda_handle & handle, 
                                      int queue_idx, bool profile);
template mgard_cuda_ret 
solve_tridiag_M_l_row_cuda_sm<float>(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      float * dv,    int lddv,
                                      float * ddist_x,
                                      float * am, float * bm,
                                      int B, int ghost_col,
                                      mgard_cuda_handle & handle, 
                                      int queue_idx, bool profile);

template mgard_cuda_ret 
solve_tridiag_M_l_col_forward_cuda_sm<double>(int nr,         int nc,
                                              int row_stride, int col_stride,
                                              double * bm,
                                              double * dv,    int lddv,
                                              int B, int ghost_row,
                                              mgard_cuda_handle & handle, 
                                              int queue_idx, bool profile);
template mgard_cuda_ret 
solve_tridiag_M_l_col_forward_cuda_sm<float>(int nr,         int nc,
                                              int row_stride, int col_stride,
                                              float * bm,
                                              float * dv,    int lddv,
                                              int B, int ghost_row,
                                              mgard_cuda_handle & handle, 
                                              int queue_idx, bool profile);

template mgard_cuda_ret 
solve_tridiag_M_l_col_backward_cuda_sm<double>(int nr,         int nc,
                                              int row_stride, int col_stride,
                                              double * am,    double * ddist_y,
                                              double * dv,    int lddv,
                                              int B, int ghost_row,
                                              mgard_cuda_handle & handle, 
                                              int queue_idx, bool profile);
template mgard_cuda_ret 
solve_tridiag_M_l_col_backward_cuda_sm<float>(int nr,         int nc,
                                              int row_stride, int col_stride,
                                              float * am,    float * ddist_y,
                                              float * dv,    int lddv,
                                              int B, int ghost_row,
                                              mgard_cuda_handle & handle, 
                                              int queue_idx, bool profile);

template mgard_cuda_ret 
solve_tridiag_M_l_col_cuda_sm<double>(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      double * dv,    int lddv,
                                      double * ddist_y,
                                      double * am, double * bm,
                                      int B, int ghost_row,
                                      mgard_cuda_handle & handle, 
                                      int queue_idx, bool profile);
template mgard_cuda_ret 
solve_tridiag_M_l_col_cuda_sm<float>(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      float * dv,    int lddv,
                                      float * ddist_y,
                                      float * am, float * bm,
                                      int B, int ghost_row,
                                      mgard_cuda_handle & handle, 
                                      int queue_idx, bool profile);




}
}
