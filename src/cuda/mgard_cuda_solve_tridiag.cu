#include "cuda/mgard_cuda_solve_tridiag.h"
#include "cuda/mgard_cuda_common_internal.h"

namespace mgard_cuda {

template <typename T>
__global__ void
_solve_tridiag_1(int nrow,       int ncol,
                 int nr,         int nc,
                 int row_stride, int col_stride,
                 int * dirow,    int * dicol, 
                 T * dcoords_x,
                 T * dcoeff,    int lddcoeff,
                 T * dv,        int lddv) {
  int idx0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  T am, bm, h1, h2;
  for (int idx = idx0; idx < nr; idx += (blockDim.x * gridDim.x) * row_stride) {
    int r = dirow[idx];
    T * vec = dv + r * lddv;
    T * coeff = dcoeff + idx * lddcoeff;
    am = 2.0 * _get_dist(dcoords_x, dicol[0], dicol[col_stride]); //dicol[col_stride] - dicol[0]
    bm = _get_dist(dcoords_x, dicol[0], dicol[col_stride]) / am; //dicol[col_stride] - dicol[0]
    int counter = 1;
    coeff[0] = am;
    for (int i = col_stride; i < nc - 1; i += col_stride) {
      h1 = _get_dist(dcoords_x, dicol[i - col_stride], dicol[i]);
      h2 = _get_dist(dcoords_x, dicol[i], dicol[i + col_stride]);
      vec[dicol[i]] -= vec[dicol[i - col_stride]] * bm;
      am = 2.0 * (h1 + h2) - bm * h1;
      bm = h2 / am;
      coeff[counter] = am;
      ++counter;
    }
    h2 = _get_dist(dcoords_x, dicol[nc - 1 - col_stride], dicol[nc - 1]);
    am = 2.0 * h2 - bm * h2;
    vec[dicol[nc - 1]] -= vec[dicol[nc - 1 - col_stride]] * bm;
    coeff[counter] = am;
    /* Start of backward pass */
    vec[dicol[nc - 1]] /= am;
    --counter;
    for (int i = nc - 1 - col_stride; i >= 0; i -= col_stride) {
      h2 = _get_dist(dcoords_x, dicol[i], dicol[i + col_stride]);
      vec[dicol[i]] = (vec[dicol[i]] - h2 * vec[dicol[i + col_stride]]) / coeff[counter];
      --counter;
    }
  }
}

template <typename T>
void
solve_tridiag_1(mgard_cuda_handle<T> & handle,
                int nrow,       int ncol,
                int nr,         int nc,
                int row_stride, int col_stride,
                int * dirow,    int * dicol,
                T * dcoords_x,
                T * dv,     int lddv, 
                int queue_idx) {

  T * dcoeff;
  size_t dcoeff_pitch;
  cudaMallocPitchHelper((void**)&dcoeff, &dcoeff_pitch, nc * sizeof(T), nr);
  int lddcoeff = dcoeff_pitch / sizeof(T);
  cudaMemset2DHelper(dcoeff, dcoeff_pitch, 0, nc * sizeof(T), nr);
  int total_thread = ceil((float)nr / row_stride);
  int tb = min(handle.B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);
  _solve_tridiag_1<<<blockPerGrid, threadsPerBlock,
                     0, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                             nrow,   ncol,
                                             nr,     nc,
                                             row_stride, col_stride,
                                             dirow,  dicol,
                                             dcoords_x,
                                             dcoeff, lddcoeff,
                                             dv,     lddv);
  gpuErrchk(cudaGetLastError ()); 
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void
solve_tridiag_1<double>(mgard_cuda_handle<double> & handle,
                int nrow,       int ncol,
                int nr,         int nc,
                int row_stride, int col_stride,
                int * dirow,    int * dicol,
                double * dcoords_x,
                double * dv,     int lddv, 
                int queue_idx);
template void
solve_tridiag_1<float>(mgard_cuda_handle<float> & handle,
                int nrow,       int ncol,
                int nr,         int nc,
                int row_stride, int col_stride,
                int * dirow,    int * dicol,
                float * dcoords_x,
                float * dv,     int lddv, 
                int queue_idx);

template <typename T>
__global__ void
_solve_tridiag_2(int nrow,       int ncol,
                 int nr,         int nc,
                 int row_stride, int col_stride,
                 int * dirow,    int * dicol,
                 T * dcoords_y,
                 T * dcoeff, int lddcoeff,
                 T * dv,    int lddv) {
  int idx0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  T am, bm, h1, h2;
  for (int idx = idx0; idx < nc; idx += (blockDim.x * gridDim.x) * col_stride) {
    int c = dicol[idx];
    T * vec = dv + c;
    T * coeff = dcoeff + idx;
    am = 2.0 * _get_dist(dcoords_y, dirow[0], dirow[row_stride]); //dirow[row_stride] - dirow[0]
    bm = _get_dist(dcoords_y, dirow[0], dirow[row_stride]) / am; //dirow[row_stride] - dirow[0]
    int counter = 1;
    coeff[0 * lddcoeff] = am;
    for (int i = row_stride; i < nr - 1; i += row_stride) {
      h1 = _get_dist(dcoords_y, dirow[i - row_stride], dirow[i]);
      h2 = _get_dist(dcoords_y, dirow[i], dirow[i + row_stride]);
      vec[dirow[i] * lddv] -= vec[dirow[i - row_stride] * lddv] * bm;
      am = 2.0 * (h1 + h2) - bm * h1;
      bm = h2 / am;
      coeff[counter * lddcoeff] = am;
      ++counter;
    }
    h2 = _get_dist(dcoords_y, dirow[nr - 1 - row_stride], dirow[nr - 1]);
    am = 2.0 * h2 - bm * h2;
    vec[dirow[nr - 1] * lddv] -= vec[dirow[nr - 1 - row_stride] * lddv] * bm;
    coeff[counter * lddcoeff] = am;
    vec[dirow[nr - 1] * lddv] /= am;
    --counter;
    for (int i = nr - 1 - row_stride; i >= 0; i -= row_stride) {
      h2 = _get_dist(dcoords_y, dirow[i], dirow[i + row_stride]);
      vec[dirow[i] * lddv] =
        (vec[dirow[i] * lddv] - h2 * vec[dirow[i + row_stride] * lddv]) /
        coeff[counter * lddcoeff];
      --counter;
    }
  }
}

template <typename T>
void
solve_tridiag_2(mgard_cuda_handle<T> & handle, 
                int nrow,       int ncol,
                int nr,         int nc,
                int row_stride, int col_stride,
                int * dirow,    int * dicol,
                T * dcoords_y,
                T * dv,    int lddv, 
                int queue_idx) {

  T * dcoeff;
  size_t dcoeff_pitch;
  cudaMallocPitchHelper((void**)&dcoeff, &dcoeff_pitch, nc * sizeof(T), nr);
  int lddcoeff = dcoeff_pitch / sizeof(T);
  cudaMemset2DHelper(dcoeff, dcoeff_pitch, 0, nc * sizeof(T), nr);
  int total_thread = ceil((float)nc / col_stride);
  int tb = min(handle.B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);
  _solve_tridiag_2<<<blockPerGrid, threadsPerBlock,
                     0, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                             nrow,       ncol,
                                             nr,         nc,
                                             row_stride, col_stride,
                                             dirow,      dicol,
                                             dcoords_y,
                                             dcoeff, lddcoeff,
                                             dv,         lddv);
  gpuErrchk(cudaGetLastError ()); 
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}


template void
solve_tridiag_2<double>(mgard_cuda_handle<double> & handle, 
                int nrow,       int ncol,
                int nr,         int nc,
                int row_stride, int col_stride,
                int * dirow,    int * dicol,
                double * dcoords_y,
                double * dv,    int lddv, 
                int queue_idx);
template void
solve_tridiag_2<float>(mgard_cuda_handle<float> & handle, 
                int nrow,       int ncol,
                int nr,         int nc,
                int row_stride, int col_stride,
                int * dirow,    int * dicol,
                float * dcoords_y,
                float * dv,    int lddv, 
                int queue_idx);

template <typename T>
__global__ void
_calc_am_bm(int n, T * ddist, T * am, T * bm) {
  int c = threadIdx.x;
  int c_sm = threadIdx.x;
  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);
  T * sm = SharedMemory<T>();
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
    am[c] = am_sm[c_sm];
    bm[c] = bm_sm[c_sm];
  }
}

template <typename T>
void
calc_am_bm(mgard_cuda_handle<T> & handle,
           int n,  T * ddist, T * am, T * bm,    
           int queue_idx) {

  // int total_thread_y = 1;
  int total_thread_x = handle.B;
  int tby = 1;
  int tbx = min(handle.B, total_thread_x);
  size_t sm_size = handle.B * 3 * sizeof(T);
  int gridy = 1;
  int gridx = 1;
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _calc_am_bm<<<blockPerGrid, threadsPerBlock, 
                sm_size, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                              n, ddist, am, bm);
  gpuErrchk(cudaGetLastError ());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void
calc_am_bm<double>(mgard_cuda_handle<double> & handle,
           int n,  double * ddist, double * am, double * bm,    
           int queue_idx);
template void
calc_am_bm<float>(mgard_cuda_handle<float> & handle,
           int n,  float * ddist, float * am, float * bm,    
           int queue_idx);

template <typename T>
__global__ void
_solve_tridiag_forward_1_cpt(int nr,         int nc,
                             int row_stride, int col_stride,
                             T * bm,         
                             int ghost_col,
                             T * dv,         int lddv) {

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
  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);
  T * sm = SharedMemory<T>();
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
void 
solve_tridiag_forward_1_cpt(mgard_cuda_handle<T> & handle,
                            int nr,         int nc,
                            int row_stride, int col_stride,
                            T * bm,
                            T * dv,    int lddv,
                            int queue_idx) {
  int ghost_col = handle.B;
  int total_row = ceil((double)nr/(row_stride));
  //int total_col = 1;
  //int total_thread_y = 1;
  int total_thread_x = total_row;
  int tby = 1;
  int tbx = max(handle.B, min(handle.B, total_thread_x));
  size_t sm_size = (handle.B+1)*(handle.B+ghost_col) * sizeof(T);
  int gridy = 1;
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _solve_tridiag_forward_1_cpt<<<blockPerGrid, threadsPerBlock, 
                                 sm_size, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                                        nr,         nc,
                                                        row_stride, col_stride,
                                                        bm,         
                                                        ghost_col,
                                                        dv,         lddv);
  gpuErrchk(cudaGetLastError ());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
solve_tridiag_forward_1_cpt<double>(mgard_cuda_handle<double> & handle,
                            int nr,         int nc,
                            int row_stride, int col_stride,
                            double * bm,
                            double * dv,    int lddv,
                            int queue_idx);
template void 
solve_tridiag_forward_1_cpt<float>(mgard_cuda_handle<float> & handle,
                            int nr,         int nc,
                            int row_stride, int col_stride,
                            float * bm,
                            float * dv,    int lddv,
                            int queue_idx);


template <typename T>
__global__ void
_solve_tridiag_backward_1_cpt(int nr,             int nc,
                              int row_stride,     int col_stride,
                              T * ddist_x,
                              T * am,       
                              int ghost_col,
                              T * dv,        int lddv) {
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
  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);
  T * sm = SharedMemory<T>();
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
void 
solve_tridiag_backward_1_cpt(mgard_cuda_handle<T> & handle, 
                             int nr,         int nc,
                             int row_stride, int col_stride,
                             T * ddist_x,
                             T * am,    
                             T * dv,    int lddv,
                             int queue_idx) {
  int ghost_col = handle.B;
  int total_row = ceil((double)nr/(row_stride));
  // int total_col = 1;
  // int total_thread_y = 1;
  int total_thread_x = total_row;
  int tby = 1;
  int tbx = max(handle.B, min(handle.B, total_thread_x));
  size_t sm_size = (handle.B+2)*(handle.B+ghost_col) * sizeof(T);
  int gridy = 1;
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _solve_tridiag_backward_1_cpt<<<blockPerGrid, threadsPerBlock, 
                                  sm_size, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                                              nr,         nc,
                                                              row_stride, col_stride,
                                                              ddist_x,
                                                              am,         
                                                              ghost_col,
                                                              dv,         lddv);
  gpuErrchk(cudaGetLastError ());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
solve_tridiag_backward_1_cpt<double>(mgard_cuda_handle<double> & handle, 
                             int nr,         int nc,
                             int row_stride, int col_stride,
                             double * ddist_x,
                             double * am,    
                             double * dv,    int lddv,
                             int queue_idx);
template void 
solve_tridiag_backward_1_cpt<float>(mgard_cuda_handle<float> & handle, 
                             int nr,         int nc,
                             int row_stride, int col_stride,
                             float * ddist_x,
                             float * am,    
                             float * dv,    int lddv,
                             int queue_idx);


template <typename T>
void 
solve_tridiag_1_cpt(mgard_cuda_handle<T> & handle, 
                    int nr,         int nc,
                    int row_stride, int col_stride,
                    T * ddist_x,
                    T * am, T * bm, 
                    T * dv,    int lddv,
                    int queue_idx) {

  calc_am_bm(handle, 
             ceil((float)nc/col_stride), ddist_x, am, bm,
             queue_idx);
  
  solve_tridiag_forward_1_cpt(handle,
                              nr,         nc,
                              row_stride, col_stride,
                              bm,
                              dv,    lddv,
                              queue_idx);

  solve_tridiag_backward_1_cpt(handle,
                               nr,         nc,
                               row_stride, col_stride,
                               ddist_x,
                               am,     
                               dv,    lddv,
                               queue_idx);
}

template void 
solve_tridiag_1_cpt<double>(mgard_cuda_handle<double> & handle, 
                    int nr,         int nc,
                    int row_stride, int col_stride,
                    double * ddist_x,
                    double * am, double * bm,
                    double * dv,    int lddv, 
                    int queue_idx);
template void 
solve_tridiag_1_cpt<float>(mgard_cuda_handle<float> & handle, 
                    int nr,         int nc,
                    int row_stride, int col_stride,
                    float * ddist_x,
                    float * am, float * bm, 
                    float * dv,    int lddv,
                    int queue_idx);

template <typename T>
__global__ void
_solve_tridiag_forward_2_cpt(int nr,             int nc,
                             int row_stride,     int col_stride,
                             T * bm, 
                             int ghost_row,
                             T * dv,        int lddv) {

  /* Global idx */
  register int c0 = blockIdx.x * blockDim.x;
  register int c0_stride = c0 * col_stride;
  register int r = 0;

  /* Local col idx */
  // register int r_sm = threadIdx.x; // for computation
  register int c_sm = threadIdx.x; // for load data

  T * vec;

  /* SM allocation */
  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);
  T * sm = SharedMemory<T>();
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
void 
solve_tridiag_forward_2_cpt(mgard_cuda_handle<T> & handle, 
                            int nr,         int nc,
                            int row_stride, int col_stride,
                            T * bm,
                            T * dv,    int lddv,
                            int queue_idx) {
  int ghost_row = handle.B;
  // int total_row = 1;
  int total_col = ceil((double)nc/(col_stride));
  // int total_thread_y = 1;
  int total_thread_x = total_col;
  int tby = 1;
  int tbx = min(handle.B, total_thread_x);
  tbx = max(handle.B, tbx);
  size_t sm_size = (handle.B+1)*(handle.B+ghost_row) * sizeof(T);
  int gridy = 1;
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _solve_tridiag_forward_2_cpt<<<blockPerGrid, threadsPerBlock, 
                                 sm_size, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                                              nr,         nc,
                                                              row_stride, col_stride,
                                                              bm,
                                                              ghost_row,
                                                              dv,         lddv);
  gpuErrchk(cudaGetLastError ());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}


template void 
solve_tridiag_forward_2_cpt<double>(mgard_cuda_handle<double> & handle, 
                            int nr,         int nc,
                            int row_stride, int col_stride,
                            double * bm,
                            double * dv,    int lddv,
                            int queue_idx);
template void 
solve_tridiag_forward_2_cpt<float>(mgard_cuda_handle<float> & handle, 
                            int nr,         int nc,
                            int row_stride, int col_stride,
                            float * bm,
                            float * dv,    int lddv,
                            int queue_idx);

template <typename T>
__global__ void
_solve_tridiag_backward_2_cpt(int nr,             int nc,
                              int row_stride,     int col_stride,
                              T * ddist_x,
                              T * am,
                              int ghost_row,        
                              T * dv,        int lddv) {
  /* Global idx */
  register int c0 = blockIdx.x * blockDim.x;
  register int c0_stride = c0 * col_stride;
  register int r = 0;

  /* Local col idx */
  // register int r_sm = threadIdx.x; // for computation
  register int c_sm = threadIdx.x; // for load data

  T * vec;

  /* SM allocation */
  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);
  T * sm = SharedMemory<T>();
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
void 
solve_tridiag_backward_2_cpt(mgard_cuda_handle<T> & handle, 
                             int nr,         int nc,
                             int row_stride, int col_stride,
                             T * ddist_y,
                             T * am,    
                             T * dv,    int lddv,
                             int queue_idx) {

  int ghost_row = handle.B;
  // int total_row = 1;
  int total_col = ceil((double)nc/(col_stride));
  // int total_thread_y = 1;
  int total_thread_x = total_col;
  int tby = 1;
  int tbx = min(handle.B, total_thread_x);
  tbx = max(handle.B, tbx);
  size_t sm_size = (handle.B+2)*(handle.B+ghost_row) * sizeof(T);
  int gridy = 1;
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _solve_tridiag_backward_2_cpt<<<blockPerGrid, threadsPerBlock, 
                                  sm_size, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                                              nr,         nc,
                                                              row_stride, col_stride,
                                                              ddist_y,
                                                              am,
                                                              ghost_row,
                                                              dv,         lddv);
  gpuErrchk(cudaGetLastError ());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
solve_tridiag_backward_2_cpt<double>(mgard_cuda_handle<double> & handle, 
                             int nr,         int nc,
                             int row_stride, int col_stride,
                             double * ddist_y,
                             double * am,    
                             double * dv,    int lddv,
                             int queue_idx);
template void 
solve_tridiag_backward_2_cpt<float>(mgard_cuda_handle<float> & handle, 
                             int nr,         int nc,
                             int row_stride, int col_stride,
                             float * ddist_y,
                             float * am,    
                             float * dv,    int lddv,
                             int queue_idx);

template <typename T>
void 
solve_tridiag_2_cpt(mgard_cuda_handle<T> & handle,
                    int nr,         int nc,
                    int row_stride, int col_stride,
                    T * ddist_y,
                    T * am, T * bm, 
                    T * dv,    int lddv,
                    int queue_idx) {


  calc_am_bm(handle,
             ceil((float)nr/row_stride), ddist_y, am, bm,
             queue_idx);

  solve_tridiag_forward_2_cpt(handle,
                              nr,         nc,
                              row_stride, col_stride,
                              bm,
                              dv,         lddv,
                              queue_idx);

  solve_tridiag_backward_2_cpt(handle,
                               nr,         nc,
                               row_stride, col_stride,
                               ddist_y,
                               am,         
                               dv,         lddv,
                               queue_idx);
}

template void 
solve_tridiag_2_cpt<double>(mgard_cuda_handle<double> & handle,
                    int nr,         int nc,
                    int row_stride, int col_stride,
                    double * ddist_y,
                    double * am, double * bm, 
                    double * dv,    int lddv,
                    int queue_idx);
template void 
solve_tridiag_2_cpt<float>(mgard_cuda_handle<float> & handle,
                    int nr,         int nc,
                    int row_stride, int col_stride,
                    float * ddist_y,
                    float * am, float * bm, 
                    float * dv,    int lddv,
                    int queue_idx);

}
