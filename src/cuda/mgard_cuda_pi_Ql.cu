#include "cuda/mgard_cuda_pi_Ql.h"
#include "cuda/mgard_cuda_common_internal.h"

namespace mgard_cuda {

template <typename T>
__global__ void 
_pi_Ql(int nrow,           int ncol,
            int nr,             int nc,
            int row_stride,     int col_stride,
            int * irow,         int * icol,
            T * dcoords_y, T * dcoords_x,
            T * dv,        int lddv) {

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

      int h1_col = _get_dist(dcoords_x, icol[x], icol[x + col_stride]);  //icol[x+col_stride]  - icol[x];
      int h2_col = _get_dist(dcoords_x, icol[x + col_stride], icol[x + col_Cstride]);  //icol[x+col_Cstride] - icol[x+col_stride];
      int hsum_col = h1_col + h2_col;
   
      int h1_row = _get_dist(dcoords_y, irow[y], irow[y + row_stride]);  //irow[y+row_stride]  - irow[y];
      int h2_row = _get_dist(dcoords_y, irow[y + row_stride], irow[y + row_Cstride]);  //irow[y+row_Cstride] - irow[y+row_stride];
      int hsum_row = h1_row + h2_row;
      a01 -= (h1_col * a02 + h2_col * a00) / hsum_col;
      a10 -= (h1_row * a20 + h2_row * a00) / hsum_row;
      a11 -= 1.0 / (hsum_row * hsum_col) * (a00 * h2_col * h2_row + a02 * h1_col * h2_row + a20 * h2_col * h1_row + a22 * h1_col * h1_row);
      
      dv[get_idx(lddv, irow[y],             icol[x+col_stride]  )] = a01;
      dv[get_idx(lddv, irow[y+row_stride],  icol[x]             )] = a10;
      dv[get_idx(lddv, irow[y+row_stride],  icol[x+col_stride]  )] = a11;

      if (x + col_Cstride == nc - 1) {
        a12 -= (h1_row * a22 + h2_row * a02) / hsum_row;
        dv[get_idx(lddv, irow[y+row_stride],  icol[x+col_Cstride] )] = a12;
      }
      if (y + row_Cstride == nr - 1) {
        a21 -= (h1_col * a22 + h2_col * a20) / hsum_col;
        dv[get_idx(lddv, irow[y+row_Cstride], icol[x+col_stride]  )] = a21;
      }
    }
  }

}

template <typename T>
void 
pi_Ql(mgard_cuda_handle<T> & handle,
      int nrow,           int ncol,
      int nr,             int nc,
      int row_stride,     int col_stride,
      int * dirow,        int * dicol,
      T * dcoords_y, T * dcoords_x,
      T * dv,        int lddv, 
      int queue_idx) {

  int total_thread_y = floor((double)nr/(row_stride * 2));
  int total_thread_x = floor((double)nc/(col_stride * 2));
  int tby = min(handle.B, total_thread_y);
  int tbx = min(handle.B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _pi_Ql<<<blockPerGrid, threadsPerBlock,
           0, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                             nrow,       ncol,
                                             nr,         nc,
                                             row_stride, col_stride,
                                             dirow,      dicol,
                                             dcoords_y,  dcoords_x,
                                             dv,         lddv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}


template void 
pi_Ql<double>(mgard_cuda_handle<double> & handle,
              int nrow,           int ncol,
              int nr,             int nc,
              int row_stride,     int col_stride,
              int * dirow,        int * dicol,
              double * dcoords_y, double * dcoords_x,
              double * dv,        int lddv, 
              int queue_idx);
template void 
pi_Ql<float>(mgard_cuda_handle<float> & handle,
             int nrow,           int ncol,
             int nr,             int nc,
             int row_stride,     int col_stride,
             int * dirow,        int * dicol,
             float * dcoords_y, float * dcoords_x,
             float * dv,        int lddv, 
             int queue_idx);

template <typename T>
__global__ void 
_pi_Ql_cpt(int nr,         int nc,
           int row_stride, int col_stride,
           T * ddist_y,    T * ddist_x,
           T * dv,         int lddv) {

  register int c0 = blockIdx.x * blockDim.x;
  //register int c0_stride = c0 * col_stride;
  register int r0 = blockIdx.y * blockDim.y;
  //register int r0_stride = r0 * row_stride;

  register int total_row = ceil((double)nr/(row_stride));
  register int total_col = ceil((double)nc/(col_stride));

  register int c_sm = threadIdx.x;
  register int r_sm = threadIdx.y;

  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);
  T * sm = SharedMemory<T>();

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
void 
pi_Ql_cpt(mgard_cuda_handle<T> & handle,
          int nr,         int nc,
          int row_stride, int col_stride,
          T * ddist_y, T * ddist_x,
          T * dv,    int lddv,
          int queue_idx) {

  int total_row = ceil((double)nr/(row_stride));
  int total_col = ceil((double)nc/(col_stride));
  int total_thread_y = total_row - 1;
  int total_thread_x = total_col - 1;
  int tby = min(handle.B, total_thread_y);
  int tbx = min(handle.B, total_thread_x);
  size_t sm_size = ((handle.B+1) * (handle.B+1) + 2 * handle.B) * sizeof(T);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _pi_Ql_cpt<<<blockPerGrid, threadsPerBlock, 
               sm_size, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                          nr,         nc,
                                          row_stride, col_stride,
                                          ddist_y,    ddist_x,
                                          dv,         lddv);
  gpuErrchk(cudaGetLastError ());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
pi_Ql_cpt<double>(mgard_cuda_handle<double> & handle,
          int nr,         int nc,
          int row_stride, int col_stride,
          double * ddist_y, double * ddist_x,
          double * dv,    int lddv,
          int queue_idx);
template void 
pi_Ql_cpt<float>(mgard_cuda_handle<float> & handle,
          int nr,         int nc,
          int row_stride, int col_stride,
          float * ddist_y, float * ddist_x,
          float * dv,    int lddv,
          int queue_idx);

template <typename T>
__global__ void 
_pi_Ql_first_1(const int nrow, const int ncol,
               const int nr,   const int nc,
               int * irow,     int * icol_p,
               T * dist_r,     T * dist_c,
               T * v,          int ldv) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  for (int y = y0; y < nr; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < ncol-nc; x += blockDim.x * gridDim.x) {
      int r = irow[y];
      int c = icol_p[x];
      register T center = v[get_idx(ldv, r, c    )];
      register T left   = v[get_idx(ldv, r, c - 1)];
      register T right  = v[get_idx(ldv, r, c + 1)];
      register T h1     = dist_c[c - 1];
      register T h2     = dist_c[c];
      center -= (h2 * left + h1 * right) / (h1 + h2);
      v[get_idx(ldv, r, c)] = center;
    }
  }
}

template <typename T>
void 
pi_Ql_first_1(mgard_cuda_handle<T> & handle, 
              const int nrow, const int ncol,
              const int nr,   const int nc,
              int * dirow,    int * dicol_p,
              T * ddist_r,    T * ddist_c,
              T * dv,         int lddv,
              int queue_idx) {  

  int total_thread_x = ncol - nc;
  int total_thread_y = nr;
  if (total_thread_y == 0 || total_thread_x == 0) return; 
  int tbx = min(handle.B, total_thread_x);
  int tby = min(handle.B, total_thread_y);
  int gridx = ceil((float)total_thread_x/tbx);
  int gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _pi_Ql_first_1<<<blockPerGrid, threadsPerBlock, 
                   0, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                                  nrow,    ncol,
                                                  nr,      nc,
                                                  dirow,   dicol_p,
                                                  ddist_r, ddist_c,
                                                  dv,      lddv);
  gpuErrchk(cudaGetLastError()); 
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
pi_Ql_first_1<double>(mgard_cuda_handle<double> & handle, 
              const int nrow, const int ncol,
              const int nr,   const int nc,
              int * dirow,    int * dicol_p,
              double * ddist_r,    double * ddist_c,
              double * dv,         int lddv,
              int queue_idx);
template void 
pi_Ql_first_1<float>(mgard_cuda_handle<float> & handle, 
              const int nrow, const int ncol,
              const int nr,   const int nc,
              int * dirow,    int * dicol_p,
              float * ddist_r,    float * ddist_c,
              float * dv,         int lddv,
              int queue_idx);

template <typename T>
__global__ void 
_pi_Ql_first_2(const int nrow, const int ncol,
               const int nr,   const int nc,
               int * irow_p,   int * icol,
               T * dist_r,     T * dist_c,
               T * v,          int ldv) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  for (int y = y0; y < nrow-nr; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x) {
      int r = irow_p[y];
      int c = icol[x];
      register T center = v[get_idx(ldv, r,     c)];
      register T up     = v[get_idx(ldv,   r - 1, c)];
      register T down   = v[get_idx(ldv, r + 1, c)];
      register T h1     = dist_r[r - 1];
      register T h2     = dist_r[r];
      center -= (h2 * up + h1 * down) / (h1 + h2);
      v[get_idx(ldv, r, c)] = center;
    }
  }
}

template <typename T>
void 
pi_Ql_first_2(mgard_cuda_handle<T> & handle, 
              const int nrow, const int ncol,
              const int nr,   const int nc,
              int * dirow_p,  int * dicol,
              T * ddist_r,    T * ddist_c,
              T * dv,         int lddv,
              int queue_idx) {

  int total_thread_x = nc;
  int total_thread_y = nrow - nr;
  if (total_thread_y == 0 || total_thread_x == 0) return; 
  int tbx = min(handle.B, total_thread_x);
  int tby = min(handle.B, total_thread_y);
  int gridx = ceil((float)total_thread_x/tbx);
  int gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  _pi_Ql_first_2<<<blockPerGrid, threadsPerBlock, 
                   0, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                                  nrow,    ncol,
                                                  nr,      nc,
                                                  dirow_p, dicol,
                                                  ddist_r, ddist_c,
                                                  dv,      lddv);
  gpuErrchk(cudaGetLastError ()); 
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif 
}

template void 
pi_Ql_first_2<double>(mgard_cuda_handle<double> & handle, 
              const int nrow, const int ncol,
              const int nr,   const int nc,
              int * dirow_p,  int * dicol,
              double * ddist_r,    double * ddist_c,
              double * dv,         int lddv,
              int queue_idx);
template void 
pi_Ql_first_2<float>(mgard_cuda_handle<float> & handle, 
              const int nrow, const int ncol,
              const int nr,   const int nc,
              int * dirow_p,  int * dicol,
              float * ddist_r,    float * ddist_c,
              float * dv,         int lddv,
              int queue_idx);

template <typename T>
__global__ void 
_pi_Ql_first_12(const int nrow, const int ncol,
                const int nr,   const int nc,
                int * irow_p,   int * icol_p,
                T * dist_r,     T * dist_c,
                T * v,          int ldv) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  for (int y = y0; y < nrow-nr; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < ncol-nc; x += blockDim.x * gridDim.x) {
      int r = irow_p[y];
      int c = icol_p[x];
      register T center    = v[get_idx(ldv, r,     c    )];
      register T upleft    = v[get_idx(ldv, r - 1, c - 1)];
      register T upright   = v[get_idx(ldv, r - 1, c + 1)];
      register T downleft  = v[get_idx(ldv, r + 1, c - 1)];
      register T downright = v[get_idx(ldv, r + 1, c + 1)];
      register T h1_c = dist_c[c-1];
      register T h2_c = dist_c[c];
      register T h1_r = dist_r[r-1];
      register T h2_r = dist_r[r];
      center -= (upleft * h2_c * h2_r + upright * h1_c * h2_r + 
                 downleft * h2_c * h1_r + downright * h1_c * h1_r)
                 /((h1_c+h2_c)*(h1_r+h2_r));
      v[get_idx(ldv, r, c)] = center;
    }
  }
}

template <typename T>
void 
pi_Ql_first_12(mgard_cuda_handle<T> & handle, 
               const int nrow, const int ncol,
               const int nr,    const int nc,
               int * dirow_p,   int * dicol_p,
               T * ddist_r,     T * ddist_c,
               T * dv,          int lddv,
               int queue_idx) {  

  int total_thread_x = ncol - nc;
  int total_thread_y = nrow - nr;
  if (total_thread_y == 0 || total_thread_x == 0) return; 
  int tbx = min(handle.B, total_thread_x);
  int tby = min(handle.B, total_thread_y);
  int gridx = ceil((float)total_thread_x/tbx);
  int gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _pi_Ql_first_12<<<blockPerGrid, threadsPerBlock, 
                    0, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                                nrow,    ncol,
                                                nr,      nc,
                                                dirow_p, dicol_p,
                                                ddist_r, ddist_c,
                                                dv,      lddv);
  gpuErrchk(cudaGetLastError ()); 
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif   
}

template void 
pi_Ql_first_12<double>(mgard_cuda_handle<double> & handle, 
               const int nrow, const int ncol,
               const int nr,    const int nc,
               int * dirow_p,   int * dicol_p,
               double * ddist_r,     double * ddist_c,
               double * dv,          int lddv,
               int queue_idx);
template void 
pi_Ql_first_12<float>(mgard_cuda_handle<float> & handle, 
               const int nrow, const int ncol,
               const int nr,    const int nc,
               int * dirow_p,   int * dicol_p,
               float * ddist_r,     float * ddist_c,
               float * dv,          int lddv,
               int queue_idx);

template <typename T>
__global__ void 
_pi_Ql_cpt(int nr,         int nc,         int nf, 
           int row_stride, int col_stride, int fib_stride,
           T * ddist_r,    T * ddist_c,    T * ddist_f,
           T * dv,         int lddv1,      int lddv2) {

  register int r0 = blockIdx.z * blockDim.z;
  register int c0 = blockIdx.y * blockDim.y;
  register int f0 = blockIdx.x * blockDim.x;
    
  register int total_row = ceil((double)nr/(row_stride));
  register int total_col = ceil((double)nc/(col_stride));
  register int total_fib = ceil((double)nf/(fib_stride));

  register int r_sm = threadIdx.z;
  register int c_sm = threadIdx.y;
  register int f_sm = threadIdx.x;

  register int r_sm_ex = blockDim.z;
  register int c_sm_ex = blockDim.y;
  register int f_sm_ex = blockDim.x;

  register int r_gl;
  register int c_gl;
  register int f_gl;

  register int r_gl_ex;
  register int c_gl_ex;
  register int f_gl_ex;

  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);

  T * sm = SharedMemory<T>();

  //extern __shared__ double sm[]; // size: (blockDim.x + 1) * (blockDim.y + 1) * (blockDim.z + 1)
  int ldsm1 = blockDim.x + 1;
  int ldsm2 = blockDim.y + 1;
  T * v_sm = sm;
  T * dist_f_sm = sm + (blockDim.x + 1) * (blockDim.y + 1) * (blockDim.z + 1);
  T * dist_c_sm = dist_f_sm + blockDim.x;
  T * dist_r_sm = dist_c_sm + blockDim.y;

  for (int r = r0; r < total_row - 1; r += blockDim.z * gridDim.z) {
    r_gl = (r + r_sm) * row_stride;
    r_gl_ex = (r + blockDim.z) * row_stride;
    for (int c = c0; c < total_col - 1; c += blockDim.y * gridDim.y) {
      c_gl = (c + c_sm) * col_stride;
      c_gl_ex = (c + blockDim.y) * col_stride;
      for (int f = f0; f < total_fib - 1; f += blockDim.x * gridDim.x) {
        f_gl = (f + f_sm) * fib_stride;
        f_gl_ex = (f + blockDim.x) * fib_stride;
        /* Load v */
        if (r + r_sm < total_row && c + c_sm < total_col && f + f_sm < total_fib) {
          // load cubic
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl)];
          // load extra surfaces
          if (r + blockDim.z < total_row && r_sm == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)] = dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl)];
          }
          if (c + blockDim.y < total_col && c_sm == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)] = dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl)];
          }
          if (f + blockDim.x < total_fib && f_sm == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)] = dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl_ex)];
          }
          // load extra edges
          if (c + blockDim.y < total_col && f + blockDim.x < total_fib && c_sm == 0 && f_sm == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)] = dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)];
          }
          if (r + blockDim.z < total_row && f + blockDim.x < total_fib && r_sm == 0 && f_sm == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)] = dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)];
          }
          if (r + blockDim.z < total_row && c + blockDim.y < total_col && r_sm == 0 && c_sm == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)] = dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)];
          }
          // load extra vertex
          if (r + blockDim.z < total_row && c + blockDim.y < total_col && f + blockDim.x < total_fib &&
              r_sm == 0 && c_sm == 0 && f_sm == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm_ex)] = dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)];
          }

          // load dist
          if (c_sm == 0 && f_sm == 0 && r + r_sm < total_row) {
            dist_r_sm[r_sm] = ddist_r[r + r_sm];
          }
          if (r_sm == 0 && f_sm == 0 && c + c_sm < total_col) {
            dist_c_sm[c_sm] = ddist_c[c + c_sm];
          }
          if (c_sm == 0 && r_sm == 0 && f + f_sm < total_fib) {
            dist_f_sm[f_sm] = ddist_f[f + f_sm];
          }
          __syncthreads();

          T h1_row = dist_r_sm[r_sm - 1];
          T h2_row = dist_r_sm[r_sm];
          T h1_col = dist_c_sm[c_sm - 1];
          T h2_col = dist_c_sm[c_sm];
          T h1_fib = dist_f_sm[f_sm - 1];
          T h2_fib = dist_f_sm[f_sm];

          /* Compute */
          // edges
          if (r_sm % 2 != 0 && c_sm % 2 == 0 && f_sm % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] -= 
              (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm, f_sm)] * h2_row + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm, f_sm)] * h1_row) / 
              (h1_row + h2_row);
          }
          if (r_sm % 2 == 0 && c_sm % 2 != 0 && f_sm % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] -= 
              (v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm-1, f_sm)] * h2_col + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm+1, f_sm)] * h1_col) / 
              (h1_col + h2_col);
          }
          if (r_sm % 2 == 0 && c_sm % 2 == 0 && f_sm % 2 != 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] -= 
              (v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm-1)] * h2_fib + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm+1)] * h1_fib) / 
              (h1_fib + h2_fib);
          }
          // surfaces
          if (r_sm % 2 == 0 && c_sm % 2 != 0 && f_sm % 2 != 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] -= 
              (v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm-1, f_sm-1)] * h2_col * h2_fib + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm+1, f_sm-1)] * h1_col * h2_fib + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm-1, f_sm+1)] * h2_col * h1_fib +
               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm+1, f_sm+1)] * h1_col * h1_fib) / 
              ((h1_col + h2_col) * (h1_fib + h2_fib));
          }
          if (r_sm % 2 != 0 && c_sm % 2 == 0 && f_sm % 2 != 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] -= 
              (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm, f_sm-1)] * h2_row * h2_fib + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm, f_sm-1)] * h1_row * h2_fib + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm, f_sm+1)] * h2_row * h1_fib +
               v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm, f_sm+1)] * h1_row * h1_fib) / 
              ((h1_row + h2_row) * (h1_fib + h2_fib));
          }
          if (r_sm % 2 != 0 && c_sm % 2 != 0 && f_sm % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] -= 
              (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm-1, f_sm)] * h2_row * h2_col + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm-1, f_sm)] * h1_row * h2_col + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm+1, f_sm)] * h2_row * h1_col +
               v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm)] * h1_row * h1_col) / 
              ((h1_row + h2_row) * (h1_col + h2_col));
          }

          // core
          if (r_sm % 2 != 0 && c_sm % 2 != 0 && f_sm % 2 != 0) {

            T x00 = (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm-1, f_sm-1)] * h2_fib + 
                          v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm-1, f_sm+1)] * h1_fib) /
                         (h2_fib + h1_fib);
            T x01 = (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm+1, f_sm-1)] * h2_fib + 
                          v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm+1, f_sm+1)] * h1_fib) /
                         (h2_fib + h1_fib);
            T x10 = (v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm-1, f_sm-1)] * h2_fib + 
                          v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm-1, f_sm+1)] * h1_fib) /
                         (h2_fib + h1_fib);
            T x11 = (v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm-1)] * h2_fib + 
                          v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm+1)] * h1_fib) /
                         (h2_fib + h1_fib);
            T y0  = (h2_col * x00 + h1_col * x01) / (h2_col + h1_col);
            T y1  = (h2_col * x10 + h1_col * x11) / (h2_col + h1_col);
            T z   = (h2_row * y0 + h1_row * y1) / (h2_row + h1_row);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] -= z;
          }

          // store
          dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl)] = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];

          /* extra computaion for global boarder */
          // extra surface
          if (r + blockDim.z == total_row - 1) {
            if (r_sm == 0) {
              //edge
              if (c_sm % 2 != 0 && f_sm % 2 == 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm-1, f_sm)] * h2_col + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm+1, f_sm)] * h1_col) / 
                  (h1_col + h2_col);
              }
              if (c_sm % 2 == 0 && f_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm-1)] * h2_fib + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm+1)] * h1_fib) / 
                  (h1_fib + h2_fib);
              }
              //surface
              if (c_sm % 2 != 0 && f_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm-1, f_sm-1)] * h2_col * h2_fib + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm+1, f_sm-1)] * h1_col * h2_fib + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm-1, f_sm+1)] * h2_col * h1_fib +
                   v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm+1, f_sm+1)] * h1_col * h1_fib) / 
                  ((h1_col + h2_col) * (h1_fib + h2_fib));
              }
              dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl)] = v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)];
            }
          }

          if (c + blockDim.y == total_col - 1) {
            if (c_sm == 0) {
              //edge
              if (r_sm % 2 != 0 && f_sm % 2 == 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm_ex, f_sm)] * h2_row + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm_ex, f_sm)] * h1_row) / 
                  (h1_row + h2_row);
              }
              if (r_sm % 2 == 0 && f_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm-1)] * h2_fib + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm+1)] * h1_fib) / 
                  (h1_fib + h2_fib);
              }
              //surface
              if (r_sm % 2 != 0 && f_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm_ex, f_sm-1)] * h2_row * h2_fib + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm_ex, f_sm-1)] * h1_row * h2_fib + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm_ex, f_sm+1)] * h2_row * h1_fib +
                   v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm_ex, f_sm+1)] * h1_row * h1_fib) / 
                  ((h1_row + h2_row) * (h1_fib + h2_fib));
              }
              dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl)] = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)];
            }
          }

          if (f + blockDim.x == total_fib - 1) {
            if (f_sm == 0) {
              //edge
              if (r_sm % 2 != 0 && c_sm % 2 == 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm, f_sm_ex)] * h2_row + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm, f_sm_ex)] * h1_row) / 
                  (h1_row + h2_row);
              }
              if (r_sm % 2 == 0 && c_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm-1, f_sm_ex)] * h2_col + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm+1, f_sm_ex)] * h1_col) / 
                  (h1_col + h2_col);
              }
              //surface
              if (r_sm % 2 != 0 && c_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm-1, f_sm_ex)] * h2_row * h2_col + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm-1, f_sm_ex)] * h1_row * h2_col + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm+1, f_sm_ex)] * h2_row * h1_col +
                   v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm_ex)] * h1_row * h1_col) / 
                  ((h1_row + h2_row) * (h1_col + h2_col));
              }
              dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl_ex)] = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)];
            }
          }

          //edge
          if (c + blockDim.y == total_col - 1 && f + blockDim.x == total_fib - 1) {
            if (c_sm == 0 && f_sm == 0) {
              if (r_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm_ex, f_sm_ex)] * h2_row + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm_ex, f_sm_ex)] * h1_row) / 
                  (h1_row + h2_row);
              }
              dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)] = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)];
            }
          }

          if (r + blockDim.z == total_row - 1 && f + blockDim.x == total_fib - 1) {
            if (r_sm == 0 && f_sm == 0) {
              if (c_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm-1, f_sm_ex)] * h2_col + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm+1, f_sm_ex)] * h1_col) / 
                  (h1_col + h2_col);
              }
              dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)] = v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)];
            }
          }

          if (r + blockDim.z == total_row - 1 && c + blockDim.y == total_col - 1) {
            if (r_sm == 0 && c_sm == 0) {
              if (f_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm-1)] * h2_fib + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm+1)] * h1_fib) / 
                  (h1_fib + h2_fib);  
              }
              dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)] = v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)];
            }
          }
        }// restrict boundary
      } // end f
    } // end c
  } // end r
}

template <typename T>
void 
pi_Ql_cpt(mgard_cuda_handle<T> & handle, 
          int nr,           int nc,           int nf, 
          int row_stride,   int col_stride,   int fib_stride, 
          T * ddist_r, T * ddist_c, T * ddist_f,
          T * dv,      int lddv1,        int lddv2, 
          int queue_idx) {
    
  int B_adjusted = min(8, handle.B);  
  int total_row = ceil((double)nr/(row_stride));
  int total_col = ceil((double)nc/(col_stride));
  int total_fib = ceil((double)nf/(fib_stride));
  int total_thread_z = total_row - 1;
  int total_thread_y = total_col - 1;
  int total_thread_x = total_fib - 1;
  int tbz = min(B_adjusted, total_thread_z);
  int tby = min(B_adjusted, total_thread_y);
  int tbx = min(B_adjusted, total_thread_x);
  size_t sm_size = ((B_adjusted+1) * (B_adjusted+1) * (B_adjusted+1) + 3 * B_adjusted) * sizeof(T);
  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  _pi_Ql_cpt<<<blockPerGrid, threadsPerBlock, 
                       sm_size, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                          nr,         nc,         nf,
                                          row_stride, col_stride, fib_stride,
                                          ddist_r,    ddist_c,     ddist_f, 
                                          dv,         lddv1,      lddv2);
  gpuErrchk(cudaGetLastError ());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
pi_Ql_cpt<double>(mgard_cuda_handle<double> & handle, 
          int nr,           int nc,           int nf, 
          int row_stride,   int col_stride,   int fib_stride, 
          double * ddist_r, double * ddist_c, double * ddist_f,
          double * dv,      int lddv1,        int lddv2, 
          int queue_idx);
template void 
pi_Ql_cpt<float>(mgard_cuda_handle<float> & handle, 
          int nr,           int nc,           int nf, 
          int row_stride,   int col_stride,   int fib_stride, 
          float * ddist_r, float * ddist_c, float * ddist_f,
          float * dv,      int lddv1,        int lddv2, 
          int queue_idx);



template <typename T>
__global__ void 
_pi_Ql_first_1(int nrow,        int ncol,         int nfib,
               int nr,          int nc,           int nf, 
               int * irow,      int * icol,       int * ifib_p,
               T * dist_r,      T * dist_c,       T * dist_f,
               T * v,           int ldv1,         int ldv2) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;
  for (int z = z0; z < nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nfib-nf; x += blockDim.x * gridDim.x) {
        int f = ifib_p[x];
        int c = icol[y];
        int r = irow[z];
        register T left = v[get_idx(ldv1, ldv2, r, c, f-1)];
        register T right = v[get_idx(ldv1, ldv2, r, c, f+1)];
        register T center = v[get_idx(ldv1, ldv2, r, c, f)];
        register T h1 = dist_f[f-1];
        register T h2 = dist_f[f];
        center -= (h2 * left + h1 * right) / (h1 + h2);
        v[get_idx(ldv1, ldv2, r, c, f)] = center;
      }
    }
  }
}


template <typename T>
void 
pi_Ql_first_1(mgard_cuda_handle<T> & handle,
              int nrow,        int ncol,         int nfib,
              int nr,          int nc,           int nf, 
              int * dirow,     int * dicol,      int * difib_p,
              T * ddist_r, T * ddist_c, T * ddist_f,
              T * dv,      int lddv1,        int lddv2, 
              int queue_idx) {
 
  int B_adjusted = min(8, handle.B);  
  int total_thread_z = nr;
  int total_thread_y = nc;
  int total_thread_x = nfib - nf;
  if (total_thread_z == 0 || total_thread_y == 0 || total_thread_x == 0) return; 
  int tbz = min(B_adjusted, total_thread_z);
  int tby = min(B_adjusted, total_thread_y);
  int tbx = min(B_adjusted, total_thread_x);
  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  _pi_Ql_first_1<<<blockPerGrid, threadsPerBlock, 
                   0, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                          nrow,       ncol,       nfib,
                                          nr,         nc,         nf,
                                          dirow,      dicol,      difib_p,
                                          ddist_r,    ddist_c,    ddist_f,
                                          dv,         lddv1,      lddv2);
  gpuErrchk(cudaGetLastError ());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
pi_Ql_first_1<double>(mgard_cuda_handle<double> & handle,
              int nrow,        int ncol,         int nfib,
              int nr,          int nc,           int nf, 
              int * dirow,     int * dicol,      int * difib_p,
              double * ddist_r, double * ddist_c, double * ddist_f,
              double * dv,      int lddv1,        int lddv2, 
              int queue_idx);
template void 
pi_Ql_first_1<float>(mgard_cuda_handle<float> & handle,
              int nrow,        int ncol,         int nfib,
              int nr,          int nc,           int nf, 
              int * dirow,     int * dicol,      int * difib_p,
              float * ddist_r, float * ddist_c, float * ddist_f,
              float * dv,      int lddv1,        int lddv2, 
              int queue_idx);

template <typename T>
__global__ void 
_pi_Ql_first_2(int nrow,        int ncol,         int nfib,
               int nr,          int nc,           int nf, 
               int * irow,      int * icol_p,      int * ifib,
               T * dist_r,      T * dist_c,       T * dist_f,
               T * v,           int ldv1,         int ldv2) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;
  for (int z = z0; z < nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < ncol-nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nf; x += blockDim.x * gridDim.x) {
        int f = ifib[x];
        int c = icol_p[y];
        int r = irow[z];
        register T front = v[get_idx(ldv1, ldv2, r, c-1, f)];
        register T back = v[get_idx(ldv1, ldv2, r, c+1, f)];
        register T center = v[get_idx(ldv1, ldv2, r, c, f)];
        register T h1 = dist_c[c-1];
        register T h2 = dist_c[c];
        center -= (h2 * front + h1 * back) / (h1 + h2);
        v[get_idx(ldv1, ldv2, r, c, f)] = center;
      }
    }
  }
}


template <typename T>
void 
pi_Ql_first_2(mgard_cuda_handle<T> & handle, 
              int nrow,        int ncol,         int nfib,
              int nr,          int nc,           int nf, 
              int * dirow,     int * dicol_p,      int * difib,
              T * ddist_r, T * ddist_c, T * ddist_f,
              T * dv,      int lddv1,        int lddv2, 
              int queue_idx) {
    
  int B_adjusted = min(8, handle.B);  
  int total_thread_z = nr;
  int total_thread_y = ncol-nc;
  int total_thread_x = nf;
  if (total_thread_z == 0 || total_thread_y == 0 || total_thread_x == 0) return; 
  int tbz = min(B_adjusted, total_thread_z);
  int tby = min(B_adjusted, total_thread_y);
  int tbx = min(B_adjusted, total_thread_x);
  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  _pi_Ql_first_2<<<blockPerGrid, threadsPerBlock, 
                   0, *(cudaStream_t *)handle.get(queue_idx)>>>(nrow,       ncol,       nfib,
                                nr,         nc,         nf,
                                dirow,      dicol_p,     difib,
                                ddist_r,    ddist_c,    ddist_f,
                                dv,         lddv1,      lddv2);
  gpuErrchk(cudaGetLastError ());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
pi_Ql_first_2<double>(mgard_cuda_handle<double> & handle, 
              int nrow,        int ncol,         int nfib,
              int nr,          int nc,           int nf, 
              int * dirow,     int * dicol_p,      int * difib,
              double * ddist_r, double * ddist_c, double * ddist_f,
              double * dv,      int lddv1,        int lddv2, 
              int queue_idx);
template void 
pi_Ql_first_2<float>(mgard_cuda_handle<float> & handle, 
              int nrow,        int ncol,         int nfib,
              int nr,          int nc,           int nf, 
              int * dirow,     int * dicol_p,      int * difib,
              float * ddist_r, float * ddist_c, float * ddist_f,
              float * dv,      int lddv1,        int lddv2, 
              int queue_idx);

template <typename T>
__global__ void 
_pi_Ql_first_3(int nrow,        int ncol,         int nfib,
               int nr,          int nc,           int nf, 
               int * irow_p,     int * icol,      int * ifib,
               T * dist_r,      T * dist_c,       T * dist_f,
               T * v,           int ldv1,         int ldv2) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;
  for (int z = z0; z < nrow-nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nf; x += blockDim.x * gridDim.x) {
        int f = ifib[x];
        int c = icol[y];
        int r = irow_p[z];
        register T up = v[get_idx(ldv1, ldv2, r-1, c, f)];
        register T down = v[get_idx(ldv1, ldv2, r+1, c, f)];
        register T center = v[get_idx(ldv1, ldv2, r, c, f)];
        register T h1 = dist_r[r-1];
        register T h2 = dist_r[r];
        center -= (h2 * up + h1 * down) / (h1 + h2);
        v[get_idx(ldv1, ldv2, r, c, f)] = center;
      }
    }
  }
}

template <typename T>
void 
pi_Ql_first_3(mgard_cuda_handle<T> & handle, 
              int nrow,        int ncol,         int nfib,
              int nr,          int nc,           int nf, 
              int * dirow_p,    int * dicol,      int * difib,
              T * ddist_r, T * ddist_c, T * ddist_f,
              T * dv,      int lddv1,        int lddv2, 
              int queue_idx) {
    
  int B_adjusted = min(8, handle.B);  
  int total_thread_z = nrow-nr;
  int total_thread_y = nc;
  int total_thread_x = nf;
  if (total_thread_z == 0 || total_thread_y == 0 || total_thread_x == 0) return; 
  int tbz = min(B_adjusted, total_thread_z);
  int tby = min(B_adjusted, total_thread_y);
  int tbx = min(B_adjusted, total_thread_x);
  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  _pi_Ql_first_3<<<blockPerGrid, threadsPerBlock, 
                   0, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                          nrow,       ncol,       nfib,
                                          nr,         nc,         nf,
                                          dirow_p,    dicol,      difib,
                                          ddist_r,    ddist_c,    ddist_f,
                                          dv,         lddv1,      lddv2);
  gpuErrchk(cudaGetLastError ());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
pi_Ql_first_3<double>(mgard_cuda_handle<double> & handle, 
              int nrow,        int ncol,         int nfib,
              int nr,          int nc,           int nf, 
              int * dirow_p,    int * dicol,      int * difib,
              double * ddist_r, double * ddist_c, double * ddist_f,
              double * dv,      int lddv1,        int lddv2, 
              int queue_idx);
template void 
pi_Ql_first_3<float>(mgard_cuda_handle<float> & handle, 
              int nrow,        int ncol,         int nfib,
              int nr,          int nc,           int nf, 
              int * dirow_p,    int * dicol,      int * difib,
              float * ddist_r, float * ddist_c, float * ddist_f,
              float * dv,      int lddv1,        int lddv2, 
              int queue_idx);

template <typename T>
__global__ void 
_pi_Ql_first_12(int nrow,        int ncol,         int nfib,
                int nr,          int nc,           int nf, 
                int * irow,      int * icol_p,     int * ifib_p,
                T * dist_r,      T * dist_c,       T * dist_f,
                T * v,           int ldv1,         int ldv2) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;
  for (int z = z0; z < nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < ncol-nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nfib-nf; x += blockDim.x * gridDim.x) {
        int f = ifib_p[x];
        int c = icol_p[y];
        int r = irow[z];
        register T leftfront = v[get_idx(ldv1, ldv2, r, c-1, f-1)];
        register T rightfront = v[get_idx(ldv1, ldv2, r, c-1, f+1)];
        register T leftback = v[get_idx(ldv1, ldv2, r, c+1, f-1)];
        register T rightback = v[get_idx(ldv1, ldv2, r, c+1, f+1)];
        register T center = v[get_idx(ldv1, ldv2, r, c, f)];
        register T h1_f = dist_f[f-1];
        register T h2_f = dist_f[f];
        register T h1_c = dist_c[c-1];
        register T h2_c = dist_c[c];
        center -= (leftfront * h2_f * h2_c + 
                   rightfront * h1_f * h2_c + 
                   leftback * h2_f * h1_c + 
                   rightback * h1_f * h1_c)
                   /((h1_f+h2_f)*(h1_c+h2_c));
        v[get_idx(ldv1, ldv2, r, c, f)] = center;
      }
    }
  }
}


template <typename T>
void 
pi_Ql_first_12(mgard_cuda_handle<T> & handle, 
               int nrow,        int ncol,         int nfib,
               int nr,          int nc,           int nf, 
               int * dirow,     int * dicol_p,    int * difib_p,
               T * ddist_r,     T * ddist_c,      T * ddist_f,
               T * dv,          int lddv1,        int lddv2, 
               int queue_idx) {
    
  int B_adjusted = min(8, handle.B);  
  int total_thread_z = nr;
  int total_thread_y = ncol - nc;
  int total_thread_x = nfib - nf;
  if (total_thread_z == 0 || total_thread_y == 0 || total_thread_x == 0) return; 
  int tbz = min(B_adjusted, total_thread_z);
  int tby = min(B_adjusted, total_thread_y);
  int tbx = min(B_adjusted, total_thread_x);
  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  _pi_Ql_first_12<<<blockPerGrid, threadsPerBlock, 
                    0, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                          nrow,       ncol,       nfib,
                                          nr,         nc,         nf,
                                          dirow,      dicol_p,    difib_p,
                                          ddist_r,    ddist_c,    ddist_f,
                                          dv,         lddv1,      lddv2);
  gpuErrchk(cudaGetLastError ());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
pi_Ql_first_12<double>(mgard_cuda_handle<double> & handle, 
               int nrow,        int ncol,         int nfib,
               int nr,          int nc,           int nf, 
               int * dirow,     int * dicol_p,    int * difib_p,
               double * ddist_r,     double * ddist_c,      double * ddist_f,
               double * dv,          int lddv1,        int lddv2, 
               int queue_idx);

template void 
pi_Ql_first_12<float>(mgard_cuda_handle<float> & handle, 
               int nrow,        int ncol,         int nfib,
               int nr,          int nc,           int nf, 
               int * dirow,     int * dicol_p,    int * difib_p,
               float * ddist_r,     float * ddist_c,      float * ddist_f,
               float * dv,          int lddv1,        int lddv2, 
               int queue_idx);


template <typename T>
__global__ void 
_pi_Ql_first_13(int nrow,        int ncol,         int nfib,
                int nr,          int nc,           int nf, 
                int * irow_p,    int * icol,       int * ifib_p,
                T * dist_r,      T * dist_c,       T * dist_f,
                T * v,           int ldv1,         int ldv2) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;
  for (int z = z0; z < nrow-nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nfib-nf; x += blockDim.x * gridDim.x) {
        int f = ifib_p[x];
        int c = icol[y];
        int r = irow_p[z];
        register T leftup = v[get_idx(ldv1, ldv2, r-1, c, f-1)];
        register T rightup = v[get_idx(ldv1, ldv2, r-1, c, f+1)];
        register T leftdown = v[get_idx(ldv1, ldv2, r+1, c, f-1)];
        register T rightdown = v[get_idx(ldv1, ldv2, r+1, c, f+1)];
        register T center = v[get_idx(ldv1, ldv2, r, c, f)];
        register T h1_f = dist_f[f-1];
        register T h2_f = dist_f[f];
        register T h1_r = dist_r[r-1];
        register T h2_r = dist_r[r];
        center -= (leftup * h2_f * h2_r + 
                   rightup * h1_f * h2_r + 
                   leftdown * h2_f * h1_r + 
                   rightdown * h1_f * h1_r)
                   /((h1_f+h2_f)*(h1_r+h2_r));
        v[get_idx(ldv1, ldv2, r, c, f)] = center;
      }
    }
  }
}


template <typename T>
void 
pi_Ql_first_13(mgard_cuda_handle<T> & handle, 
               int nrow,        int ncol,         int nfib,
               int nr,          int nc,           int nf, 
               int * dirow_p,   int * dicol,      int * difib_p,
               T * ddist_r, T * ddist_c, T * ddist_f,
               T * dv,      int lddv1,        int lddv2, 
               int queue_idx) {
    
  int B_adjusted = min(8, handle.B);  
  int total_thread_z = nrow - nr;
  int total_thread_y = nc;
  int total_thread_x = nfib - nf;
  if (total_thread_z == 0 || total_thread_y == 0 || total_thread_x == 0) return; 
  int tbz = min(B_adjusted, total_thread_z);
  int tby = min(B_adjusted, total_thread_y);
  int tbx = min(B_adjusted, total_thread_x);
  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  _pi_Ql_first_13<<<blockPerGrid, threadsPerBlock, 
                    0, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                          nrow,       ncol,       nfib,
                                          nr,         nc,         nf,
                                          dirow_p,    dicol,      difib_p,
                                          ddist_r,    ddist_c,    ddist_f,
                                          dv,         lddv1,      lddv2);
  gpuErrchk(cudaGetLastError ());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
pi_Ql_first_13<double>(mgard_cuda_handle<double> & handle, 
               int nrow,        int ncol,         int nfib,
               int nr,          int nc,           int nf, 
               int * dirow_p,   int * dicol,      int * difib_p,
               double * ddist_r, double * ddist_c, double * ddist_f,
               double * dv,      int lddv1,        int lddv2, 
               int queue_idx);
template void 
pi_Ql_first_13<float>(mgard_cuda_handle<float> & handle, 
               int nrow,        int ncol,         int nfib,
               int nr,          int nc,           int nf, 
               int * dirow_p,   int * dicol,      int * difib_p,
               float * ddist_r, float * ddist_c, float * ddist_f,
               float * dv,      int lddv1,        int lddv2, 
               int queue_idx);

template <typename T>
__global__ void 
_pi_Ql_first_23(int nrow,        int ncol,         int nfib,
                int nr,          int nc,           int nf, 
                int * irow_p,    int * icol_p,     int * ifib,
                T * dist_r,      T * dist_c,       T * dist_f,
                T * v,           int ldv1,         int ldv2) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;
  for (int z = z0; z < nrow-nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < ncol-nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nf; x += blockDim.x * gridDim.x) {
        int f = ifib[x];
        int c = icol_p[y];
        int r = irow_p[z];
        register T frontup = v[get_idx(ldv1, ldv2, r-1, c-1, f)];
        register T frontdown = v[get_idx(ldv1, ldv2, r+1, c-1, f)];
        register T backup = v[get_idx(ldv1, ldv2, r-1, c+1, f)];
        register T backdown = v[get_idx(ldv1, ldv2, r+1, c+1, f)];
        register T center = v[get_idx(ldv1, ldv2, r, c, f)];
        register T h1_c = dist_c[c-1];
        register T h2_c = dist_c[c];
        register T h1_r = dist_r[r-1];
        register T h2_r = dist_r[r];
        center -= (frontup * h2_c * h2_r + 
                   frontdown * h1_c * h2_r + 
                   backup * h2_c * h1_r + 
                   backdown * h1_c * h1_r)
                   /((h1_c+h2_c)*(h1_r+h2_r));
        v[get_idx(ldv1, ldv2, r, c, f)] = center;
      }
    }
  }
}


template <typename T>
void 
pi_Ql_first_23(mgard_cuda_handle<T> & handle, 
               int nrow,        int ncol,         int nfib,
               int nr,          int nc,           int nf, 
               int * dirow_p,   int * dicol_p,    int * difib,
               T * ddist_r, T * ddist_c, T * ddist_f,
               T * dv,      int lddv1,        int lddv2, 
               int queue_idx) {
    
  int B_adjusted = min(8, handle.B);  
  int total_thread_z = nrow - nr;
  int total_thread_y = ncol - nc;
  int total_thread_x = nf;
  if (total_thread_z == 0 || total_thread_y == 0 || total_thread_x == 0) return; 
  int tbz = min(B_adjusted, total_thread_z);
  int tby = min(B_adjusted, total_thread_y);
  int tbx = min(B_adjusted, total_thread_x);
  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  _pi_Ql_first_23<<<blockPerGrid, threadsPerBlock, 
                    0, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                          nrow,       ncol,       nfib,
                                          nr,         nc,         nf,
                                          dirow_p,    dicol_p,    difib,
                                          ddist_r,    ddist_c,    ddist_f,
                                          dv,         lddv1,      lddv2);
  gpuErrchk(cudaGetLastError ());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
pi_Ql_first_23<double>(mgard_cuda_handle<double> & handle, 
               int nrow,        int ncol,         int nfib,
               int nr,          int nc,           int nf, 
               int * dirow_p,   int * dicol_p,    int * difib,
               double * ddist_r, double * ddist_c, double * ddist_f,
               double * dv,      int lddv1,        int lddv2, 
               int queue_idx);
template void 
pi_Ql_first_23<float>(mgard_cuda_handle<float> & handle, 
               int nrow,        int ncol,         int nfib,
               int nr,          int nc,           int nf, 
               int * dirow_p,   int * dicol_p,    int * difib,
               float * ddist_r, float * ddist_c, float * ddist_f,
               float * dv,      int lddv1,        int lddv2, 
               int queue_idx);

template <typename T>
__global__ void 
_pi_Ql_first_123(int nrow,        int ncol,         int nfib,
                 int nr,          int nc,           int nf, 
                 int * irow_p,    int * icol_p,     int * ifib_p,
                 T * dist_r,      T * dist_c,       T * dist_f,
                 T * v,           int ldv1,         int ldv2) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;
  for (int z = z0; z < nrow-nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < ncol-nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nfib-nf; x += blockDim.x * gridDim.x) {
        int f = ifib_p[x];
        int c = icol_p[y];
        int r = irow_p[z];
        register T rightfrontup = v[get_idx(ldv1, ldv2, r-1, c-1, f-1)];
        register T rightfrontdown = v[get_idx(ldv1, ldv2, r+1, c-1, f-1)];
        register T rightbackup = v[get_idx(ldv1, ldv2, r-1, c+1, f-1)];
        register T rightbackdown = v[get_idx(ldv1, ldv2, r+1, c+1, f-1)];
        register T leftfrontup = v[get_idx(ldv1, ldv2, r-1, c-1, f+1)];
        register T leftfrontdown = v[get_idx(ldv1, ldv2, r+1, c-1, f+1)];
        register T leftbackup = v[get_idx(ldv1, ldv2, r-1, c+1, f+1)];
        register T leftbackdown = v[get_idx(ldv1, ldv2, r+1, c+1, f+1)];
        register T center = v[get_idx(ldv1, ldv2, r, c, f)];
        register T h1_f = dist_f[f-1];
        register T h2_f = dist_f[f];
        register T h1_c = dist_c[c-1];
        register T h2_c = dist_c[c];
        register T h1_r = dist_r[r-1];
        register T h2_r = dist_r[r];
        T x00 = (rightfrontup * h2_f + leftfrontup * h1_f) / (h2_f + h1_f);
        T x01 = (rightbackup * h2_f +  leftbackup * h1_f) / (h2_f + h1_f);
        T x10 = (rightfrontdown * h2_f + leftfrontdown * h1_f) / (h2_f + h1_f);
        T x11 = (rightbackdown * h2_f + leftbackdown * h1_f) / (h2_f + h1_f);
        T y0  = (h2_c * x00 + h1_c * x01) / (h2_c + h1_c);
        T y1  = (h2_c * x10 + h1_c * x11) / (h2_c + h1_c);
        T z   = (h2_r * y0 + h1_r * y1) / (h2_r + h1_r);
        center -= z;
        v[get_idx(ldv1, ldv2, r, c, f)] = center;
      }
    }
  }
}

template <typename T>
void 
pi_Ql_first_123(mgard_cuda_handle<T> & handle, 
                int nrow,        int ncol,         int nfib,
                int nr,          int nc,           int nf, 
                int * dirow_p,    int * dicol_p,     int * difib_p,
                T * ddist_r, T * ddist_c, T * ddist_f,
                T * dv,      int lddv1,        int lddv2, 
                int queue_idx) {
    
  int B_adjusted = min(8, handle.B);  
  int total_thread_z = nrow - nr;
  int total_thread_y = ncol - nc;
  int total_thread_x = nfib - nf;
  if (total_thread_z == 0 || total_thread_y == 0 || total_thread_x == 0) return; 
  int tbz = min(B_adjusted, total_thread_z);
  int tby = min(B_adjusted, total_thread_y);
  int tbx = min(B_adjusted, total_thread_x);
  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  _pi_Ql_first_123<<<blockPerGrid, threadsPerBlock, 
                     0, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                          nrow,       ncol,       nfib,
                                          nr,         nc,         nf,
                                          dirow_p,    dicol_p,    difib_p,
                                          ddist_r,    ddist_c,    ddist_f,
                                          dv,         lddv1,      lddv2);


  gpuErrchk(cudaGetLastError ());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
pi_Ql_first_123<double>(mgard_cuda_handle<double> & handle, 
                int nrow,        int ncol,         int nfib,
                int nr,          int nc,           int nf, 
                int * dirow_p,    int * dicol_p,     int * difib_p,
                double * ddist_r, double * ddist_c, double * ddist_f,
                double * dv,      int lddv1,        int lddv2, 
                int queue_idx);
template void 
pi_Ql_first_123<float>(mgard_cuda_handle<float> & handle, 
                int nrow,        int ncol,         int nfib,
                int nr,          int nc,           int nf, 
                int * dirow_p,    int * dicol_p,     int * difib_p,
                float * ddist_r, float * ddist_c, float * ddist_f,
                float * dv,      int lddv1,        int lddv2, 
                int queue_idx);
}