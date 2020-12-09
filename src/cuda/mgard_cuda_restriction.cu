#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda_restriction.h"

namespace mgard_cuda {

template <typename T>
__global__ void _restriction_1(int nrow, int ncol, int nr, int nc,
                               int row_stride, int col_stride, int *dirow,
                               int *dicol, T *dcoords_x, T *dv, int lddv) {
  int col_Cstride = col_stride * 2;
  int r0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  for (int r = r0; r < nr; r += (blockDim.x * gridDim.x) * row_stride) {
    T *vec = dv + dirow[r] * lddv;
    T h1 = _get_dist(dcoords_x, dicol[0],
                     dicol[col_stride]); // dicol[col_Pstride] - dicol[0];
    T h2 = _get_dist(
        dcoords_x, dicol[col_stride],
        dicol[col_Cstride]); // dicol[col_stride] - dicol[col_Pstride];
    T hsum = h1 + h2;
    vec[dicol[0]] += h2 * vec[dicol[col_stride]] / hsum;
    for (int i = col_Cstride; i <= nc - col_Cstride; i += col_Cstride) {
      vec[dicol[i]] += h1 * vec[dicol[i - col_stride]] / hsum;
      h1 = _get_dist(dcoords_x, dicol[i], dicol[i + col_stride]);
      h2 = _get_dist(dcoords_x, dicol[i + col_stride], dicol[i + col_Cstride]);
      hsum = h1 + h2;
      vec[dicol[i]] += h2 * vec[dicol[i + col_stride]] / hsum;
    }
    vec[dicol[nc - 1]] += h1 * vec[dicol[nc - col_stride - 1]] / hsum;
  }
}

template <typename T>
void restriction_1(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nr,
                   int nc, int row_stride, int col_stride, int *dirow,
                   int *dicol, T *dcoords_x, T *dv, int lddv, int queue_idx) {

  int total_thread_x = ceil((float)nr / (row_stride));
  int tbx = min(handle.B, total_thread_x);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);
  _restriction_1<<<blockPerGrid, threadsPerBlock, 0,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nr, nc, row_stride, col_stride, dirow, dicol, dcoords_x, dv,
      lddv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void restriction_1<double>(mgard_cuda_handle<double> &handle, int nrow,
                                    int ncol, int nr, int nc, int row_stride,
                                    int col_stride, int *dirow, int *dicol,
                                    double *dcoords_x, double *dv, int lddv,
                                    int queue_idx);
template void restriction_1<float>(mgard_cuda_handle<float> &handle, int nrow,
                                   int ncol, int nr, int nc, int row_stride,
                                   int col_stride, int *dirow, int *dicol,
                                   float *dcoords_x, float *dv, int lddv,
                                   int queue_idx);

template <typename T>
__global__ void _restriction_2(int nrow, int ncol, int nr, int nc,
                               int row_stride, int col_stride, int *dirow,
                               int *dicol, T *dcoords_y, T *dv, int lddv) {
  int row_Cstride = row_stride * 2;
  int c0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  for (int c = c0; c < nc; c += (blockDim.x * gridDim.x) * col_stride) {
    T *vec = dv + dicol[c];
    T h1 = _get_dist(dcoords_y, dirow[0],
                     dirow[row_stride]); // dirow[row_Pstride] - dirow[0];
    T h2 = _get_dist(
        dcoords_y, dirow[row_stride],
        dirow[row_Cstride]); // dirow[row_stride] - dirow[row_Pstride];
    T hsum = h1 + h2;
    vec[dirow[0] * lddv] += h2 * vec[dirow[row_stride] * lddv] / hsum;
    for (int i = row_Cstride; i <= nr - row_Cstride; i += row_Cstride) {
      vec[dirow[i] * lddv] += h1 * vec[dirow[i - row_stride] * lddv] / hsum;
      h1 = _get_dist(dcoords_y, dirow[i], dirow[i + row_stride]);
      h2 = _get_dist(dcoords_y, dirow[i + row_stride], dirow[i + row_Cstride]);
      hsum = h1 + h2;
      vec[dirow[i] * lddv] += h2 * vec[dirow[i + row_stride] * lddv] / hsum;
    }
    vec[dirow[nr - 1] * lddv] +=
        h1 * vec[dirow[nr - row_stride - 1] * lddv] / hsum;
  }
}

template <typename T>
void restriction_2(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nr,
                   int nc, int row_stride, int col_stride, int *dirow,
                   int *dicol, T *dcoords_y, T *dv, int lddv, int queue_idx) {

  int total_thread_x = ceil((float)nc / (col_stride));
  int tbx = min(handle.B, total_thread_x);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);
  _restriction_2<<<blockPerGrid, threadsPerBlock, 0,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nr, nc, row_stride, col_stride, dirow, dicol, dcoords_y, dv,
      lddv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void restriction_2<double>(mgard_cuda_handle<double> &handle, int nrow,
                                    int ncol, int nr, int nc, int row_stride,
                                    int col_stride, int *dirow, int *dicol,
                                    double *dcoords_y, double *dv, int lddv,
                                    int queue_idx);
template void restriction_2<float>(mgard_cuda_handle<float> &handle, int nrow,
                                   int ncol, int nr, int nc, int row_stride,
                                   int col_stride, int *dirow, int *dicol,
                                   float *dcoords_y, float *dv, int lddv,
                                   int queue_idx);

// assume number of main col and ghost col are even numbers
template <typename T>
__global__ void _restriction_1_cpt(int nr, int nc, int row_stride,
                                   int col_stride, T *ddist_x, int ghost_col,
                                   T *__restrict__ dv, int lddv) {

  // int ghost_col = 2;
  register int total_row = ceil((double)nr / (row_stride));
  register int total_col = ceil((double)nc / (col_stride));

  // index on dirow and dicol
  register int r0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  register int c0 = threadIdx.x;
  register int c0_stride = threadIdx.x * col_stride;

  // index on sm
  register int r0_sm = threadIdx.y;
  register int c0_sm = threadIdx.x;

  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);
  T *sm = SharedMemory<T>();
  // extern __shared__ double sm[]; // row = blockDim.y; col = blockDim.x +
  // ghost_col;
  register int ldsm = blockDim.x + ghost_col;

  T *vec_sm = sm + r0_sm * ldsm;
  T *dist_x_sm = sm + (blockDim.y) * ldsm;

  register T result = 1;
  register T h1 = 1;
  register T h2 = 1;
  register T h3;
  register T h4;

  register T result1;
  register T result2;
  register int rest_col;
  register int real_ghost_col;
  register int real_main_col;

  register T prev_vec_sm;
  register T prev_h1;
  register T prev_h2;

  for (int r = r0; r < nr; r += gridDim.y * blockDim.y * row_stride) {

    T *vec = dv + r * lddv;

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
      // load main column
      real_main_col = min(blockDim.x, rest_col);
      if (c0_sm < real_main_col) {
        vec_sm[c0_sm + real_ghost_col] =
            vec[c0_stride + real_ghost_col * col_stride];
        if (r0_sm == 0) {
          dist_x_sm[c0_sm + real_ghost_col] = ddist_x[c0 + real_ghost_col];
        }
      }
      __syncthreads();

      if (c0_sm % 2 == 0) {
        // computation
        result = vec_sm[c0_sm];
        if (c0_sm == 0) {
          // result = vec_sm[c0_sm];
          h1 = prev_h1;
          h2 = prev_h2;
          h3 = dist_x_sm[c0_sm];
          h4 = dist_x_sm[c0_sm + 1];
          result1 = h1 * prev_vec_sm / (h1 + h2);
          // h1 = dist_x_sm[c0_sm];
          // h2 = dist_x_sm[c0_sm+1];
          result2 = h4 * vec_sm[c0_sm + 1] / (h3 + h4);
          // result = result + result1 + result2;
        } else {
          // result = vec_sm[c0_sm];
          h1 = dist_x_sm[c0_sm - 2];
          h2 = dist_x_sm[c0_sm - 1];
          h3 = dist_x_sm[c0_sm];
          h4 = dist_x_sm[c0_sm + 1];
          result1 = h1 * vec_sm[c0_sm - 1] / (h1 + h2);
          // h1 = dist_x_sm[c0_sm];
          // h2 = dist_x_sm[c0_sm+1];
          result2 = h4 * vec_sm[c0_sm + 1] / (h3 + h4);
        }
        result = result + result1 + result2;
        vec[c0_stride] = result;
      }
      __syncthreads();

      // store last column
      if (c0_sm == 0) {
        prev_vec_sm = vec_sm[blockDim.x - 1];
        prev_h1 = dist_x_sm[blockDim.x - 2]; //_get_dist(dcoords_x_sm,
                                             // blockDim.x - 2, blockDim.x - 1);
        prev_h2 =
            dist_x_sm[blockDim.x -
                      1]; //_get_dist(dcoords_x_sm, blockDim.x - 1, blockDim.x);
      }

      // advance c0
      c0 += blockDim.x;
      c0_stride += blockDim.x * col_stride;
      __syncthreads();

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
    } // end while

    if (c0_sm < rest_col) {
      vec_sm[c0_sm + real_ghost_col] =
          vec[c0_stride + real_ghost_col * col_stride];
      // dcoords_x_sm[c0_sm + real_ghost_col] = dcoords_x[c0_stride +
      // real_ghost_col * col_stride];
      dist_x_sm[c0_sm + real_ghost_col] = ddist_x[c0 + real_ghost_col];
    }
    __syncthreads();

    if (real_ghost_col + rest_col == 1) {
      if (c0_sm == 0) {
        result = vec_sm[c0_sm];
        h1 = prev_h1;
        h2 = prev_h2;
        result += h1 * prev_vec_sm / (h1 + h2);
        vec[c0_stride] = result;
      }
    } else {
      if (c0_sm < real_ghost_col + rest_col) {
        if (c0_sm % 2 == 0) {
          if (c0_sm == 0) {
            result = vec_sm[c0_sm];
            h1 = prev_h1;
            h2 = prev_h2;
            result += h1 * prev_vec_sm / (h1 + h2);
            h1 = dist_x_sm[c0_sm]; //_get_dist(dcoords_x_sm, c0_sm, c0_sm+1);
            h2 = dist_x_sm[c0_sm +
                           1]; //_get_dist(dcoords_x_sm, c0_sm+1, c0_sm+2);
            result += h2 * vec_sm[c0_sm + 1] / (h1 + h2);
          } else if (c0_sm == real_ghost_col + rest_col - 1) {
            result = vec_sm[c0_sm];
            h1 = dist_x_sm[c0_sm -
                           2]; //_get_dist(dcoords_x_sm, c0_sm-2, c0_sm-1);;
            h2 = dist_x_sm[c0_sm -
                           1]; //_get_dist(dcoords_x_sm, c0_sm-1, c0_sm);;
            result += h1 * vec_sm[c0_sm - 1] / (h1 + h2);
          } else {
            result = vec_sm[c0_sm];
            h1 = dist_x_sm[c0_sm -
                           2]; //_get_dist(dcoords_x_sm, c0_sm-2, c0_sm-1);
            h2 =
                dist_x_sm[c0_sm - 1]; //_get_dist(dcoords_x_sm, c0_sm-1, c0_sm);
            result += h1 * vec_sm[c0_sm - 1] / (h1 + h2);
            h1 = dist_x_sm[c0_sm]; //_get_dist(dcoords_x_sm, c0_sm, c0_sm+1);
            h2 = dist_x_sm[c0_sm +
                           1]; //_get_dist(dcoords_x_sm, c0_sm+1, c0_sm+2);
            result += h2 * vec_sm[c0_sm + 1] / (h1 + h2);
          }
          vec[c0_stride] = result;
        }
        __syncthreads();
      }
    }
  }
}

template <typename T>
void restriction_1_cpt(mgard_cuda_handle<T> &handle, int nr, int nc,
                       int row_stride, int col_stride, T *ddist_x, T *dv,
                       int lddv, int queue_idx) {

  int ghost_col = handle.B;
  int total_row = ceil((double)nr / (row_stride));
  int total_col = ceil((double)nc / (col_stride));
  int total_thread_y = ceil((double)nr / (row_stride));
  int total_thread_x = min(handle.B, total_col);
  int tby = min(handle.B, total_thread_y);
  int tbx = min(handle.B, total_thread_x);
  size_t sm_size = ((tbx + ghost_col) * (tby + 1)) * sizeof(T);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = 1;
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _restriction_1_cpt<<<blockPerGrid, threadsPerBlock, sm_size,
                       *(cudaStream_t *)handle.get(queue_idx)>>>(
      nr, nc, row_stride, col_stride, ddist_x, ghost_col, dv, lddv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void restriction_1_cpt<double>(mgard_cuda_handle<double> &handle,
                                        int nr, int nc, int row_stride,
                                        int col_stride, double *ddist_x,
                                        double *dv, int lddv, int queue_idx);
template void restriction_1_cpt<float>(mgard_cuda_handle<float> &handle, int nr,
                                       int nc, int row_stride, int col_stride,
                                       float *ddist_x, float *dv, int lddv,
                                       int queue_idx);

// assume number of main col and ghost col are even numbers
template <typename T>
__global__ void _restriction_2_cpt(int nr, int nc, int row_stride,
                                   int col_stride, T *ddist_y, int ghost_row,
                                   T *__restrict__ dv, int lddv) {

  // int ghost_col = 2;
  register int total_row = ceil((double)nr / (row_stride));
  register int total_col = ceil((double)nc / (col_stride));

  // index on dirow and dicol
  register int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;
  register int r0 = threadIdx.y;
  register int r0_stride = threadIdx.y * row_stride;

  // index on sm
  register int r0_sm = threadIdx.y;
  register int c0_sm = threadIdx.x;

  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);
  T *sm = SharedMemory<T>();
  // extern __shared__ double sm[]; // row = blockDim.y; col = blockDim.x +
  // ghost_col;
  register int ldsm = blockDim.x;

  T *vec_sm = sm + c0_sm;
  T *dist_y_sm = sm + (blockDim.y + ghost_row) * ldsm;

  register T result = 1;
  register T h1 = 1;
  register T h2 = 1;
  register T h3;
  register T h4;

  register T result1;
  register T result2;
  register int rest_row;
  register int real_ghost_row;
  register int real_main_row;

  register T prev_vec_sm;
  register T prev_h1;
  register T prev_h2;

  for (int c = c0; c < nc; c += gridDim.x * blockDim.x * col_stride) {

    T *vec = dv + c;

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
        vec_sm[(r0_sm + real_ghost_row) * ldsm] =
            vec[(r0_stride + real_ghost_row * row_stride) * lddv];
      }
      if (c0_sm == 0 && r0_sm < real_main_row) {
        dist_y_sm[r0_sm + real_ghost_row] = ddist_y[r0 + real_ghost_row];
      }
      __syncthreads();

      if (r0_sm % 2 == 0) {
        // computation
        result = vec_sm[r0_sm * ldsm];
        if (r0_sm == 0) {
          // result = vec_sm[c0_sm];
          h1 = prev_h1;
          h2 = prev_h2;
          h3 = dist_y_sm[r0_sm];
          h4 = dist_y_sm[r0_sm + 1];
          result1 = h1 * prev_vec_sm / (h1 + h2);
          // h1 = dist_x_sm[c0_sm];
          // h2 = dist_x_sm[c0_sm+1];
          result2 = h4 * vec_sm[(r0_sm + 1) * ldsm] / (h3 + h4);
          // result = result + result1 + result2;
        } else {
          // result = vec_sm[c0_sm];
          h1 = dist_y_sm[r0_sm - 2];
          h2 = dist_y_sm[r0_sm - 1];
          h3 = dist_y_sm[r0_sm];
          h4 = dist_y_sm[r0_sm + 1];
          result1 = h1 * vec_sm[(r0_sm - 1) * ldsm] / (h1 + h2);
          // h1 = dist_x_sm[c0_sm];
          // h2 = dist_x_sm[c0_sm+1];
          result2 = h4 * vec_sm[(r0_sm + 1) * ldsm] / (h3 + h4);
        }
        result = result + result1 + result2;
        vec[r0_stride * lddv] = result;
      }
      __syncthreads();

      // store last column
      if (r0_sm == 0) {
        prev_vec_sm = vec_sm[(blockDim.y - 1) * ldsm];
        prev_h1 = dist_y_sm[blockDim.y - 2]; //_get_dist(dcoords_x_sm,
                                             // blockDim.x - 2, blockDim.x - 1);
        prev_h2 =
            dist_y_sm[blockDim.y -
                      1]; //_get_dist(dcoords_x_sm, blockDim.x - 1, blockDim.x);
      }
      __syncthreads();
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
    } // end while

    if (r0_sm < rest_row) {
      vec_sm[(r0_sm + real_ghost_row) * ldsm] =
          vec[(r0_stride + real_ghost_row * row_stride) * lddv];
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
        result += h1 * prev_vec_sm / (h1 + h2);
        vec[r0_stride * lddv] = result;
      }
    } else {
      if (r0_sm < real_ghost_row + rest_row) {
        if (r0_sm % 2 == 0) {
          if (r0_sm == 0) {
            result = vec_sm[r0_sm * ldsm];
            h1 = prev_h1;
            h2 = prev_h2;
            result += h1 * prev_vec_sm / (h1 + h2);
            h1 = dist_y_sm[r0_sm]; //_get_dist(dcoords_x_sm, c0_sm, c0_sm+1);
            h2 = dist_y_sm[r0_sm +
                           1]; //_get_dist(dcoords_x_sm, c0_sm+1, c0_sm+2);
            result += h2 * vec_sm[(r0_sm + 1) * ldsm] / (h1 + h2);
          } else if (r0_sm == real_ghost_row + rest_row - 1) {
            result = vec_sm[r0_sm * ldsm];
            h1 = dist_y_sm[r0_sm -
                           2]; //_get_dist(dcoords_x_sm, c0_sm-2, c0_sm-1);;
            h2 = dist_y_sm[r0_sm -
                           1]; //_get_dist(dcoords_x_sm, c0_sm-1, c0_sm);;
            result += h1 * vec_sm[(r0_sm - 1) * ldsm] / (h1 + h2);
          } else {
            result = vec_sm[r0_sm * ldsm];
            h1 = dist_y_sm[r0_sm -
                           2]; //_get_dist(dcoords_x_sm, c0_sm-2, c0_sm-1);
            h2 =
                dist_y_sm[r0_sm - 1]; //_get_dist(dcoords_x_sm, c0_sm-1, c0_sm);
            result += h1 * vec_sm[(r0_sm - 1) * ldsm] / (h1 + h2);
            h1 = dist_y_sm[r0_sm]; //_get_dist(dcoords_x_sm, c0_sm, c0_sm+1);
            h2 = dist_y_sm[r0_sm +
                           1]; //_get_dist(dcoords_x_sm, c0_sm+1, c0_sm+2);
            result += h2 * vec_sm[(r0_sm + 1) * ldsm] / (h1 + h2);
          }
          vec[r0_stride * lddv] = result;
        }
        __syncthreads();
      }
    }
  }
}

template <typename T>
void restriction_2_cpt(mgard_cuda_handle<T> &handle, int nr, int nc,
                       int row_stride, int col_stride, T *ddist_y, T *dv,
                       int lddv, int queue_idx) {

  int ghost_row = handle.B;
  int total_row = ceil((double)nr / (row_stride));
  int total_col = ceil((double)nc / (col_stride));
  int total_thread_y = min(handle.B, total_row);
  int total_thread_x = ceil((double)nc / (col_stride));
  int tby = min(handle.B, total_thread_y);
  int tbx = min(handle.B, total_thread_x);
  size_t sm_size = ((tby + ghost_row) * (tbx + 1)) * sizeof(double);
  int gridy = 1;
  int gridx =
      ceil((float)total_thread_x / tbx); // ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _restriction_2_cpt<<<blockPerGrid, threadsPerBlock, sm_size,
                       *(cudaStream_t *)handle.get(queue_idx)>>>(
      nr, nc, row_stride, col_stride, ddist_y, ghost_row, dv, lddv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void restriction_2_cpt<double>(mgard_cuda_handle<double> &handle,
                                        int nr, int nc, int row_stride,
                                        int col_stride, double *ddist_y,
                                        double *dv, int lddv, int queue_idx);
template void restriction_2_cpt<float>(mgard_cuda_handle<float> &handle, int nr,
                                       int nc, int row_stride, int col_stride,
                                       float *ddist_y, float *dv, int lddv,
                                       int queue_idx);

template <typename T>
__global__ void _restriction_first_1(int nrow, int ncol, int nr, int nc,
                                     int row_stride, int col_stride, int *irow,
                                     int *icol_p, T *dist_c, T *dv, int lddv) {
  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  for (int x = idx; x < nr; x += (blockDim.x * gridDim.x) * row_stride) {
    T *vec = dv + irow[x] * lddv;
    for (int i = 0; i < ncol - nc; i += col_stride) {
      T h1 = dist_c[icol_p[i] - 1];
      T h2 = dist_c[icol_p[i]];
      T hsum = h1 + h2;
      vec[icol_p[i] - 1] += h2 * vec[icol_p[i]] / hsum;
      vec[icol_p[i] + 1] += h1 * vec[icol_p[i]] / hsum;
    }
  }
}

template <typename T>
void restriction_first_1(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                         int nr, int nc, int row_stride, int col_stride,
                         int *dirow, int *dicol_p, T *ddist_c, T *dv, int lddv,
                         int queue_idx) {

  int total_thread = ceil((float)nr / row_stride);
  int tb = min(handle.B, total_thread);
  int grid = ceil((float)total_thread / tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);
  _restriction_first_1<<<blockPerGrid, threadsPerBlock, 0,
                         *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nr, nc, row_stride, col_stride, dirow, dicol_p, ddist_c, dv,
      lddv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void restriction_first_1<double>(mgard_cuda_handle<double> &handle,
                                          int nrow, int ncol, int nr, int nc,
                                          int row_stride, int col_stride,
                                          int *dirow, int *dicol_p,
                                          double *ddist_c, double *dv, int lddv,
                                          int queue_idx);
template void restriction_first_1<float>(mgard_cuda_handle<float> &handle,
                                         int nrow, int ncol, int nr, int nc,
                                         int row_stride, int col_stride,
                                         int *dirow, int *dicol_p,
                                         float *ddist_c, float *dv, int lddv,
                                         int queue_idx);

template <typename T>
__global__ void _restriction_first_2(int nrow, int ncol, int nr, int nc,
                                     int row_stride, int col_stride,
                                     int *irow_p, int *icol, T *dist_r, T *dv,
                                     int lddv) {
  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  for (int x = idx; x < nc; x += (blockDim.x * gridDim.x) * col_stride) {
    T *vec = dv + icol[x];
    for (int i = 0; i < nrow - nr; i += row_stride) {
      T h1 = dist_r[irow_p[i] - 1];
      T h2 = dist_r[irow_p[i]];
      T hsum = h1 + h2;
      vec[(irow_p[i] - 1) * lddv] += h2 * vec[irow_p[i] * lddv] / hsum;
      vec[(irow_p[i] + 1) * lddv] += h1 * vec[irow_p[i] * lddv] / hsum;
    }
  }
}

template <typename T>
void restriction_first_2(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                         int nr, int nc, int row_stride, int col_stride,
                         int *irow_p, int *icol, T *dist_r, T *dv, int lddv,
                         int queue_idx) {

  int total_thread = ceil((float)nc / col_stride);
  int tb = min(handle.B, total_thread);
  int grid = ceil((float)total_thread / tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);
  _restriction_first_2<<<blockPerGrid, threadsPerBlock, 0,
                         *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nr, nc, row_stride, col_stride, irow_p, icol, dist_r, dv,
      lddv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void restriction_first_2<double>(mgard_cuda_handle<double> &handle,
                                          int nrow, int ncol, int nr, int nc,
                                          int row_stride, int col_stride,
                                          int *irow_p, int *icol,
                                          double *dist_r, double *dv, int lddv,
                                          int queue_idx);
template void restriction_first_2<float>(mgard_cuda_handle<float> &handle,
                                         int nrow, int ncol, int nr, int nc,
                                         int row_stride, int col_stride,
                                         int *irow_p, int *icol, float *dist_r,
                                         float *dv, int lddv, int queue_idx);
// __global__ void
// _restriction_l_row_cuda_sm_pf(int nrow,       int ncol,
//                          int nr,         int nc,
//                          int row_stride, int col_stride,
//                          int * __restrict__ dirow,    int * __restrict__
//                          dicol, double * __restrict__ dv,    int lddv, double
//                          * __restrict__ dcoords_x, int ghost_col) {

//   //int ghost_col = 2;
//   register int total_row = ceil((double)nr/(row_stride));
//   register int total_col = ceil((double)nc/(col_stride));

//   // index on dirow and dicol
//   register int r0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
//   register int c0 = threadIdx.x * col_stride;

//   // index on sm
//   register int r0_sm = threadIdx.y;
//   register int c0_sm = threadIdx.x;

//   extern __shared__ double sm[]; // row = blockDim.y; col = blockDim.x +
//   ghost_col; register int ldsm = blockDim.x + ghost_col;
//   // printf("ldsm = %d\n", ldsm);

//   double * vec_sm = sm + r0_sm * ldsm;
//   double * dcoords_x_sm = sm + blockDim.y * ldsm;

//   register double result = 1;
//   register double h1 = 1;
//   register double h2 = 1;

//   register int main_col = blockDim.x;
//   register int rest_load_col;
//   register int rest_comp_col;
//   register int curr_ghost_col;
//   register int curr_main_col;
//   register int next_ghost_col;
//   register int next_main_col;

//   register double prev_vec_sm;
//   register double prev_h1;
//   register double prev_h2;

//   register double next_dv;
//   register double next_dcoords_x;

//   for (int r = r0; r < nr; r += gridDim.y * blockDim.y * row_stride) {

//     double * vec = dv + dirow[r] * lddv;

//     rest_load_col = total_col;
//     rest_comp_col = total_col;
//     curr_ghost_col = min(ghost_col, rest_load_col);

//     // load first ghost
//     if (c0_sm < curr_ghost_col) {
//       vec_sm[c0_sm] = vec[dicol[c0]];
//       if (r0_sm == 0) {
//         dcoords_x_sm[c0_sm] = dcoords_x[dicol[c0]];
//       }
//     }
//     rest_load_col -= curr_ghost_col;
//     //load main column
//     curr_main_col = min(blockDim.x, rest_load_col);
//     if (c0_sm < curr_main_col) {
//       vec_sm[c0_sm + curr_ghost_col] = vec[dicol[c0 + curr_ghost_col *
//       col_stride]]; if (r0_sm == 0) {
//         dcoords_x_sm[c0_sm + curr_ghost_col] = dcoords_x[dicol[c0 +
//         curr_ghost_col * col_stride]];
//       }
//     }
//     rest_load_col -= curr_main_col;
//     __syncthreads();

//     prev_vec_sm = 0.0;
//     prev_h1 = _get_dist(dcoords_x_sm, 0, 1);
//     prev_h2 = _get_dist(dcoords_x_sm, 1, 2);

//     while (rest_comp_col > main_col) {
//       //load next main column
//       next_main_col = min(blockDim.x, rest_load_col);
//       int next_c0 = c0 + (curr_main_col + curr_ghost_col) * col_stride;
//       if (c0_sm < next_main_col) {
//         next_dv = vec[dicol[next_c0]];
//         if (r0_sm == 0) {
//           next_dcoords_x = dcoords_x[dicol[next_c0]];
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
//           h1 = _get_dist(dcoords_x_sm, c0_sm, c0_sm+1);
//           h2 = _get_dist(dcoords_x_sm, c0_sm+1, c0_sm+2);
//           result += h2 * vec_sm[c0_sm+1] / (h1+h2);
//         } else {
//           result = vec_sm[c0_sm];
//           h1 = _get_dist(dcoords_x_sm, c0_sm-2, c0_sm-1);
//           h2 = _get_dist(dcoords_x_sm, c0_sm-1, c0_sm);
//           result += h1 * vec_sm[c0_sm-1] / (h1+h2);
//           h1 = _get_dist(dcoords_x_sm, c0_sm, c0_sm+1);
//           h2 = _get_dist(dcoords_x_sm, c0_sm+1, c0_sm+2);
//           result += h2 * vec_sm[c0_sm+1] / (h1+h2);
//         }
//         vec[dicol[c0]] = result;
//       }

//       rest_comp_col -= main_col;

//       // store last column
//       if (c0_sm == 0) {
//         prev_vec_sm = vec_sm[blockDim.x - 1];
//         prev_h1 = _get_dist(dcoords_x_sm, blockDim.x - 2, blockDim.x - 1);
//         prev_h2 = _get_dist(dcoords_x_sm, blockDim.x - 1, blockDim.x);
//       }

//       __syncthreads();

//       // advance c0
//       c0 += blockDim.x * col_stride;

//       // copy ghost to main
//       next_ghost_col = curr_main_col + curr_ghost_col - main_col;
//       if (c0_sm < next_ghost_col) {
//         vec_sm[c0_sm] = vec_sm[c0_sm + main_col];
//         if (r0_sm == 0) {
//           dcoords_x_sm[c0_sm] = dcoords_x_sm[c0_sm + main_col];
//         }
//       }
//       __syncthreads();
//       // copy next main to main
//       if (c0_sm < next_main_col) {
//         vec_sm[c0_sm + next_ghost_col] = next_dv;
//         if (r0_sm == 0) {
//           dcoords_x_sm[c0_sm + next_ghost_col] = next_dcoords_x;
//         }
//       }
//       rest_load_col -= next_main_col;

//       curr_ghost_col = next_ghost_col;
//       curr_main_col = next_main_col;
//       //rest_col -= real_main_col;
//     } //end while

//     // if (c0_sm < col) {
//     //    vec_sm[c0_sm + real_ghost_col] = vec[dicol[c0 + real_ghost_col *
//     col_stride]];
//     //    dcoords_x_sm[c0_sm + real_ghost_col] = dcoords_x[dicol[c0 +
//     real_ghost_col * col_stride]];
//     // }
//     // __syncthreads();

//     if (rest_comp_col == 1) {
//       if (c0_sm == 0) {
//         result = vec_sm[c0_sm];
//         h1 = prev_h1;
//         h2 = prev_h2;
//         result += h1 * prev_vec_sm / (h1+h2);
//         vec[dicol[c0]] = result;
//       }
//     } else {
//       if (c0_sm < rest_comp_col) {
//         if (c0_sm % 2 == 0) {
//           if (c0_sm == 0) {
//             result = vec_sm[c0_sm];
//             h1 = prev_h1;
//             h2 = prev_h2;
//             result += h1 * prev_vec_sm / (h1+h2);
//             h1 = _get_dist(dcoords_x_sm, c0_sm, c0_sm+1);
//             h2 = _get_dist(dcoords_x_sm, c0_sm+1, c0_sm+2);
//             result += h2 * vec_sm[c0_sm+1] / (h1+h2);
//           } else if (c0_sm == rest_comp_col - 1) {
//             result = vec_sm[c0_sm];
//             h1 = _get_dist(dcoords_x_sm, c0_sm-2, c0_sm-1);;
//             h2 = _get_dist(dcoords_x_sm, c0_sm-1, c0_sm);;
//             result += h1 * vec_sm[c0_sm-1] / (h1+h2);
//           } else {
//             result = vec_sm[c0_sm];
//             h1 = _get_dist(dcoords_x_sm, c0_sm-2, c0_sm-1);
//             h2 = _get_dist(dcoords_x_sm, c0_sm-1, c0_sm);
//             result += h1 * vec_sm[c0_sm-1] / (h1+h2);
//             h1 = _get_dist(dcoords_x_sm, c0_sm, c0_sm+1);
//             h2 = _get_dist(dcoords_x_sm, c0_sm+1, c0_sm+2);
//             result += h2 * vec_sm[c0_sm+1] / (h1+h2);
//           }
//           vec[dicol[c0]] = result;
//         }
//         __syncthreads();
//       }
//     }
//   }
// }

// mgard_cuda_ret
// restriction_l_row_cuda_sm_pf(int nrow,       int ncol,
//                      int nr,         int nc,
//                      int row_stride, int col_stride,
//                      int * dirow,    int * dicol,
//                      double * dv,    int lddv,
//                      double * dcoords_x,
//                      int B, int ghost_col,
//                      mgard_cuda_handle<T> & handle,
//                      int queue_idx, bool profile) {

//   cudaEvent_t start, stop;
//   float milliseconds = 0;
//   cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

//   int total_row = ceil((double)nr/(row_stride));
//   int total_col = ceil((double)nc/(col_stride));
//   int total_thread_y = ceil((double)nr/(row_stride));
//   int total_thread_x = min(B, total_col);

//   int tby = min(B, total_thread_y);
//   int tbx = min(B, total_thread_x);

//   size_t sm_size = ((tbx + ghost_col) * (tby + 1)) * sizeof(double);

//   int gridy = ceil((float)total_thread_y/tby);
//   int gridx = 1; //ceil((float)total_thread_x/tbx);
//   dim3 threadsPerBlock(tbx, tby);
//   dim3 blockPerGrid(gridx, gridy);

//   if (profile) {
//     gpuErrchk(cudaEventCreate(&start));
//     gpuErrchk(cudaEventCreate(&stop));
//     gpuErrchk(cudaEventRecord(start, stream));
//   }

//   _restriction_l_row_cuda_sm_pf<<<blockPerGrid, threadsPerBlock,
//                                   sm_size, stream>>>(nrow,       ncol,
//                                                      nr,         nc,
//                                                      row_stride, col_stride,
//                                                      dirow,      dicol,
//                                                      dv,         lddv,
//                                                      dcoords_x,
//                                                      ghost_col);
//   gpuErrchk(cudaGetLastError ());

//   if (profile) {
//     gpuErrchk(cudaEventRecord(stop, stream));
//     gpuErrchk(cudaEventSynchronize(stop));
//     gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
//     gpuErrchk(cudaEventDestroy(start));
//     gpuErrchk(cudaEventDestroy(stop));
//   }

//   return mgard_cuda_ret(0, milliseconds/1000.0);
// }
} // namespace mgard_cuda
