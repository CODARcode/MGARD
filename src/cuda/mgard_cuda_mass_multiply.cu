#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda_mass_multiply.h"

namespace mgard_cuda {

template <typename T>
__global__ void _mass_multiply_1(int nrow, int ncol, int nr, int nc,
                                 int row_stride, int col_stride, int *dirow,
                                 int *dicol, T *dcoords_x, T *dv, int lddv) {
  int r0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  for (int r = r0; r < nr; r += (blockDim.x * gridDim.x) * row_stride) {
    T *vec = dv + dirow[r] * lddv;
    T temp1, temp2;
    T h1, h2;
    temp1 = vec[dicol[0]];
    h1 = _get_dist(dcoords_x, dicol[0],
                   dicol[col_stride]); // dicol[col_stride] - dicol[0];
    vec[dicol[0]] = 2.0 * h1 * temp1 + h1 * vec[dicol[col_stride]];
    for (int i = col_stride; i <= nc - 1 - col_stride; i += col_stride) {
      temp2 = vec[dicol[i]];
      h1 = _get_dist(dcoords_x, dicol[i - col_stride], dicol[i]);
      h2 = _get_dist(dcoords_x, dicol[i], dicol[i + col_stride]);
      vec[dicol[i]] =
          h1 * temp1 + 2 * (h1 + h2) * temp2 + h2 * vec[dicol[i + col_stride]];
      temp1 = temp2;
    }
    vec[dicol[nc - 1]] =
        _get_dist(dcoords_x, dicol[nc - col_stride - 1], dicol[nc - 1]) *
            temp1 +
        2 * _get_dist(dcoords_x, dicol[nc - col_stride - 1], dicol[nc - 1]) *
            vec[dicol[nc - 1]];
  }
}

template <typename T>
void mass_multiply_1(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nr,
                     int nc, int row_stride, int col_stride, int *dirow,
                     int *dicol, T *dcoords_x, T *dv, int lddv, int queue_idx) {

  int total_thread_x = ceil((float)nr / (row_stride));
  int tbx = min(handle.B, total_thread_x);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);
  _mass_multiply_1<<<blockPerGrid, threadsPerBlock, 0,
                     *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nr, nc, row_stride, col_stride, dirow, dicol, dcoords_x, dv,
      lddv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void mass_multiply_1<double>(mgard_cuda_handle<double> &handle,
                                      int nrow, int ncol, int nr, int nc,
                                      int row_stride, int col_stride,
                                      int *dirow, int *dicol, double *dcoords_x,
                                      double *dv, int lddv, int queue_idx);
template void mass_multiply_1<float>(mgard_cuda_handle<float> &handle, int nrow,
                                     int ncol, int nr, int nc, int row_stride,
                                     int col_stride, int *dirow, int *dicol,
                                     float *dcoords_x, float *dv, int lddv,
                                     int queue_idx);

template <typename T>
__global__ void _mass_multiply_2(int nrow, int ncol, int nr, int nc,
                                 int row_stride, int col_stride, int *dirow,
                                 int *dicol, T *dcoords_y, T *dv, int lddv) {
  int c0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  for (int c = c0; c < nc; c += (blockDim.x * gridDim.x) * col_stride) {
    T *vec = dv + dicol[c];
    T temp1, temp2;
    T h1, h2;
    temp1 = vec[dirow[0] * lddv];
    h1 = _get_dist(dcoords_y, dirow[0],
                   dirow[row_stride]); // dirow[row_stride] - dirow[0];
    vec[dirow[0] * lddv] =
        2.0 * h1 * temp1 + h1 * vec[dirow[row_stride] * lddv];
    for (int i = row_stride; i <= nr - 1 - row_stride; i += row_stride) {
      temp2 = vec[dirow[i] * lddv];
      h1 = _get_dist(dcoords_y, dirow[i - row_stride], dirow[i]);
      h2 = _get_dist(dcoords_y, dirow[i], dirow[i + row_stride]);
      vec[dirow[i] * lddv] = h1 * temp1 + 2 * (h1 + h2) * temp2 +
                             h2 * vec[dirow[i + row_stride] * lddv];
      temp1 = temp2;
    }
    vec[dirow[nr - 1] * lddv] =
        _get_dist(dcoords_y, dirow[nr - row_stride - 1], dirow[nr - 1]) *
            temp1 +
        2 * _get_dist(dcoords_y, dirow[nr - row_stride - 1], dirow[nr - 1]) *
            vec[dirow[nr - 1] * lddv];
  }
}

template <typename T>
void mass_multiply_2(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nr,
                     int nc, int row_stride, int col_stride, int *dirow,
                     int *dicol, T *dcoords_y, T *dv, int lddv, int queue_idx) {

  int total_thread_x = ceil((float)nc / (col_stride));
  int tbx = min(handle.B, total_thread_x);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);
  _mass_multiply_2<<<blockPerGrid, threadsPerBlock, 0,
                     *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nr, nc, row_stride, col_stride, dirow, dicol, dcoords_y, dv,
      lddv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void mass_multiply_2<double>(mgard_cuda_handle<double> &handle,
                                      int nrow, int ncol, int nr, int nc,
                                      int row_stride, int col_stride,
                                      int *dirow, int *dicol, double *dcoords_y,
                                      double *dv, int lddv, int queue_idx);
template void mass_multiply_2<float>(mgard_cuda_handle<float> &handle, int nrow,
                                     int ncol, int nr, int nc, int row_stride,
                                     int col_stride, int *dirow, int *dicol,
                                     float *dcoords_y, float *dv, int lddv,
                                     int queue_idx);

template <typename T>
__global__ void _mass_multiply_1_cpt(int nr, int nc, int row_stride,
                                     int col_stride, T *ddist_x, int ghost_col,
                                     T *__restrict__ dv, int lddv) {

  register int total_row = ceil((double)nr / (row_stride));
  register int total_col = ceil((double)nc / (col_stride));

  // index on dirow and dicol
  register int r0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  register int c0 = threadIdx.x;
  register int c0_stride = threadIdx.x * col_stride;

  // index on sm
  register int r0_sm = threadIdx.y;
  register int c0_sm = threadIdx.x;

  // extern __shared__ double sm[]; // row = blockDim.y; col = blockDim.x +
  // ghost_col;

  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);

  T *sm = SharedMemory<T>();

  register int ldsm = blockDim.x + ghost_col;

  T *vec_sm = sm + r0_sm * ldsm;
  T *dist_x_sm = sm + (blockDim.y) * ldsm;

  register T result = 1;
  register T h1 = 1;
  register T h2 = 1;

  register int rest_col;
  register int real_ghost_col;
  register int real_main_col;

  register T prev_vec_sm;
  register T prev_dist_x;

  for (int r = r0; r < nr; r += gridDim.y * blockDim.y * row_stride) {

    T *vec = dv + r * lddv;

    prev_vec_sm = 0.0;
    prev_dist_x = 0.0;

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

      // computation
      if (c0_sm == 0) {
        h1 = prev_dist_x;
        h2 = dist_x_sm[c0_sm];
        result = h1 * prev_vec_sm + 2 * (h1 + h2) * vec_sm[c0_sm] +
                 h2 * vec_sm[c0_sm + 1];
      } else {
        h1 = dist_x_sm[c0_sm - 1];
        h2 = dist_x_sm[c0_sm];
        result = h1 * vec_sm[c0_sm - 1] + 2 * (h1 + h2) * vec_sm[c0_sm] +
                 h2 * vec_sm[c0_sm + 1];
      }
      vec[c0] = result;
      __syncthreads();

      // store last column
      if (c0_sm == 0) {
        prev_vec_sm = vec_sm[blockDim.x - 1];
        prev_dist_x = dist_x_sm[blockDim.x - 1];
      }

      // advance c0
      c0_stride += blockDim.x * col_stride;
      c0 += blockDim.x;

      // copy ghost to main
      real_ghost_col = min(ghost_col, real_main_col - (blockDim.x - ghost_col));
      if (c0_sm < real_ghost_col) {
        vec_sm[c0_sm] = vec_sm[c0_sm + blockDim.x];
        if (r0_sm == 0) {
          dist_x_sm[c0_sm] = dist_x_sm[c0_sm + blockDim.x];
        }
      }
      __syncthreads();
      rest_col -= real_main_col;
    } // end while

    if (c0_sm < rest_col) {
      vec_sm[c0_sm + real_ghost_col] =
          vec[c0_stride + real_ghost_col * col_stride];
      if (r0_sm == 0) {
        dist_x_sm[c0_sm + real_ghost_col] = ddist_x[c0 + real_ghost_col];
      }
    }
    __syncthreads();

    if (real_ghost_col + rest_col == 1) {
      if (c0_sm == 0) {
        h1 = prev_dist_x;
        result = h1 * prev_vec_sm + 2 * h1 * vec_sm[c0_sm];
        vec[c0_stride] = result;
      }
    } else {
      if (c0_sm < real_ghost_col + rest_col) {
        if (c0_sm == 0) {
          h1 = prev_dist_x;
          h2 = dist_x_sm[c0_sm];
          result = h1 * prev_vec_sm + 2 * (h1 + h2) * vec_sm[c0_sm] +
                   h2 * vec_sm[c0_sm + 1];
        } else if (c0_sm == real_ghost_col + rest_col - 1) {
          h1 = dist_x_sm[c0_sm - 1];
          result = h1 * vec_sm[c0_sm - 1] + 2 * h1 * vec_sm[c0_sm];
        } else {
          h1 = dist_x_sm[c0_sm - 1];
          h2 = dist_x_sm[c0_sm];
          result = h1 * vec_sm[c0_sm - 1] + 2 * (h1 + h2) * vec_sm[c0_sm] +
                   h2 * vec_sm[c0_sm + 1];
        }
        __syncthreads();
        vec[c0_stride] = result;
      }
    }
  }
}

template <typename T>
void mass_multiply_1_cpt(mgard_cuda_handle<T> &handle, int nr, int nc,
                         int row_stride, int col_stride, T *ddist_x, T *dv,
                         int lddv, int queue_idx) {

  int total_row = ceil((double)nr / (row_stride));
  int total_col = ceil((double)nc / (col_stride));
  int total_thread_y = ceil((double)nr / (row_stride));
  int total_thread_x = min(handle.B, total_col);
  int tby = min(handle.B, total_thread_y);
  int tbx = min(handle.B, total_thread_x);
  int ghost_col = handle.B;
  size_t sm_size = ((tbx + ghost_col) * (tby + 1 + 1)) * sizeof(T);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = 1;
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _mass_multiply_1_cpt<<<blockPerGrid, threadsPerBlock, sm_size,
                         *(cudaStream_t *)handle.get(queue_idx)>>>(
      nr, nc, row_stride, col_stride, ddist_x, ghost_col, dv, lddv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void mass_multiply_1_cpt<double>(mgard_cuda_handle<double> &handle,
                                          int nr, int nc, int row_stride,
                                          int col_stride, double *ddist_x,
                                          double *dv, int lddv, int queue_idx);
template void mass_multiply_1_cpt<float>(mgard_cuda_handle<float> &handle,
                                         int nr, int nc, int row_stride,
                                         int col_stride, float *ddist_x,
                                         float *dv, int lddv, int queue_idx);

template <typename T>
__global__ void _mass_multiply_2_cpt(int nr, int nc, int row_stride,
                                     int col_stride, T *ddist_y, int ghost_row,
                                     T *__restrict__ dv, int lddv) {

  register int total_row = ceil((double)nr / (row_stride));
  register int total_col = ceil((double)nc / (col_stride));

  // index on dirow and dicol
  register int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;
  register int r0 = threadIdx.y;
  register int r0_stride = threadIdx.y * row_stride;
  register int c_dist = threadIdx.x;

  // index on sm
  register int r0_sm = threadIdx.y;
  register int c0_sm = threadIdx.x;

  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem); // row = blockDim.y; col = blockDim.x
  // + ghost_col;
  T *sm = SharedMemory<T>();

  register int ldsm = blockDim.x;
  T *vec_sm = sm + c0_sm;
  T *dist_y_sm = sm + (blockDim.y + ghost_row) * ldsm;

  register T result = 1;
  register T h1 = 1;
  register T h2 = 1;

  register int rest_row;
  register int real_ghost_row;
  register int real_main_row;

  register T prev_vec_sm;
  register T prev_dist_y;

  for (int c = c0; c < nc; c += gridDim.x * blockDim.x * col_stride) {

    T *vec = dv + c;

    prev_vec_sm = 0.0;
    prev_dist_y = 0.0;

    rest_row = total_row;
    real_ghost_row = min(ghost_row, rest_row);

    // load first ghost
    if (r0_sm < real_ghost_row) {
      vec_sm[r0_sm * ldsm] = vec[r0_stride * lddv];
    }
    if (c0_sm == 0 && r0_sm < real_ghost_row) {
      dist_y_sm[r0_sm] = ddist_y[r0];
    }
    __syncthreads();

    rest_row -= real_ghost_row;

    while (rest_row > blockDim.y - real_ghost_row) {
      // load main column
      real_main_row = min(blockDim.y, rest_row);
      if (r0_sm < real_main_row) {
        vec_sm[(r0_sm + real_ghost_row) * ldsm] =
            vec[(r0_stride + real_ghost_row * row_stride) * lddv];
      }
      if (c0_sm == 0 && r0_sm < real_main_row) {
        dist_y_sm[r0_sm + real_ghost_row] = ddist_y[r0 + real_ghost_row];
      }
      __syncthreads();

      // computation
      if (r0_sm == 0) {
        h1 = prev_dist_y;
        h2 = dist_y_sm[r0_sm]; // broadcast from sm
        result = h1 * prev_vec_sm + 2 * (h1 + h2) * vec_sm[r0_sm * ldsm] +
                 h2 * vec_sm[(r0_sm + 1) * ldsm];
      } else {
        h1 = dist_y_sm[r0_sm - 1];
        h2 = dist_y_sm[r0_sm];
        result = h1 * vec_sm[(r0_sm - 1) * ldsm] +
                 2 * (h1 + h2) * vec_sm[r0_sm * ldsm] +
                 h2 * vec_sm[(r0_sm + 1) * ldsm];
      }
      vec[r0_stride * lddv] = result;
      __syncthreads();

      // store last row
      if (r0_sm == 0) {
        prev_vec_sm = vec_sm[(blockDim.y - 1) * ldsm];
        prev_dist_y = dist_y_sm[blockDim.y - 1];
      }
      // advance c0
      r0_stride += blockDim.y * row_stride;
      r0 += blockDim.y;
      c_dist += blockDim.y;
      __syncthreads();
      // copy ghost to main
      real_ghost_row = min(ghost_row, real_main_row - (blockDim.y - ghost_row));
      if (r0_sm < real_ghost_row) {
        vec_sm[r0_sm * ldsm] = vec_sm[(r0_sm + blockDim.y) * ldsm];
      }
      if (c0_sm == 0 && r0_sm < real_ghost_row) {
        dist_y_sm[r0_sm] = dist_y_sm[r0_sm + blockDim.y];
      }

      rest_row -= real_main_row;
      __syncthreads();
    } // end while

    if (r0_sm < rest_row) {
      vec_sm[(r0_sm + real_ghost_row) * ldsm] =
          vec[(r0_stride + real_ghost_row * row_stride) * lddv];
    }
    //__syncthreads();
    if (c0_sm == 0 && r0_sm < rest_row) {
      dist_y_sm[r0_sm + real_ghost_row] = ddist_y[r0 + real_ghost_row];
    }

    if (real_ghost_row + rest_row == 1) {

      if (r0_sm == 0) {
        h1 = prev_dist_y;
        result = h1 * prev_vec_sm + 2 * h1 * vec_sm[r0_sm * ldsm];
        vec[r0_stride * lddv] = result;
      }
      //__syncthreads();
    } else {
      if (r0_sm < real_ghost_row + rest_row) {
        if (r0_sm == 0) {
          h1 = prev_dist_y;
          h2 = dist_y_sm[r0_sm];
          result = h1 * prev_vec_sm + 2 * (h1 + h2) * vec_sm[r0_sm * ldsm] +
                   h2 * vec_sm[(r0_sm + 1) * ldsm];
        } else if (r0_sm == real_ghost_row + rest_row - 1) {
          h1 = dist_y_sm[r0_sm - 1];
          result =
              h1 * vec_sm[(r0_sm - 1) * ldsm] + 2 * h1 * vec_sm[r0_sm * ldsm];
        } else {
          h1 = dist_y_sm[r0_sm - 1];
          h2 = dist_y_sm[r0_sm];
          result = h1 * vec_sm[(r0_sm - 1) * ldsm] +
                   2 * (h1 + h2) * vec_sm[r0_sm * ldsm] +
                   h2 * vec_sm[(r0_sm + 1) * ldsm];
        }
        vec[r0_stride * lddv] = result;
      }
    }
    __syncthreads();
  }
}

template <typename T>
void mass_multiply_2_cpt(mgard_cuda_handle<T> &handle, int nr, int nc,
                         int row_stride, int col_stride, T *ddist_y, T *dv,
                         int lddv, int queue_idx) {

  int total_row = ceil((double)nr / (row_stride));
  int total_col = ceil((double)nc / (col_stride));
  int total_thread_y = min(handle.B, total_row);
  int total_thread_x = ceil((double)nc / (col_stride));
  int tby = min(handle.B, total_thread_y);
  int tbx = min(handle.B, total_thread_x);
  int ghost_row = handle.B;
  size_t sm_size = ((tby + ghost_row) * (tbx + 1)) * sizeof(T);
  int gridy = 1;
  int gridx =
      ceil((float)total_thread_x / tbx); // ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _mass_multiply_2_cpt<<<blockPerGrid, threadsPerBlock, sm_size,
                         *(cudaStream_t *)handle.get(queue_idx)>>>(
      nr, nc, row_stride, col_stride, ddist_y, ghost_row, dv, lddv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void mass_multiply_2_cpt<double>(mgard_cuda_handle<double> &handle,
                                          int nr, int nc, int row_stride,
                                          int col_stride, double *ddist_y,
                                          double *dv, int lddv, int queue_idx);
template void mass_multiply_2_cpt<float>(mgard_cuda_handle<float> &handle,
                                         int nr, int nc, int row_stride,
                                         int col_stride, float *ddist_y,
                                         float *dv, int lddv, int queue_idx);

/*
template <typename T>
__global__ void
_mass_mult_l_row_cuda_sm_pf(int nrow,       int ncol,
                         int nr,         int nc,
                         int row_stride, int col_stride,
                         int * __restrict__ dirow,    int * __restrict__ dicol,
                         T * __restrict__ dv,    int lddv,
                         T * __restrict__ dcoords_x,
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

  extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  T * sm = reinterpret_cast<T *>(smem);

  //extern __shared__ T sm[]; // row = blockDim.y; col = blockDim.x + ghost_col;
  register int ldsm = blockDim.x + ghost_col;
  // printf("ldsm = %d\n", ldsm);

  T * vec_sm = sm + r0_sm * ldsm;
  T * dcoords_x_sm = sm + blockDim.y * ldsm;

  register T result = 1;
  register T h1 = 1;
  register T h2 = 1;

  register int main_col = blockDim.x;
  register int rest_load_col;
  register int rest_comp_col;
  register int curr_ghost_col;
  register int curr_main_col;
  register int next_ghost_col;
  register int next_main_col;

  register T prev_vec_sm;
  register T prev_dicol;
  register T prev_dcoord_x;

  register T next_dv;
  register T next_dcoords_x;

  for (int r = r0; r < nr; r += gridDim.y * blockDim.y * row_stride) {

    T * vec = dv + dirow[r] * lddv;

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
      vec_sm[c0_sm + curr_ghost_col] = vec[dicol[c0 + curr_ghost_col *
col_stride]]; if (r0_sm == 0) { dcoords_x_sm[c0_sm + curr_ghost_col] =
dcoords_x[dicol[c0 + curr_ghost_col * col_stride]];
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
        //h1 = _get_dist(dcoords_x,  prev_dicol, dicol[c0]);
        h1 = dcoords_x_sm[c0_sm] - prev_dcoord_x;
        //h2 = _get_dist(dcoords_x,  dicol[c0], dicol[c0 + col_stride]);
        h2 = _get_dist(dcoords_x_sm,  c0_sm, c0_sm + 1);

        result = h1 * prev_vec_sm + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 *
vec_sm[c0_sm + 1]; } else {
        //h1 = _get_dist(dcoords_x, dicol[c0 - col_stride], dicol[c0]);
        h1 = _get_dist(dcoords_x_sm, c0_sm - 1, c0_sm);
        //h2 = _get_dist(dcoords_x, dicol[c0], dicol[c0 + col_stride]);
        h2 = _get_dist(dcoords_x_sm, c0_sm, c0_sm + 1);

        result = h1 * vec_sm[c0_sm - 1] + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 *
vec_sm[c0_sm + 1];
      }
      vec[dicol[c0]] = result;

      rest_comp_col -= main_col;

      // store last column
      if (c0_sm == 0) {
        prev_vec_sm = vec_sm[blockDim.x - 1];
        prev_dicol = dicol[c0 + (blockDim.x - 1) * col_stride];
        prev_dcoord_x = dcoords_x[dicol[c0 + (blockDim.x - 1) *
col_stride]];//dcoords_x_sm[blockDim.x - 1];
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
    //    vec_sm[c0_sm + real_ghost_col] = vec[dicol[c0 + real_ghost_col *
col_stride]];
    //    dcoords_x_sm[c0_sm + real_ghost_col] = dcoords_x[dicol[c0 +
real_ghost_col * col_stride]];
    // }
    // __syncthreads();

    if (rest_comp_col == 1) {
      if (c0_sm == 0) {
        //h1 = _get_dist(dcoords_x,  prev_dicol, dicol[c0]);
        h1 = dcoords_x_sm[c0_sm] - prev_dcoord_x;
        result = h1 * prev_vec_sm + 2 * h1 * vec_sm[c0_sm];
        vec[dicol[c0]] = result;
      }

    } else {

    if (c0_sm < rest_comp_col) {

      if (c0_sm == 0) {
        //h1 = _get_dist(dcoords_x,  prev_dicol, dicol[c0]);
        //h2 = _get_dist(dcoords_x,  dicol[c0], dicol[c0 + col_stride]);

        h1 = dcoords_x_sm[c0_sm] - prev_dcoord_x;
        h2 = _get_dist(dcoords_x_sm,  c0_sm, c0_sm + 1);

        result = h1 * prev_vec_sm + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 *
vec_sm[c0_sm + 1]; } else if (c0_sm == rest_comp_col - 1) {
        //h1 = _get_dist(dcoords_x, dicol[c0 - col_stride], dicol[c0]);
        h1 = _get_dist(dcoords_x_sm, c0_sm - 1, c0_sm);
        result = h1 * vec_sm[c0_sm - 1] + 2 * h1 * vec_sm[c0_sm];
      } else {
        // h1 = _get_dist(dcoords_x, dicol[c0 - col_stride], dicol[c0]);
        // h2 = _get_dist(dcoords_x, dicol[c0], dicol[c0 + col_stride]);

        h1 = _get_dist(dcoords_x_sm, c0_sm - 1, c0_sm);
        h2 = _get_dist(dcoords_x_sm, c0_sm, c0_sm + 1);
        result = h1 * vec_sm[c0_sm - 1] + 2 * (h1 + h2) * vec_sm[c0_sm] + h2 *
vec_sm[c0_sm + 1];
      }
      __syncthreads();
      vec[dicol[c0]] = result;
    }
  }


  }
}

template <typename T>
mgard_cuda_ret
mass_mult_l_row_cuda_sm_pf(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     T * dv,    int lddv,
                     T * dcoords_x,
                     int B, int ghost_col,
                     mgard_cuda_handle<T> & handle,
                     int queue_idx, bool profile) {


  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

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

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _mass_mult_l_row_cuda_sm_pf<<<blockPerGrid, threadsPerBlock,
                                sm_size, stream>>>(nrow,       ncol,
                                                   nr,         nc,
                                                   row_stride, col_stride,
                                                   dirow,      dicol,
                                                   dv,         lddv,
                                                   dcoords_x,
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




template mgard_cuda_ret
mass_mult_l_col_cuda<double>(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     double * dv,    int lddv,
                     double * dcoords_y,
                     int B, mgard_cuda_handle<double> & handle,
                     int queue_idx, bool profile);
template mgard_cuda_ret
mass_mult_l_col_cuda<float>(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     float * dv,    int lddv,
                     float * dcoords_y,
                     int B, mgard_cuda_handle<float> & handle,
                     int queue_idx, bool profile);


template mgard_cuda_ret
mass_mult_l_row_cuda_sm<double>(int nr,         int nc,
                        int row_stride, int col_stride,
                        double * dv,    int lddv,
                        double * ddist_x,
                        int B, int ghost_col,
                        mgard_cuda_handle<double> & handle,
                        int queue_idx, bool profile);

template mgard_cuda_ret
mass_mult_l_row_cuda_sm<float>(int nr,         int nc,
                        int row_stride, int col_stride,
                        float * dv,    int lddv,
                        float * ddist_x,
                        int B, int ghost_col,
                        mgard_cuda_handle<float> & handle,
                        int queue_idx, bool profile);

template mgard_cuda_ret
mass_mult_l_col_cuda_sm<double>(int nr,         int nc,
                        int row_stride, int col_stride,
                        double * dv,    int lddv,
                        double * ddist_y,
                        int B, int ghost_row,
                        mgard_cuda_handle<double> & handle,
                        int queue_idx, bool profile);

template mgard_cuda_ret
mass_mult_l_col_cuda_sm<float>(int nr,         int nc,
                        int row_stride, int col_stride,
                        float * dv,    int lddv,
                        float * ddist_y,
                        int B, int ghost_row,
                        mgard_cuda_handle<float> & handle,
                        int queue_idx, bool profile);

template mgard_cuda_ret
mass_mult_l_row_cuda_sm_pf<double>(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     double * dv,    int lddv,
                     double * dcoords_x,
                     int B, int ghost_col,
                     mgard_cuda_handle<double> & handle,
                     int queue_idx, bool profile);
template mgard_cuda_ret
mass_mult_l_row_cuda_sm_pf<float>(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     float * dv,    int lddv,
                     float * dcoords_x,
                     int B, int ghost_col,
                     mgard_cuda_handle<float> & handle,
                     int queue_idx, bool profile);
*/

} // namespace mgard_cuda
