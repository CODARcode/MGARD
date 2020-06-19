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
_prolongate_last_row_cuda(const int nrow,    const int ncol,
                          const int nr,      const int nc,
                          int * irow,        int * icolP,
                          T * dv,       int lddv,
                          T * coords_x, T * coords_y) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;

   
   //if (x < ncol-nc && y < nr) {
    for (int y = y0; y < nr; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < ncol-nc; x += blockDim.x * gridDim.x) {

      int r = irow[y];
      int c = icolP[x];
      //dv[get_idx(lddv, r, c    )] ++;
      //printf ("thread (%d, %d) working on (%d, %d): %f\n", y, x, r, c, dv[get_idx(lddv, r, c    )]);
      register T center = dv[get_idx(lddv, r, c    )];
      register T left   = dv[get_idx(lddv, r, c - 1)];
      register T right  = dv[get_idx(lddv, r, c + 1)];
      register T h1     = mgard_common::_get_dist(coords_x, c - 1, c    );
      register T h2     = mgard_common::_get_dist(coords_x, c,     c + 1);
      // printf ("thread (%d, %d) working on (%d, %d): %f, left=%f, right=%f\n", y, x, r, c, dv[get_idx(lddv, r, c    )], left, right);


      center = (h2 * left + h1 * right) / (h1 + h2);
      //center -= (left + right)/2;
      //center -= 1;

      dv[get_idx(lddv, r, c    )] = center;
    }
  }
}

template <typename T>
mgard_cuda_ret 
prolongate_last_row_cuda(const int nrow,     const int ncol,
                         const int nr,       const int nc,
                         int * dirow,        int * dicolP,
                         T * dcoords_x, T * dcoords_y,
                         T * dv,        const int lddv,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile) {  

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  int total_thread_x = ncol-nc;
  int total_thread_y = nr;
  int tbx = min(B, total_thread_x);
  int tby = min(B, total_thread_y);
  int gridx = ceil((float)total_thread_x/tbx);
  int gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  _prolongate_last_row_cuda<<<blockPerGrid, threadsPerBlock,
                          0, stream>>>(nrow,      ncol,
                                       nr,        nc,
                                       dirow,     dicolP,
                                       dv,        lddv,
                                       dcoords_x, dcoords_y);
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
prolongate_last_row_cuda<double>(const int nrow,     const int ncol,
                         const int nr,       const int nc,
                         int * dirow,        int * dicolP,
                         double * dcoords_x, double * dcoords_y,
                         double * dv,        const int lddv,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);
template mgard_cuda_ret 
prolongate_last_row_cuda<float>(const int nrow,     const int ncol,
                         const int nr,       const int nc,
                         int * dirow,        int * dicolP,
                         float * dcoords_x, float * dcoords_y,
                         float * dv,        const int lddv,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);



template <typename T>
__global__ void 
_prolongate_last_col_cuda(const int nrow,    const int ncol,
                      const int nr,      const int nc,
                      int * irowP,       int * icol,
                      T * dv,       int lddv,
                      T * coords_x, T * coords_y) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;

  //if (x < nc && y < nrow-nr) {
  for (int y = y0; y < nrow-nr; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x) {
      int r = irowP[y];
      int c = icol[x];
      //printf ("thread (%d, %d) working on (%d, %d): %f\n", y, x, r, c, dv[get_idx(lddv, r, c    )]);
      register T center = dv[get_idx(lddv, r,     c)];
      register T up   = dv[get_idx(lddv,   r - 1, c)];
      register T down  = dv[get_idx(lddv, r + 1, c)];
      register T h1     = mgard_common::_get_dist(coords_y, r - 1, r    );
      register T h2     = mgard_common::_get_dist(coords_y, r,     r + 1);

      center = (h2 * up + h1 * down) / (h1 + h2);

      dv[get_idx(lddv, r, c    )] = center;
    }
  }
}

template <typename T>
mgard_cuda_ret 
prolongate_last_col_cuda(const int nrow,     const int ncol,
                         const int nr,       const int nc,
                         int * dirowP,        int * dicol,
                         T * dcoords_x, T * dcoords_y,
                         T * dv,        const int lddv,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile) {  

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  int total_thread_x = nc;
  int total_thread_y = nrow-nr;
  int tbx = min(B, total_thread_x);
  int tby = min(B, total_thread_y);
  int gridx = ceil((float)total_thread_x/tbx);
  int gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  _prolongate_last_col_cuda<<<blockPerGrid, threadsPerBlock,
                          0, stream>>>(nrow,      ncol,
                                       nr,        nc,
                                       dirowP,    dicol,
                                       dv,        lddv,
                                       dcoords_x, dcoords_y);
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
prolongate_last_col_cuda<double>(const int nrow,     const int ncol,
                         const int nr,       const int nc,
                         int * dirowP,        int * dicol,
                         double * dcoords_x, double * dcoords_y,
                         double * dv,        const int lddv,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);
template mgard_cuda_ret 
prolongate_last_col_cuda<float>(const int nrow,     const int ncol,
                         const int nr,       const int nc,
                         int * dirowP,        int * dicol,
                         float * dcoords_x, float * dcoords_y,
                         float * dv,        const int lddv,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);


template <typename T>
__global__ void 
_prolongate_last_row_col_cuda(const int nrow,     const int ncol,
                             const int nr,       const int nc,
                             int * dirowP,       int * dicolP,
                             T * dv,        int lddv,
                             T * dcoords_x, T * dcoords_y) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;

  //if (x < ncol-nc && y < nrow-nr) {
  for (int y = y0; y < nrow-nr; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < ncol-nc; x += blockDim.x * gridDim.x) {
      int r = dirowP[y];
      int c = dicolP[x];
      //printf ("thread (%d, %d) working on (%d, %d): %f\n", y, x, r, c, dv[get_idx(lddv, r, c    )]);
      register T center    = dv[get_idx(lddv, r,     c    )];
      register T upleft    = dv[get_idx(lddv, r - 1, c - 1)];
      register T upright   = dv[get_idx(lddv, r - 1, c + 1)];
      register T downleft  = dv[get_idx(lddv, r + 1, c - 1)];
      register T downright = dv[get_idx(lddv, r + 1, c + 1)];

      register T x1 = 0.0;
      register T y1 = 0.0;

      register T x2 = mgard_common::_get_dist(dcoords_x, c - 1, c + 1);
      register T y2 = mgard_common::_get_dist(dcoords_y, r - 1, r + 1);

      register T x = mgard_common::_get_dist(dcoords_x, c, c + 1);
      register T y = mgard_common::_get_dist(dcoords_y, r, r + 1);

      T temp =
              mgard_common::interp_2d_cuda(upleft, downleft, upright, downright, x1, x2, y1, y2, x, y);

      center = temp;

      dv[get_idx(lddv, r, c    )] = center;
    }
  }

}


template <typename T>
mgard_cuda_ret 
prolongate_last_row_col_cuda(const int nrow,     const int ncol,
                         const int nr,       const int nc,
                         int * dirowP,        int * dicolP,
                         T * dcoords_x, T * dcoords_y,
                         T * dv,        const int lddv,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile) {  

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  int total_thread_x = ncol-nc;
  int total_thread_y = nrow-nr;
  int tbx = min(B, total_thread_x);
  int tby = min(B, total_thread_y);
  int gridx = ceil((float)total_thread_x/tbx);
  int gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  _prolongate_last_row_col_cuda<<<blockPerGrid, threadsPerBlock,
                          0, stream>>>(nrow,      ncol,
                                       nr,        nc,
                                       dirowP,    dicolP,
                                       dv,        lddv,
                                       dcoords_x, dcoords_y);
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
prolongate_last_row_col_cuda<double>(const int nrow,     const int ncol,
                         const int nr,       const int nc,
                         int * dirowP,        int * dicolP,
                         double * dcoords_x, double * dcoords_y,
                         double * dv,        const int lddv,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);
template mgard_cuda_ret 
prolongate_last_row_col_cuda<float>(const int nrow,     const int ncol,
                         const int nr,       const int nc,
                         int * dirowP,        int * dicolP,
                         float * dcoords_x, float * dcoords_y,
                         float * dv,        const int lddv,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);
// template <typename T>
// __global__ void
// _prolongate_last_row_cuda(int nrow,       int ncol,
//                           int nr,         int nc,
//                           int row_stride, int col_stride,
//                           int * dirow,    int * dicolP,
//                           T * dv,    int lddv,
//                           T * dcoords_x) {
//   int r0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
//   //int y = threadIdx.y * stride;
//   for (int r = r0; r < nr; r += (blockDim.x * gridDim.x) * row_stride) {
//     // printf("thread %d working on row %d \n", r0, dirow[r]);
//     T * vec = dv + dirow[r] * lddv;
//     for (int i = 0; i < ncol-nc; i++) {
//       T h1 = 1;//mgard_common::get_h_cuda(dcoords_x, icolP[i] - 1, 1);
//       T h2 = 1;//mgard_common::get_h_cuda(dcoords_x, icolP[i]    , 1);
//       T hsum = h1 + h2;
//       // printf("thread %d working on vec = %f %f %f \n", r0, vec[dicolP[i] - 1], vec[dicolP[i]], vec[dicolP[i] + 1]);
//       vec[dicolP[i]] = (h2 * vec[dicolP[i] - 1] + h1 * vec[dicolP[i] + 1]) / hsum;
//       // printf("thread %d working on vec = %f \n", r0, vec[dicolP[i]]);
//     }

//   }
// }

// template <typename T>
// mgard_cuda_ret 
// prolongate_last_row_cuda(int nrow,       int ncol, 
//                          int nr,         int nc,
//                          int row_stride, int col_stride,
//                          int * dirow,    int * dicolP,
//                          T * dv,    int lddv,
//                          T * dcoords_x,
//                          int B, mgard_cuda_handle & handle, 
//                          int queue_idx, bool profile) {

//   cudaEvent_t start, stop;
//   float milliseconds = 0;
//   cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

//   int total_thread_x = ceil((float)nr/row_stride);
//   int tbx = min(B, total_thread_x);
//   int gridx = ceil((float)total_thread_x/tbx);
//   dim3 threadsPerBlock(tbx, 1);
//   dim3 blockPerGrid(gridx, 1);

//   if (profile) {
//     gpuErrchk(cudaEventCreate(&start));
//     gpuErrchk(cudaEventCreate(&stop));
//     gpuErrchk(cudaEventRecord(start, stream));
//   }

//   _prolongate_last_row_cuda<<<blockPerGrid, threadsPerBlock, 
//                               0, stream>>>(nrow,       ncol,
//                                            nr,         nc, 
//                                            row_stride, col_stride,
//                                            dirow,      dicolP,
//                                            dv,         lddv,
//                                            dcoords_x);
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


// template <typename T>
// __global__ void
// _prolongate_last_col_cuda(int nrow,       int ncol,
//                           int nr,         int nc,
//                           int row_stride, int col_stride,
//                           int * dirowP,   int * dicol, 
//                           T * dv,    int lddv,
//                           T * dcoords_y) {
//   int c0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
//   //int y = threadIdx.y * stride;
//   //for (int c = c0; c < nc; c += (blockDim.x * gridDim.x) * col_stride) {
//   for (int c = c0; c < ncol; c += (blockDim.x * gridDim.x) * col_stride) {
//     //printf("thread working on %d \n", x);
//     //double * vec = dv + dicol[c];
//     T * vec = dv + c;
//     for (int i = 0; i < nrow-nr; i++) {
//       T h1 = 1; //mgard_common::get_h_cuda(dcoords_y, irowP[i] - 1, 1);
//       T h2 = 1; //mgard_common::get_h_cuda(dcoords_y, irowP[i]    , 1);
//       T hsum = h1 + h2;
//       // printf("thread %d working on vec = %f %f %f \n", c0, vec[(dirowP[i] - 1)*lddv], vec[dirowP[i]*lddv], vec[(dirowP[i] + 1)*lddv]);
//       vec[dirowP[i] * lddv] = (h2 * vec[(dirowP[i] - 1) * lddv] + h1 * vec[(dirowP[i] + 1) * lddv]) / hsum;
//       // printf("thread %d working on vec = %f \n", c0, vec[dirowP[i] * lddv]);
//     }
//   }
// }

// template <typename T>
// mgard_cuda_ret 
// prolongate_last_col_cuda(int nrow,       int ncol,
//                          int nr,         int nc,
//                          int row_stride, int col_stride,
//                          int * dirowP,   int * dicol, 
//                          T * dv,    int lddv,
//                          T * dcoords_y,
//                          int B, mgard_cuda_handle & handle, 
//                          int queue_idx, bool profile) {
  
//   cudaEvent_t start, stop;
//   float milliseconds = 0;
//   cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

//   //int total_thread_x = ceil((float)nc/col_stride);
//   int total_thread_x = ceil((float)ncol/col_stride);
//   int tbx = min(B, total_thread_x);
//   int gridx = ceil((float)total_thread_x/tbx);
//   dim3 threadsPerBlock(tbx, 1);
//   dim3 blockPerGrid(gridx, 1);

//   if (profile) {
//     gpuErrchk(cudaEventCreate(&start));
//     gpuErrchk(cudaEventCreate(&stop));
//     gpuErrchk(cudaEventRecord(start, stream));
//   }

//   _prolongate_last_col_cuda<<<blockPerGrid, threadsPerBlock,
//                               0, stream>>>(nrow,       ncol, 
//                                            nr,         nc,
//                                            row_stride, col_stride,
//                                            dirowP,     dicol,
//                                            dv,         lddv,
//                                            dcoords_y);
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

// template mgard_cuda_ret 
// prolongate_last_row_cuda<double>(int nrow,       int ncol, 
//                          int nr,         int nc,
//                          int row_stride, int col_stride,
//                          int * dirow,    int * dicolP,
//                          double * dv,    int lddv,
//                          double * dcoords_x,
//                          int B, mgard_cuda_handle & handle, 
//                          int queue_idx, bool profile);
// template mgard_cuda_ret 
// prolongate_last_row_cuda<float>(int nrow,       int ncol, 
//                          int nr,         int nc,
//                          int row_stride, int col_stride,
//                          int * dirow,    int * dicolP,
//                          float * dv,    int lddv,
//                          float * dcoords_x, 
//                          int B, mgard_cuda_handle & handle, 
//                          int queue_idx, bool profile);

// template mgard_cuda_ret 
// prolongate_last_col_cuda<double>(int nrow,       int ncol,
//                          int nr,         int nc,
//                          int row_stride, int col_stride,
//                          int * dirowP,   int * dicol, 
//                          double * dv,    int lddv,
//                          double * dcoords_y,
//                          int B, mgard_cuda_handle & handle, 
//                          int queue_idx, bool profile);
// template mgard_cuda_ret 
// prolongate_last_col_cuda<float>(int nrow,       int ncol,
//                          int nr,         int nc,
//                          int row_stride, int col_stride,
//                          int * dirowP,   int * dicol, 
//                          float * dv,    int lddv,
//                          float * dcoords_y,
//                          int B, mgard_cuda_handle & handle, 
//                          int queue_idx, bool profile);

}
}