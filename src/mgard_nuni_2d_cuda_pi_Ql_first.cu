#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda_common.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>
#include <cmath>

namespace mgard_2d {
namespace mgard_gen {  

template <typename T>
__global__ void 
_pi_Ql_first_row_cuda(const int nrow,    const int ncol,
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
      //dv[mgard_common::get_index_cuda(lddv, r, c    )] ++;
      //printf ("thread (%d, %d) working on (%d, %d): %f\n", y, x, r, c, dv[mgard_common::get_index_cuda(lddv, r, c    )]);
      register T center = dv[mgard_common::get_index_cuda(lddv, r, c    )];
      register T left   = dv[mgard_common::get_index_cuda(lddv, r, c - 1)];
      register T right  = dv[mgard_common::get_index_cuda(lddv, r, c + 1)];
      register T h1     = mgard_common::_get_dist(coords_x, c - 1, c    );
      register T h2     = mgard_common::_get_dist(coords_x, c,     c + 1);
      // printf ("thread (%d, %d) working on (%d, %d): %f, left=%f, right=%f\n", y, x, r, c, dv[mgard_common::get_index_cuda(lddv, r, c    )], left, right);


      center -= (h2 * left + h1 * right) / (h1 + h2);
      //center -= (left + right)/2;
      //center -= 1;

      dv[mgard_common::get_index_cuda(lddv, r, c    )] = center;
    }
  }
}

template <typename T>
__global__ void 
_pi_Ql_first_col_cuda(const int nrow,    const int ncol,
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
      //printf ("thread (%d, %d) working on (%d, %d): %f\n", y, x, r, c, dv[mgard_common::get_index_cuda(lddv, r, c    )]);
      register T center = dv[mgard_common::get_index_cuda(lddv, r,     c)];
      register T up   = dv[mgard_common::get_index_cuda(lddv,   r - 1, c)];
      register T down  = dv[mgard_common::get_index_cuda(lddv, r + 1, c)];
      register T h1     = mgard_common::_get_dist(coords_y, r - 1, r    );
      register T h2     = mgard_common::_get_dist(coords_y, r,     r + 1);

      center -= (h2 * up + h1 * down) / (h1 + h2);

      dv[mgard_common::get_index_cuda(lddv, r, c    )] = center;
    }
  }
}

template <typename T>
__global__ void 
_pi_Ql_first_center_cuda(const int nrow,     const int ncol,
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
      //printf ("thread (%d, %d) working on (%d, %d): %f\n", y, x, r, c, dv[mgard_common::get_index_cuda(lddv, r, c    )]);
      register T center    = dv[mgard_common::get_index_cuda(lddv, r,     c    )];
      register T upleft    = dv[mgard_common::get_index_cuda(lddv, r - 1, c - 1)];
      register T upright   = dv[mgard_common::get_index_cuda(lddv, r - 1, c + 1)];
      register T downleft  = dv[mgard_common::get_index_cuda(lddv, r + 1, c - 1)];
      register T downright = dv[mgard_common::get_index_cuda(lddv, r + 1, c + 1)];


      register T x1 = 0.0;
      register T y1 = 0.0;

      register T x2 = mgard_common::_get_dist(dcoords_x, c - 1, c + 1);
      register T y2 = mgard_common::_get_dist(dcoords_y, r - 1, r + 1);

      register T x = mgard_common::_get_dist(dcoords_x, c, c + 1);
      register T y = mgard_common::_get_dist(dcoords_y, r, r + 1);

      T temp =
              mgard_common::interp_2d_cuda(upleft, downleft, upright, downright, x1, x2, y1, y2, x, y);

      center -= temp;

      dv[mgard_common::get_index_cuda(lddv, r, c    )] = center;
    }
  }

}


template <typename T>
mgard_cuda_ret 
pi_Ql_first_cuda(const int nrow,     const int ncol,
                 const int nr,       const int nc,
                 int * dirow,        int * dicol,
                 int * dirowP,       int * dicolP,
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

  _pi_Ql_first_row_cuda<<<blockPerGrid, threadsPerBlock,
                          0, stream>>>(nrow,      ncol,
                                       nr,        nc,
                                       dirow,     dicolP,
                                       dv,        lddv,
                                       dcoords_x, dcoords_y);
  gpuErrchk(cudaGetLastError ()); 
  total_thread_x = nc;
  total_thread_y = nrow-nr;
  tbx = min(B, total_thread_x);
  tby = min(B, total_thread_y);
  gridx = ceil((float)total_thread_x/tbx);
  gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock2(tbx, tby);
  dim3 blockPerGrid2(gridx, gridy);
 
  _pi_Ql_first_col_cuda<<<blockPerGrid2, threadsPerBlock2,
                          0, stream>>>(nrow,      ncol,
                                       nr,        nc,
                                       dirowP,    dicol,
                                       dv,        lddv,
                                       dcoords_x, dcoords_y);
  gpuErrchk(cudaGetLastError ()); 
  total_thread_x = ncol-nc;
  total_thread_y = nrow-nr;
  tbx = min(B, total_thread_x);
  tby = min(B, total_thread_y);
  gridx = ceil((float)total_thread_x/tbx);
  gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock3(tbx, tby);
  dim3 blockPerGrid3(gridx, gridy);
  _pi_Ql_first_center_cuda<<<blockPerGrid3, threadsPerBlock3,
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
pi_Ql_first_cuda<double>(const int nrow,     const int ncol,
                         const int nr,       const int nc,
                         int * dirow,        int * dicol,
                         int * dirowP,       int * dicolP,
                         double * dcoords_x, double * dcoords_y,
                         double * dv,        const int lddv,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);
template mgard_cuda_ret 
pi_Ql_first_cuda<float>(const int nrow,     const int ncol,
                         const int nr,       const int nc,
                         int * dirow,        int * dicol,
                         int * dirowP,       int * dicolP,
                         float * dcoords_x, float * dcoords_y,
                         float * dv,        const int lddv,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);


}
}