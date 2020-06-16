#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>
#include <cmath>

namespace mgard_gen {

template <typename T>
__global__ void 
_pi_Ql3D_first_fib(int nrow,        int ncol,         int nfib,
                   int nr,          int nc,           int nf, 
                   int * irow,      int * icol,       int * ifibP,
                   T * v,           int ldv1,         int ldv2, 
                   T * dist_r,      T * dist_c,       T * dist_f) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;

  for (int z = z0; z < nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nfib-nf; x += blockDim.x * gridDim.x) {

        int f = ifibP[x];
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
mgard_cuda_ret 
pi_Ql3D_first_fib(int nrow,        int ncol,         int nfib,
                  int nr,          int nc,           int nf, 
                  int * dirow,     int * dicol,      int * difibP,
                  T * dv,      int lddv1,        int lddv2, 
                  T * ddist_r, T * ddist_c, T * ddist_f,
                  int B, mgard_cuda_handle & handle, 
                  int queue_idx, bool profile) {
    
  B = min(8, B);  
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_z = nr;
  int total_thread_y = nc;
  int total_thread_x = nfib - nf;

  int tbz = min(B, total_thread_z);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);

  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);

  // std::cout << "threadsPerBlock: " << tbz << ", " << tby << ", " << tbx << std::endl;
  // std::cout << "blockPerGrid: " << gridz << ", " << gridy << ", " << gridx << std::endl;

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _pi_Ql3D_first_fib<<<blockPerGrid, threadsPerBlock, 
                             0, stream>>>(nrow,       ncol,       nfib,
                                          nr,         nc,         nf,
                                          dirow,      dicol,      difibP,
                                          dv,         lddv1,      lddv2,
                                          ddist_r,    ddist_c,    ddist_f);


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
_pi_Ql3D_first_col(int nrow,        int ncol,         int nfib,
                   int nr,          int nc,           int nf, 
                   int * irow,      int * icolP,      int * ifib,
                   T * v,           int ldv1,         int ldv2, 
                   T * dist_r,      T * dist_c,       T * dist_f) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;

  for (int z = z0; z < nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < ncol-nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nf; x += blockDim.x * gridDim.x) {

        int f = ifib[x];
        int c = icolP[y];
        int r = irow[z];

        // if (r == 0 && c == 1 && f == 0) {
        //   printf("thread: %d %d %d %d %d %d\n", x0,y0,z0, x, ifib[x],ifib[x0]);
        // }

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
mgard_cuda_ret 
pi_Ql3D_first_col(int nrow,        int ncol,         int nfib,
                  int nr,          int nc,           int nf, 
                  int * dirow,     int * dicolP,      int * difib,
                  T * dv,      int lddv1,        int lddv2, 
                  T * ddist_r, T * ddist_c, T * ddist_f,
                  int B, mgard_cuda_handle & handle, 
                  int queue_idx, bool profile) {
    
  B = min(8, B);  
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_z = nr;
  int total_thread_y = ncol-nc;
  int total_thread_x = nf;

  int tbz = min(B, total_thread_z);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);

  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);

  // std::cout << "threadsPerBlock: " << tbz << ", " << tby << ", " << tbx << std::endl;
  // std::cout << "blockPerGrid: " << gridz << ", " << gridy << ", " << gridx << std::endl;

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _pi_Ql3D_first_col<<<blockPerGrid, threadsPerBlock, 
                             0, stream>>>(nrow,       ncol,       nfib,
                                          nr,         nc,         nf,
                                          dirow,      dicolP,     difib,
                                          dv,         lddv1,      lddv2,
                                          ddist_r,    ddist_c,    ddist_f);


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
_pi_Ql3D_first_row(int nrow,        int ncol,         int nfib,
                   int nr,          int nc,           int nf, 
                   int * irowP,     int * icol,      int * ifib,
                   T * v,           int ldv1,         int ldv2, 
                   T * dist_r,      T * dist_c,       T * dist_f) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;

  for (int z = z0; z < nrow-nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nf; x += blockDim.x * gridDim.x) {

        int f = ifib[x];
        int c = icol[y];
        int r = irowP[z];

        // if (r == 0 && c == 1 && f == 0) {
        //   printf("thread: %d %d %d %d %d %d\n", x0,y0,z0, x, ifib[x],ifib[x0]);
        // }

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
mgard_cuda_ret 
pi_Ql3D_first_row(int nrow,        int ncol,         int nfib,
                  int nr,          int nc,           int nf, 
                  int * dirowP,    int * dicol,      int * difib,
                  T * dv,      int lddv1,        int lddv2, 
                  T * ddist_r, T * ddist_c, T * ddist_f,
                  int B, mgard_cuda_handle & handle, 
                  int queue_idx, bool profile) {
    
  B = min(8, B);  
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_z = nrow-nr;
  int total_thread_y = nc;
  int total_thread_x = nf;

  int tbz = min(B, total_thread_z);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);

  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);

  // std::cout << "threadsPerBlock: " << tbz << ", " << tby << ", " << tbx << std::endl;
  // std::cout << "blockPerGrid: " << gridz << ", " << gridy << ", " << gridx << std::endl;

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _pi_Ql3D_first_row<<<blockPerGrid, threadsPerBlock, 
                             0, stream>>>(nrow,       ncol,       nfib,
                                          nr,         nc,         nf,
                                          dirowP,     dicol,     difib,
                                          dv,         lddv1,      lddv2,
                                          ddist_r,    ddist_c,    ddist_f);


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
_pi_Ql3D_first_fib_col(int nrow,        int ncol,         int nfib,
                       int nr,          int nc,           int nf, 
                       int * irow,      int * icolP,      int * ifibP,
                       T * v,           int ldv1,         int ldv2, 
                       T * dist_r,      T * dist_c,       T * dist_f) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;

  for (int z = z0; z < nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < ncol-nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nfib-nf; x += blockDim.x * gridDim.x) {

        int f = ifibP[x];
        int c = icolP[y];
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
mgard_cuda_ret 
pi_Ql3D_first_fib_col(int nrow,        int ncol,         int nfib,
                      int nr,          int nc,           int nf, 
                      int * dirow,     int * dicolP,     int * difibP,
                      T * dv,      int lddv1,        int lddv2, 
                      T * ddist_r, T * ddist_c, T * ddist_f,
                      int B, mgard_cuda_handle & handle, 
                      int queue_idx, bool profile) {
    
  B = min(8, B);  
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_z = nr;
  int total_thread_y = ncol - nc;
  int total_thread_x = nfib - nf;

  int tbz = min(B, total_thread_z);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);

  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);

  // std::cout << "threadsPerBlock: " << tbz << ", " << tby << ", " << tbx << std::endl;
  // std::cout << "blockPerGrid: " << gridz << ", " << gridy << ", " << gridx << std::endl;

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _pi_Ql3D_first_fib_col<<<blockPerGrid, threadsPerBlock, 
                             0, stream>>>(nrow,       ncol,       nfib,
                                          nr,         nc,         nf,
                                          dirow,      dicolP,      difibP,
                                          dv,         lddv1,      lddv2,
                                          ddist_r,    ddist_c,    ddist_f);


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
_pi_Ql3D_first_fib_row(int nrow,        int ncol,         int nfib,
                       int nr,          int nc,           int nf, 
                       int * irowP,      int * icol,      int * ifibP,
                       T * v,           int ldv1,         int ldv2, 
                       T * dist_r,      T * dist_c,       T * dist_f) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;

  for (int z = z0; z < nrow-nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nfib-nf; x += blockDim.x * gridDim.x) {

        int f = ifibP[x];
        int c = icol[y];
        int r = irowP[z];

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
mgard_cuda_ret 
pi_Ql3D_first_fib_row(int nrow,        int ncol,         int nfib,
                      int nr,          int nc,           int nf, 
                      int * dirowP,     int * dicol,     int * difibP,
                      T * dv,      int lddv1,        int lddv2, 
                      T * ddist_r, T * ddist_c, T * ddist_f,
                      int B, mgard_cuda_handle & handle, 
                      int queue_idx, bool profile) {
    
  B = min(8, B);  
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_z = nrow - nr;
  int total_thread_y = nc;
  int total_thread_x = nfib - nf;

  int tbz = min(B, total_thread_z);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);

  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);

  // std::cout << "threadsPerBlock: " << tbz << ", " << tby << ", " << tbx << std::endl;
  // std::cout << "blockPerGrid: " << gridz << ", " << gridy << ", " << gridx << std::endl;

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _pi_Ql3D_first_fib_row<<<blockPerGrid, threadsPerBlock, 
                             0, stream>>>(nrow,       ncol,       nfib,
                                          nr,         nc,         nf,
                                          dirowP,     dicol,      difibP,
                                          dv,         lddv1,      lddv2,
                                          ddist_r,    ddist_c,    ddist_f);


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
_pi_Ql3D_first_col_row(int nrow,        int ncol,         int nfib,
                       int nr,          int nc,           int nf, 
                       int * irowP,     int * icolP,      int * ifib,
                       T * v,           int ldv1,         int ldv2, 
                       T * dist_r,      T * dist_c,       T * dist_f) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;

  for (int z = z0; z < nrow-nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < ncol-nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nf; x += blockDim.x * gridDim.x) {

        int f = ifib[x];
        int c = icolP[y];
        int r = irowP[z];

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
mgard_cuda_ret 
pi_Ql3D_first_col_row(int nrow,        int ncol,         int nfib,
                      int nr,          int nc,           int nf, 
                      int * dirowP,    int * dicolP,     int * difib,
                      T * dv,      int lddv1,        int lddv2, 
                      T * ddist_r, T * ddist_c, T * ddist_f,
                      int B, mgard_cuda_handle & handle, 
                      int queue_idx, bool profile) {
    
  B = min(8, B);  
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_z = nrow - nr;
  int total_thread_y = ncol - nc;
  int total_thread_x = nf;

  int tbz = min(B, total_thread_z);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);

  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);

  // std::cout << "threadsPerBlock: " << tbz << ", " << tby << ", " << tbx << std::endl;
  // std::cout << "blockPerGrid: " << gridz << ", " << gridy << ", " << gridx << std::endl;

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _pi_Ql3D_first_col_row<<<blockPerGrid, threadsPerBlock, 
                             0, stream>>>(nrow,       ncol,       nfib,
                                          nr,         nc,         nf,
                                          dirowP,     dicolP,     difib,
                                          dv,         lddv1,      lddv2,
                                          ddist_r,    ddist_c,    ddist_f);


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
_pi_Ql3D_first_fib_col_row(int nrow,        int ncol,         int nfib,
                       int nr,          int nc,           int nf, 
                       int * irowP,     int * icolP,      int * ifibP,
                       T * v,           int ldv1,         int ldv2, 
                       T * dist_r,      T * dist_c,       T * dist_f) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;

  for (int z = z0; z < nrow-nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < ncol-nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nfib-nf; x += blockDim.x * gridDim.x) {

        int f = ifibP[x];
        int c = icolP[y];
        int r = irowP[z];

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
mgard_cuda_ret 
pi_Ql3D_first_fib_col_row(int nrow,        int ncol,         int nfib,
                      int nr,          int nc,           int nf, 
                      int * dirowP,    int * dicolP,     int * difibP,
                      T * dv,      int lddv1,        int lddv2, 
                      T * ddist_r, T * ddist_c, T * ddist_f,
                      int B, mgard_cuda_handle & handle, 
                      int queue_idx, bool profile) {
    
  B = min(8, B);  
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_z = nrow - nr;
  int total_thread_y = ncol - nc;
  int total_thread_x = nfib - nf;

  int tbz = min(B, total_thread_z);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);

  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);

  // std::cout << "threadsPerBlock: " << tbz << ", " << tby << ", " << tbx << std::endl;
  // std::cout << "blockPerGrid: " << gridz << ", " << gridy << ", " << gridx << std::endl;

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _pi_Ql3D_first_fib_col_row<<<blockPerGrid, threadsPerBlock, 
                             0, stream>>>(nrow,       ncol,       nfib,
                                          nr,         nc,         nf,
                                          dirowP,     dicolP,     difibP,
                                          dv,         lddv1,      lddv2,
                                          ddist_r,    ddist_c,    ddist_f);


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
pi_Ql3D_first_fib<double>(int nrow,        int ncol,         int nfib,
                          int nr,          int nc,           int nf, 
                          int * dirow,     int * dicol,      int * difibP,
                          double * dv,      int lddv1,        int lddv2, 
                          double * ddist_r, double * ddist_c, double * ddist_f,
                          int B, mgard_cuda_handle & handle, 
                          int queue_idx, bool profile);
template mgard_cuda_ret 
pi_Ql3D_first_fib<float>(int nrow,        int ncol,         int nfib,
                         int nr,          int nc,           int nf, 
                         int * dirow,     int * dicol,      int * difibP,
                         float * dv,      int lddv1,        int lddv2, 
                         float * ddist_r, float * ddist_c, float * ddist_f,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);

template mgard_cuda_ret 
pi_Ql3D_first_col<double>(int nrow,        int ncol,         int nfib,
                          int nr,          int nc,           int nf, 
                          int * dirow,     int * dicolP,      int * difib,
                          double * dv,      int lddv1,        int lddv2, 
                          double * ddist_r, double * ddist_c, double * ddist_f,
                          int B, mgard_cuda_handle & handle, 
                          int queue_idx, bool profile);
template mgard_cuda_ret 
pi_Ql3D_first_col<float>(int nrow,        int ncol,         int nfib,
                         int nr,          int nc,           int nf, 
                         int * dirow,     int * dicolP,      int * difib,
                         float * dv,      int lddv1,        int lddv2, 
                         float * ddist_r, float * ddist_c, float * ddist_f,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);
template mgard_cuda_ret 
pi_Ql3D_first_row<double>(int nrow,        int ncol,         int nfib,
                          int nr,          int nc,           int nf, 
                          int * dirowP,     int * dicol,      int * difib,
                          double * dv,      int lddv1,        int lddv2, 
                          double * ddist_r, double * ddist_c, double * ddist_f,
                          int B, mgard_cuda_handle & handle, 
                          int queue_idx, bool profile);
template mgard_cuda_ret 
pi_Ql3D_first_row<float>(int nrow,        int ncol,         int nfib,
                         int nr,          int nc,           int nf, 
                         int * dirowP,     int * dicol,      int * difib,
                         float * dv,      int lddv1,        int lddv2, 
                         float * ddist_r, float * ddist_c, float * ddist_f,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);

template mgard_cuda_ret 
pi_Ql3D_first_fib_col<double>(int nrow,        int ncol,         int nfib,
                          int nr,          int nc,           int nf, 
                          int * dirow,     int * dicolP,      int * difibP,
                          double * dv,      int lddv1,        int lddv2, 
                          double * ddist_r, double * ddist_c, double * ddist_f,
                          int B, mgard_cuda_handle & handle, 
                          int queue_idx, bool profile);
template mgard_cuda_ret 
pi_Ql3D_first_fib_col<float>(int nrow,        int ncol,         int nfib,
                         int nr,          int nc,           int nf, 
                         int * dirow,     int * dicolP,      int * difibP,
                         float * dv,      int lddv1,        int lddv2, 
                         float * ddist_r, float * ddist_c, float * ddist_f,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);
template mgard_cuda_ret 
pi_Ql3D_first_fib_row<double>(int nrow,        int ncol,         int nfib,
                          int nr,          int nc,           int nf, 
                          int * dirowP,    int * dicol,      int * difibP,
                          double * dv,      int lddv1,        int lddv2, 
                          double * ddist_r, double * ddist_c, double * ddist_f,
                          int B, mgard_cuda_handle & handle, 
                          int queue_idx, bool profile);
template mgard_cuda_ret 
pi_Ql3D_first_fib_row<float>(int nrow,        int ncol,         int nfib,
                         int nr,          int nc,           int nf, 
                         int * dirowP,    int * dicol,      int * difibP,
                         float * dv,      int lddv1,        int lddv2, 
                         float * ddist_r, float * ddist_c, float * ddist_f,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);

template mgard_cuda_ret 
pi_Ql3D_first_col_row<double>(int nrow,        int ncol,         int nfib,
                          int nr,          int nc,           int nf, 
                          int * dirowP,    int * dicolP,      int * difib,
                          double * dv,      int lddv1,        int lddv2, 
                          double * ddist_r, double * ddist_c, double * ddist_f,
                          int B, mgard_cuda_handle & handle, 
                          int queue_idx, bool profile);
template mgard_cuda_ret 
pi_Ql3D_first_col_row<float>(int nrow,        int ncol,         int nfib,
                         int nr,          int nc,           int nf, 
                         int * dirowP,    int * dicolP,      int * difib,
                         float * dv,      int lddv1,        int lddv2, 
                         float * ddist_r, float * ddist_c, float * ddist_f,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);
template mgard_cuda_ret 
pi_Ql3D_first_fib_col_row<double>(int nrow,        int ncol,         int nfib,
                          int nr,          int nc,           int nf, 
                          int * dirowP,    int * dicolP,      int * difibP,
                          double * dv,      int lddv1,        int lddv2, 
                          double * ddist_r, double * ddist_c, double * ddist_f,
                          int B, mgard_cuda_handle & handle, 
                          int queue_idx, bool profile);
template mgard_cuda_ret 
pi_Ql3D_first_fib_col_row<float>(int nrow,        int ncol,         int nfib,
                         int nr,          int nc,           int nf, 
                         int * dirowP,    int * dicolP,      int * difibP,
                         float * dv,      int lddv1,        int lddv2, 
                         float * ddist_r, float * ddist_c, float * ddist_f,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);


}