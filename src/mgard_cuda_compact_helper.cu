#include <iomanip> 
#include <iostream>
#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"
#include "mgard_cuda.h"
#include "mgard_cuda_compacted.h"


/* 2D Original to (2^k)+1 */
__global__ void
_org_to_pow2p1(int nrow,     int ncol,
                      int nr,       int nc,
                      int * irow,   int * icol,
                      double * dv,  int lddv,
                      double * dcv, int lddcv) {
  
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;

  for (int y = y0; y < nr; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x) {
      dcv[get_idx(lddcv, y, x)] = dv[get_idx(lddv, irow[y], icol[x])];
    }
  }
}


mgard_cuda_ret 
org_to_pow2p1(int nrow,     int ncol,
             int nr,       int nc,
             int * dirow,  int * dicol,
             double * dv,  int lddv,
             double * dcv, int lddcv) {

  int B = 16;
  int total_thread_y = nr;
  int total_thread_x = nc;
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "_copy_level_l_cuda" << std::endl;
  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _org_to_pow2p1<<<blockPerGrid, threadsPerBlock>>>(nrow,  ncol,
                                                           nr,    nc,
                                                           dirow, dicol,
                                                           dv,    lddv,
                                                           dcv,   lddcv);

  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}


/* 1D Original to (2^k)+1 */
__global__ void
_org_to_pow2p1(int nrow,    int nr,       
               int * irow, 
               double * dv, double * dcv) {
  
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int x = x0; x < nr; x += blockDim.x * gridDim.x) {
    dcv[x] = dv[irow[x]];
  }
}


mgard_cuda_ret 
org_to_pow2p1(int nrow,    int nr,
              int * dirow, 
              double * dv, double * dcv) {

  int B = 16;
  int total_thread_y = 1;
  int total_thread_x = nr;
  int tby = 1;
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "_copy_level_l_cuda" << std::endl;
  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _org_to_pow2p1<<<blockPerGrid, threadsPerBlock>>>(nrow,  nr,    
                                                            dirow,
                                                            dv,    dcv);

  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}




/* 2D (2^k)+1 to original*/
__global__ void
_pow2p1_to_org(int nrow,     int ncol,
                        int nr,       int nc,
                        int * irow,   int * icol,
                        double * dcv,  int lddcv,
                        double * dv, int lddv) {
  
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int y = y0; y < nr; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x) {
       dv[get_idx(lddv, irow[y], icol[x])] = dcv[get_idx(lddcv, y, x)];
    }
  }
}

mgard_cuda_ret 
pow2p1_to_org(int nrow,     int ncol,
                       int nr,       int nc,
                       int * dirow,  int * dicol,
                       double * dcv,  int lddcv,
                       double * dv, int lddv) {

  int B = 16;
  int total_thread_y = nr;
  int total_thread_x = nc;
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "_copy_level_l_cuda" << std::endl;
  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _pow2p1_to_org<<<blockPerGrid, threadsPerBlock>>>(nrow,  ncol,
                                                             nr,    nc,
                                                             dirow, dicol,
                                                             dcv,   lddcv,
                                                             dv,    lddv);

  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}


/* 1D (2^k)+1 to original */
__global__ void
_pow2p1_to_org(int nrow, int nr, 
               int * irow,   
               double * dcv, double * dv) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int x = x0; x < nr; x += blockDim.x * gridDim.x) {
     dv[irow[x]] = dcv[x];
  }
}

mgard_cuda_ret 
pow2p1_to_org(int nrow, int nr,      
              int * dirow,  
              double * dcv, double * dv) {

  int B = 16;
  int total_thread_y = 1;
  int total_thread_x = nr;
  int tby = 1;
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "_copy_level_l_cuda" << std::endl;
  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _pow2p1_to_org<<<blockPerGrid, threadsPerBlock>>>(nrow, nr,   
                                                    dirow, 
                                                    dcv, dv);

  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}

/* 2D (2^k)+1 to compact */
__global__ void 
_pow2p1_to_cpt(int nrow,           int ncol, 
               int row_stride,      int col_stride,
               double * dv,         int lddv, 
               double * dcv,        int lddcv) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  for (int y = y0; y * row_stride < nrow; y += blockDim.y * gridDim.y) {
    for (int x = x0; x * col_stride < ncol; x += blockDim.x * gridDim.x) {
      int x_strided = x * col_stride;
      int y_strided = y * row_stride;
      dcv[get_idx(lddcv, y, x)] = dv[get_idx(lddv, y_strided, x_strided)];

    }
  }
}

mgard_cuda_ret 
pow2p1_to_cpt(int nrow,      int ncol, 
              int row_stride, int col_stride,
              double * dv,    int lddv, 
              double * dcv,   int lddcv) {
  int B = 16;
  int total_thread_y = ceil((float)nrow/row_stride);
  int total_thread_x = ceil((float)ncol/col_stride);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  //std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
  //std::cout << "grid: " << gridx << ", " << gridy <<std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _pow2p1_to_cpt<<<blockPerGrid, threadsPerBlock>>>(nrow,      ncol,
                                                    row_stride, col_stride,
                                                    dv,         lddv, 
                                                    dcv,        lddcv);
  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}


/* 1D (2^k)+1 to compact */
__global__ void 
_pow2p1_to_cpt(int nrow, int row_stride,     
               double * dv,double * dcv) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int x = x0; x * row_stride < nrow; x += blockDim.x * gridDim.x) {
    int x_strided = x * row_stride;
    dcv[x] = dv[x_strided];
  }
}

mgard_cuda_ret 
pow2p1_to_cpt(int nrow,  int row_stride, 
              double * dv, double * dcv) {
  int B = 16;
  int total_thread_y = 1;
  int total_thread_x = ceil((float)nrow/row_stride);
  int tby = 1;
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  //std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
  //std::cout << "grid: " << gridx << ", " << gridy <<std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _pow2p1_to_cpt<<<blockPerGrid, threadsPerBlock>>>(nrow, row_stride, 
                                                    dv, dcv);
  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}

/* 2D compact to (2^k)+1*/
__global__ void 
_cpt_to_pow2p1(int nrow,     int ncol,
              int row_stride, int col_stride,
              double * dcv,   int lddcv,
              double * dv,    int lddv) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  for (int y = y0; y * row_stride < nrow; y += blockDim.y * gridDim.y) {
    for (int x = x0; x * col_stride < ncol; x += blockDim.x * gridDim.x) {
      int x_strided = x * col_stride;
      int y_strided = y * row_stride;
      dv[get_idx(lddv, y_strided, x_strided)] = dcv[get_idx(lddcv, y, x)];
    }
  }
}


mgard_cuda_ret
cpt_to_pow2p1(int nrow, int ncol, 
              int row_stride, int col_stride, 
              double * dcv, int lddcv,
              double * dv, int lddv) {
  int B = 16;
  int total_thread_x = ceil((float)nrow/row_stride);
  int total_thread_y = ceil((float)ncol/col_stride);
  int tbx = min(B, total_thread_x);
  int tby = min(B, total_thread_y);
  int gridx = ceil((float)total_thread_x/tbx);
  int gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  //std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
  //std::cout << "grid: " << gridx << ", " << gridy <<std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _cpt_to_pow2p1<<<blockPerGrid, threadsPerBlock>>>(nrow, ncol,
                                                    row_stride, col_stride, 
                                                    dcv, lddcv,
                                                    dv, lddv);
  
  gpuErrchk(cudaGetLastError ());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return mgard_cuda_ret(0, milliseconds/1000.0);

}


/* 1D compact to (2^k)+1*/
__global__ void 
_cpt_to_pow2p1(int nrow, int row_stride, 
              double * dcv, double * dv) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int x = x0; x * row_stride < nrow; x += blockDim.x * gridDim.x) {
    int x_strided = x * row_stride;
    dv[x_strided] = dcv[x];
  }
}



mgard_cuda_ret
cpt_to_pow2p1(int nrow, int row_stride, 
              double * dcv, double * dv) {
    int B = 16;
    int total_thread_x = ceil((float)nrow/row_stride);
    int total_thread_y = 1;
    int tbx = min(B, total_thread_x);
    int tby = 1;
    int gridx = ceil((float)total_thread_x/tbx);
    int gridy = ceil((float)total_thread_y/tby);
    dim3 threadsPerBlock(tbx, tby);
    dim3 blockPerGrid(gridx, gridy);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    _cpt_to_pow2p1<<<blockPerGrid, threadsPerBlock>>>(nrow, row_stride, 
                                                      dcv,  dv);
    
    gpuErrchk(cudaGetLastError ());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return mgard_cuda_ret(0, milliseconds/1000.0);
}

__device__ double
_dist(double * dcoord, int x, int y) {
  return dcoord[y] - dcoord[x];
}


__global__ void
_calc_cpt_dist(int n, int stride,
               double * dcoord, double * ddist) {
  extern __shared__ double sm[]; //size = blockDim.x + 1

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int x0_sm = threadIdx.x;
  double dist;
  for (int x = x0; x * stride < n - 1; x += blockDim.x * gridDim.x) {
    // Load coordinates
    sm[x0_sm] = dcoord[x * stride];
    // printf("block %d thread %d load[%d] %f\n", blockIdx.x, threadIdx.x, x, dcoord[x * stride]);
    if (x0_sm == 0){
      sm[blockDim.x] = dcoord[(x + blockDim.x) * stride];
    }
    __syncthreads();

    // Compute distance
    dist = _dist(sm, x0_sm, x0_sm+1);
    __syncthreads();
    ddist[x] = dist;
    __syncthreads();
  }
}


mgard_cuda_ret
calc_cpt_dist(int nrow, int row_stride, 
              double * dcoord, double * ddist) {
    int B = 16;
    int total_thread_x = ceil((float)nrow/row_stride) - 1;
    int total_thread_y = 1;
    int tbx = min(B, total_thread_x);
    int tby = 1;
    int gridx = ceil((float)total_thread_x/tbx);
    int gridy = ceil((float)total_thread_y/tby);
    dim3 threadsPerBlock(tbx, tby);
    dim3 blockPerGrid(gridx, gridy);
    size_t sm_size = (tbx + 1) * sizeof(double);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    _calc_cpt_dist<<<blockPerGrid, threadsPerBlock, sm_size>>>(nrow, row_stride, 
                                                               dcoord,  ddist);
    
    gpuErrchk(cudaGetLastError ());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return mgard_cuda_ret(0, milliseconds/1000.0);
}