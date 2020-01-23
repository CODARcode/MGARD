#include <iomanip> 
#include <iostream>
#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"
#include "mgard_cuda.h"
#include "mgard_cuda_compacted.h"

__global__ void 
_original_to_compacted_cuda(int nrow,       int ncol, 
                           int row_stride, int col_stride,
                           double * dv,    int lddv, 
                           double * dwork, int lddwork) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    int y0 = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("x = %d, y = %d, stride = %d \n", x,y,stride);
    for (int y = y0; y * row_stride < nrow; y += blockDim.y * gridDim.y) {
        for (int x = x0; x * col_stride < ncol; x += blockDim.x * gridDim.x) {
            int x_strided = x * row_stride;
            int y_strided = y * col_stride;
            //printf("thread1(%d, %d): v(%d, %d)\n", x0, y0, x_strided, y_strided);
            dwork[get_idx(lddwork, y, x)] = dv[get_idx(lddv, y_strided, x_strided)];
            //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
            //y += blockDim.y * gridDim.y * stride;
        }
        //x += blockDim.x * gridDim.x * stride;
    }
}

mgard_cuda_ret 
original_to_compacted_cuda(int nrow,       int ncol, 
                          int row_stride, int col_stride,
                          double * dv,     int lddv, 
                          double * dwork,  int lddwork) {
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

    _original_to_compacted_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow, ncol,
                                                                  row_stride, col_stride,
                                                                  dv,    lddv, 
                                                                  dwork, lddwork);
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
_compacted_to_original_cuda(int nrow, int ncol,
                          int row_stride, int col_stride,
                          double * dv, int lddv, 
                          double * dwork, int lddwork) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    int y0 = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("x = %d, y = %d, stride = %d \n", x,y,stride);
    for (int y = y0; y * row_stride < nrow; y += blockDim.y * gridDim.y) {
        for (int x = x0; x * col_stride < ncol; x += blockDim.x * gridDim.x) {
            int x_strided = x * row_stride;
            int y_strided = y * col_stride;
            //printf("thread2(%d, %d): v(%d, %d)\n", x0, y0, x_strided, y_strided);
            dv[get_idx(lddv, y_strided, x_strided)] = dwork[get_idx(lddwork, y, x)];
            //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
            //y += blockDim.y * gridDim.y * stride;
        }
        //x += blockDim.x * gridDim.x * stride;
    }
}


mgard_cuda_ret
compacted_to_original_cuda(int nrow, int ncol, 
                              int row_stride, int col_stride, 
                              double * dv, int lddv, 
                              double * dwork, int lddwork) {
    

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

    _compacted_to_original_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow, ncol,
                                                                  row_stride, col_stride, 
                                                                  dv, lddv, 
                                                                  dwork, lddwork);
    
    gpuErrchk(cudaGetLastError ());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return mgard_cuda_ret(0, milliseconds/1000.0);

}


