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
    for (int x = x0; x * row_stride < nrow; x += blockDim.x * gridDim.x) {
        for (int y = y0; y * col_stride < ncol; y += blockDim.y * gridDim.y) {
            int x_strided = x * row_stride;
            int y_strided = y * col_stride;
            //printf("thread1(%d, %d): v(%d, %d)\n", x0, y0, x_strided, y_strided);
            dwork[get_idx(lddwork, x, y)] = dv[get_idx(lddv, x_strided, y_strided)];
            //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
            //y += blockDim.y * gridDim.y * stride;
        }
        //x += blockDim.x * gridDim.x * stride;
    }
}

void 
original_to_compacted_cuda(int nrow,       int ncol, 
                          int row_stride, int col_stride,
                          double * v,     int ldv, 
                          double * work,  int ldwork) {
    double * dv;
    int lddv;
    double * dwork;
    int lddwork;

    //int stride = pow (2, l); // current stride
    int nrow_level = ceil((double)nrow/row_stride);
    int ncol_level = ceil((double)ncol/col_stride);

#if CPU == 1
    size_t dv_pitch;
    cudaMallocPitch(&dv, &dv_pitch, ncol * sizeof(double), nrow);
    lddv = dv_pitch / sizeof(double);
    
    cudaMemcpy2D(dv, lddv * sizeof(double), 
                 v,  ldv  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);

    size_t dwork_pitch;
    cudaMallocPitch(&dwork, &dwork_pitch, 
                    ncol_level * sizeof(double), nrow_level);

    lddwork = dwork_pitch / sizeof(double);
    cudaMemcpy2D(dwork, lddwork * sizeof(double), 
                 work,  ldwork  * sizeof(double), 
                 ncol_level * sizeof(double), nrow_level, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dv = v;
    lddv = ldv;
    dwork = work;
    lddwork = ldwork;
#endif

    
    int B = 16;
    int total_thread_x = nrow/row_stride;
    int total_thread_y = ncol/col_stride;
    int tbx = min(B, total_thread_x);
    int tby = min(B, total_thread_y);
    int gridx = ceil(total_thread_x/tbx);
    int gridy = ceil(total_thread_y/tby);
    dim3 threadsPerBlock(tbx, tby);
    dim3 blockPerGrid(gridx, gridy);

    //std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
    //std::cout << "grid: " << gridx << ", " << gridy <<std::endl;

    _original_to_compacted_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow, ncol,
                                                                  row_stride, col_stride,
                                                                  dv, lddv, 
                                                                  dwork, lddwork);
    if (cudaSuccess != cudaGetLastError ()) 
    {
        std::cout << "CUDA KERNEL ERROR (_original_to_compacted_cuda)" << std::endl;
    }
#if CPU == 1
    cudaMemcpy2D(work,  ldwork  * sizeof(double), 
                 dwork, lddwork * sizeof(double),
                 ncol_level * sizeof(double), nrow_level, 
                 cudaMemcpyDeviceToHost);

    cudaMemcpy2D(v,  ldv  * sizeof(double), 
                 dv, lddv * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
#endif

    //cudaMemcpy(work, dev_work, nrow * ncol * sizeof(double), cudaMemcpyDeviceToHost);

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
    for (int x = x0; x * row_stride < nrow; x += blockDim.x * gridDim.x) {
        for (int y = y0; y * col_stride < ncol; y += blockDim.y * gridDim.y) {
            int x_strided = x * row_stride;
            int y_strided = y * col_stride;
            //printf("thread2(%d, %d): v(%d, %d)\n", x0, y0, x_strided, y_strided);
            dv[get_idx(lddv, x_strided, y_strided)] = dwork[get_idx(lddwork, x, y)];
            //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
            //y += blockDim.y * gridDim.y * stride;
        }
        //x += blockDim.x * gridDim.x * stride;
    }
}


void
compacted_to_original_cuda(int nrow, int ncol, 
                              int row_stride, int col_stride, 
                              double * v, int ldv, 
                              double * work, int ldwork) {
    double * dv;
    int lddv;
    double * dwork;
    int lddwork;

    //int stride = pow (2, l); // current stride
    int nrow_level = ceil((double)nrow/row_stride);
    int ncol_level = ceil((double)ncol/col_stride);

#if CPU == 1
    size_t dv_pitch;
    cudaMallocPitch(&dv, &dv_pitch, ncol * sizeof(double), nrow);
    lddv = dv_pitch / sizeof(double);
    
    cudaMemcpy2D(dv, lddv * sizeof(double), 
                 v,  ldv  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);

    size_t dwork_pitch;
    cudaMallocPitch(&dwork, &dwork_pitch, 
                    ncol_level * sizeof(double), nrow_level);

    lddwork = dwork_pitch / sizeof(double);

    cudaMemcpy2D(dwork, lddwork * sizeof(double), 
                 work,  ldwork  * sizeof(double), 
                 ncol_level * sizeof(double), nrow_level, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dv = v;
    lddv = ldv;
    dwork = work;
    lddwork = ldwork;
#endif

    
    int B = 16;
    int total_thread_x = nrow/row_stride;
    int total_thread_y = ncol/col_stride;
    int tbx = min(B, total_thread_x);
    int tby = min(B, total_thread_y);
    int gridx = ceil(total_thread_x/tbx);
    int gridy = ceil(total_thread_y/tby);
    dim3 threadsPerBlock(tbx, tby);
    dim3 blockPerGrid(gridx, gridy);

    //std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
    //std::cout << "grid: " << gridx << ", " << gridy <<std::endl;

    _compacted_to_original_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow, ncol,
                                                                  row_stride, col_stride, 
                                                                  dv, lddv, 
                                                                  dwork, lddwork);
    if (cudaSuccess != cudaGetLastError ()) 
    {
        std::cout << "CUDA KERNEL ERROR (_copy_level_cuda)" << std::endl;
    }
#if CPU == 1
    cudaMemcpy2D(work,  ldwork  * sizeof(double), 
                 dwork, lddwork * sizeof(double),
                 ncol_level * sizeof(double), nrow_level, 
                 cudaMemcpyDeviceToHost);

    cudaMemcpy2D(v,  ldv  * sizeof(double), 
                 dv, lddv * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
#endif

    //cudaMemcpy(work, dev_work, nrow * ncol * sizeof(double), cudaMemcpyDeviceToHost);

}

__global__ void 
_fused_row_cuda(int nrow,       int ncol, 
               int row_stride, int col_stride,
               int k1_row_stride, int k1_col_stride,
               int k2_row_stride, int K2_col_stride,
               double * dv,    int lddv) {
    //int stride = pow (2, l); // current stride
    
    extern __shared__ double smdv[];
    double * curr_smdv = smdv;
    double * next_smdv = smdv + blockDim.x * blockDim.x;

     // number of rows and columns each tb works on for each iter.
    int work_nrow = ceil((double)nrow/row_stride);
    int work_ncol = ceil((double)ncol/col_stride);

    int smdv_nrow = min(work_nrow - blockIdx.x * blockDim.x, blockDim.x);
    int smdv_ncol = blockDim.x;

    int ldsmdv = smdv_ncol;
    int ntb = blockDim.x;
    int idx = threadIdx.x;

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // for k1
    int assign_num = 0;

    // for k2
    register double temp1, temp2;
    register double fac = 0.5;

    // adjust to current row block
    dv = dv + get_idx(lddv, blockIdx.x * smdv_ncol * row_stride, 0);

    //printf("tb %d : work_nrow = %d, work_ncol = %d, smdv_nrow = %d, smdv_ncol = %d, %f\n", blockIdx.x, work_nrow, work_ncol, smdv_nrow, smdv_ncol, dv[0]);


    for (int i = 0; i < work_ncol; i += smdv_ncol) {      
      int curr_smdv_ncol = min(work_ncol - i, smdv_ncol);
      if (idx == 0) {
          //printf("tb %d curr_smdv_ncol = %d : %f\n", blockIdx.x, curr_smdv_ncol, dv[0]);
      }
      if (idx < curr_smdv_ncol) {
        for (int j = 0; j < smdv_nrow; j++){
          if (idx == 0) {
            printf("tb %d (j = %d) : %f\n", blockIdx.x, j, dv[get_idx(lddv, j * row_stride, idx*col_stride)]);
          }
          curr_smdv[get_idx(ldsmdv, j, idx)] = dv[get_idx(lddv, j * row_stride, idx*col_stride)];
        }
      }
      
      if (idx < smdv_nrow) {
        int global_row = global_idx * row_stride;

        // k1
        if (global_row % k1_row_stride == 0) {
          int local_stride = k1_col_stride/col_stride;
          for (int j = 0; j < curr_smdv_ncol; j += local_stride) {
            //int global_col = i * col_stride + j * col_stride;
            //if (global_col % k1_col_stride == 0) {
            curr_smdv[get_idx(ldsmdv, idx, j)] = assign_num;
            //}
          }
        }
        // k2
        // if (global_row % k2_row_stride == 0) {
        //   int local_stride = k2_col_stride/col_stride
          
        //   if(i == 0) {
        //     temp1 = curr_smdv[get_idx(ldsmdv, idx, 0)]
        //     curr_smdv[get_idx(ldsmdv, idx, 0)] = fac*(2.0*temp1+curr_smdv[get_idx(ldsmdv, idx, local_stride)]);
        //   }

        //   for (int j = local_stride; j < curr_smdv_ncol-local_stride; j += local_stride) {
            
        //     temp2 = curr_smdv[get_idx(ldsmdv, idx, j)];
        //     curr_smdv[get_idx(ldsmdv, idx, j)] = fac * (temp1+4*temp2+curr_smdv[get_idx(ldsmdv, idx, j+local_stride)]);
        //     temp1 = temp2;

        //     //int global_col = i * col_stride + j * col_stride;
        //     //if (global_col % k2_col_stride == 0) {
        //     //if (global_col == 0) {
              
        //     //}

        //     //}
        //   }

        // }


      }

      if (idx < curr_smdv_ncol) {
        for (int j = 0; j < smdv_nrow; j++){
          // if (idx == 0) {
          //   printf("tb %d (j = %d) : %f\n", blockIdx.x, j, dv[get_idx(lddv, j * row_stride, idx*col_stride)]);
          // }
          dv[get_idx(lddv, j * row_stride, idx*col_stride)] = curr_smdv[get_idx(ldsmdv, j, idx)];
        }
      }


      dv = dv + get_idx(lddv, 0, smdv_ncol * col_stride);
    }

  }


void fused_row_cuda(int nrow,       int ncol, 
                   int row_stride, int col_stride,
                   int k1_row_stride, int k1_col_stride,
                   int k2_row_stride, int k2_col_stride,
                   double * v,    int ldv) {

  double * dv;
  int lddv;
#if CPU == 1
    size_t dv_pitch;
    cudaMallocPitch(&dv, &dv_pitch, ncol * sizeof(double), nrow);
    lddv = dv_pitch / sizeof(double);
    
    cudaMemcpy2D(dv, lddv * sizeof(double), 
                 v,  ldv  * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyHostToDevice);
#endif

#if GPU == 1
    dv = v;
    lddv = ldv;
#endif

    int B = 4;
    int total_thread_x = ceil((double)nrow/row_stride);
    int tbx = B;
    int gridx = ceil((double)total_thread_x/tbx);
    dim3 threadsPerBlock(tbx, 1);
    dim3 blockPerGrid(gridx, 1);

    std::cout << "thread block: " << tbx <<std::endl;
    std::cout << "grid: " << gridx << std::endl;

    long sm_size = 2 * tbx * tbx * sizeof(double);

    _fused_row_cuda<<<blockPerGrid, threadsPerBlock, sm_size>>>(nrow,          ncol, 
                                                      row_stride,    col_stride,
                                                      k1_row_stride, k1_col_stride,
                                                      k2_row_stride, k2_col_stride,
                                                      dv,            lddv);

    gpuErrchk(cudaGetLastError ()); 
#if CPU == 1
    cudaMemcpy2D(v,  ldv  * sizeof(double), 
                 dv, lddv * sizeof(double), 
                 ncol * sizeof(double), nrow, 
                 cudaMemcpyDeviceToHost);
#endif
    
}
