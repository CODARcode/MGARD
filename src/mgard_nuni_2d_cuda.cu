#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>

namespace mgard_2d {
namespace mgard_common {

__host__ __device__
int get_index_cuda(const int ncol, const int i, const int j) {
  return ncol * i + j;
}

__host__ __device__ double 
interp_2d_cuda(double q11, double q12, double q21, double q22,
                        double x1, double x2, double y1, double y2, double x,
                        double y) {
  double x2x1, y2y1, x2x, y2y, yy1, xx1;
  x2x1 = x2 - x1;
  y2y1 = y2 - y1;
  x2x = x2 - x;
  y2y = y2 - y;
  yy1 = y - y1;
  xx1 = x - x1;
  return 1.0 / (x2x1 * y2y1) *
         (q11 * x2x * y2y + q21 * xx1 * y2y + q12 * x2x * yy1 +
          q22 * xx1 * yy1);
}

__host__ __device__ double 
get_h_cuda(const double * coords, int i, int stride) {
  return (i + stride - i);
}

__host__ __device__
double get_dist_cuda(const double * coords, int i, int j) {
  return (j - i);
}

__device__ double 
_get_dist(double * coords, int i, int j) {
  return coords[j] - coords[i];
}



} //end namespace mgard_common


namespace mgard_cannon {

__global__ void 
_copy_level_cuda(int nrow,       int ncol,
                int row_stride, int col_stride, 
                double * dv,    int lddv, 
                double * dwork, int lddwork) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * row_stride;
    int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * col_stride;
    //printf("x = %d, y = %d, stride = %d \n", x,y,stride);
    for (int x = x0; x < nrow; x += blockDim.x * gridDim.x * row_stride) {
        for (int y = y0; y < ncol; y += blockDim.y * gridDim.y * col_stride) {
            
            dwork[get_idx(lddv, x, y)] = dv[get_idx(lddwork, x, y)];
            //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
            //y += blockDim.y * gridDim.y * stride;
        }
        //x += blockDim.x * gridDim.x * stride;
    }
}



mgard_cuda_ret 
copy_level_cuda(int nrow,       int ncol, 
                int row_stride, int col_stride,
                double * dv,    int lddv,
                double * dwork, int lddwork) {

  int B = 16;
  int total_thread_y = ceil((float)nrow/row_stride);
  int total_thread_x = ceil((float)ncol/col_stride);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tbx);
  int gridx = ceil((float)total_thread_x/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  _copy_level_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                                                      row_stride, col_stride, 
                                                      dv,         lddv, 
                                                      dwork,      lddwork);
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
_mass_matrix_multiply_row_cuda(int nrow,       int ncol, 
                               int row_stride, int col_stride,
                               double * dv,    int lddv,
                               double * dcoords_x) {
  //int stride = pow (2, l); // current stride

  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  //int y = threadIdx.y * stride;
  for (int x = idx; x < nrow; x += (blockDim.x * gridDim.x) * row_stride) {
    //printf("thread working on %d \n", x);
    double * vec = dv + x * lddv;
    register double temp1, temp2;
    temp1 = vec[0];
    // printf("thread %d working on %f\n", idx, temp1);
    vec[0] = 2.0 * mgard_common::get_h_cuda(dcoords_x, 0, col_stride) * temp1 + 
                   mgard_common::get_h_cuda(dcoords_x, 0, col_stride) * vec[col_stride];
    for (int i = col_stride; i < ncol - col_stride; i += col_stride) {
        temp2 = vec[i];
        vec[i] = mgard_common::get_h_cuda(dcoords_x, i - col_stride, col_stride) * temp1 + 
                 2 * 
                    (mgard_common::get_h_cuda(dcoords_x, i - col_stride, col_stride) +
                     mgard_common::get_h_cuda(dcoords_x, i,              col_stride)) *
                 temp2 + 
                 mgard_common::get_h_cuda(dcoords_x, i, col_stride) * vec[i + col_stride];
        temp1 = temp2;
    }
    vec[ncol-1] = mgard_common::get_h_cuda(dcoords_x, ncol - col_stride - 1, col_stride) * temp1 +
              2 * mgard_common::get_h_cuda(dcoords_x, ncol - col_stride - 1, col_stride) * vec[ncol-1];
  }
}

mgard_cuda_ret 
mass_matrix_multiply_row_cuda(int nrow,       int ncol, 
                              int row_stride, int col_stride,
                              double * dv,    int lddv,
                              double * dcoords_x) {
  int B = 16;
  int total_thread = ceil((float)nrow/row_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  // std::cout << "thread block: " << tb << std::endl;
  // std::cout << "grid: " << grid << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _mass_matrix_multiply_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                                                                    row_stride, col_stride,
                                                                    dv,         lddv,
                                                                    dcoords_x);
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
_mass_matrix_multiply_col_cuda(int nrow,       int ncol, 
                              int row_stride, int col_stride,
                              double * dv,    int lddv,
                              double * dcoords_y) {
  //int stride = pow (2, l); // current stride

  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  //int y = threadIdx.y * stride;
  for (int x = idx; x < ncol; x += (blockDim.x * gridDim.x) * col_stride) {
    //printf("thread working on %d \n", x);
    double * vec = dv + x;
    register double temp1, temp2;
    temp1 = vec[0];
    //printf("thread %d working on %f\n", idx, temp1);
    vec[0] = 2.0 * mgard_common::get_h_cuda(dcoords_y, 0, row_stride) * temp1 + 
                   mgard_common::get_h_cuda(dcoords_y, 0, row_stride) * vec[row_stride * lddv];
    for (int i = row_stride; i < nrow - row_stride; i += row_stride) {
      temp2 = vec[i * lddv];
      vec[i * lddv] = mgard_common::get_h_cuda(dcoords_y, i - row_stride, row_stride) * temp1 + 
               2 * 
                  (mgard_common::get_h_cuda(dcoords_y, i - row_stride, row_stride) +
                   mgard_common::get_h_cuda(dcoords_y, i,              row_stride)) *
               temp2 + 
               mgard_common::get_h_cuda(dcoords_y, i, row_stride) * vec[(i + row_stride)  * lddv];
      temp1 = temp2;
    }
    vec[(nrow-1) * lddv] = mgard_common::get_h_cuda(dcoords_y, nrow - row_stride - 1, row_stride) * temp1 +
              2 * mgard_common::get_h_cuda(dcoords_y, nrow - row_stride - 1, row_stride) * vec[(nrow-1) * lddv];
  }
}


mgard_cuda_ret 
mass_matrix_multiply_col_cuda(int nrow,       int ncol, 
                              int row_stride, int col_stride,
                              double * dv,    int lddv,
                              double * dcoords_y) {
  int B = 16;
  int total_thread = ceil((float)ncol/col_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  // std::cout << "thread block: " << tb << std::endl;
  // std::cout << "grid: " << grid << std::endl;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _mass_matrix_multiply_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                                                                    row_stride, col_stride,
                                                                    dv,         lddv,
                                                                    dcoords_y);
  gpuErrchk(cudaGetLastError ()); 
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}

void mass_matrix_multiply_cuda(const int l, std::vector<double> &v,
                          const std::vector<double> &coords) {
  int stride = std::pow(2, l); // current stride
  int n = v.size();
  double temp1, temp2;

  // Mass matrix times nodal value-vec
  temp1 = v.front(); // save u(0) for later use
  //printf("working on %f\n", temp1);
  v.front() = 2.0 * mgard_common::get_h_cuda(coords.data(), 0, stride) * temp1 +
              mgard_common::get_h_cuda(coords.data(), 0, stride) * v[stride];
  for (int i = stride; i <= n - 1 - stride; i += stride) {
    temp2 = v[i];
    v[i] = mgard_common::get_h_cuda(coords.data(), i - stride, stride) * temp1 +
           2 *
               (mgard_common::get_h_cuda(coords.data(), i - stride, stride) +
                mgard_common::get_h_cuda(coords.data(), i, stride)) *
               temp2 +
           mgard_common::get_h_cuda(coords.data(), i, stride) * v[i + stride];
    temp1 = temp2; // save u(n) for later use
  }
  v[n - 1] = mgard_common::get_h_cuda(coords.data(), n - stride - 1, stride) * temp1 +
             2 * mgard_common::get_h_cuda(coords.data(), n - stride - 1, stride) * v[n - 1];
}


__global__ void 
_subtract_level_cuda(int nrow,       int ncol, 
                      int row_stride, int col_stride,
                      double * dv,    int lddv, 
                      double * dwork, int lddwork) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int idx_x = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;
    int idx_y = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
    //printf("x = %d, y = %d, stride = %d \n", x,y,stride);
    for (int y = idx_y; y < nrow; y += blockDim.y * gridDim.y * row_stride) {
      for (int x = idx_x; x < ncol; x += blockDim.x * gridDim.x * col_stride) {
        dv[get_idx(lddv, y, x)] -= dwork[get_idx(lddwork, y, x)];
        //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
        //y += blockDim.y * gridDim.y * stride;
      }
        //x += blockDim.x * gridDim.x * stride;
    }
}

mgard_cuda_ret 
subtract_level_cuda(int nrow,       int ncol, 
                    int row_stride, int col_stride,
                    double * dv,    int lddv, 
                    double * dwork, int lddwork) {
  int B = 16;
  int total_thread_x = ncol/col_stride;
  int total_thread_y = nrow/row_stride;
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

  _subtract_level_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                           row_stride, col_stride, 
                                                           dv,         lddv,
                                                           dwork,      lddwork);


  gpuErrchk(cudaGetLastError ()); 

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}





} //end namespace mgard_cannon

namespace mgard_gen {

__host__ __device__ int
get_lindex_cuda(const int n, const int no, const int i) {
  // no: original number of points
  // n : number of points at next coarser level (L-1) with  2^k+1 nodes
  int lindex;
  //    return floor((no-2)/(n-2)*i);
  if (i != n - 1) {
    lindex = floor(((double)no - 2.0) / ((double)n - 2.0) * i);
  } else if (i == n - 1) {
    lindex = no - 1;
  }

  return lindex;
}

__host__ __device__ double 
get_h_l_cuda(const double * coords, const int n,
             const int no, int i, int stride) {

  //    return (*get_ref_cuda(coords, n, no, i+stride) - *get_ref_cuda(coords, n, no, i));
  return (get_lindex_cuda(n, no, i + stride) - get_lindex_cuda(n, no, i));
}


__device__ double 
_get_h_l(const double * coords, int i, int stride) {

  //    return (*get_ref_cuda(coords, n, no, i+stride) - *get_ref_cuda(coords, n, no, i));
  return coords[i + stride] - coords[i];
}

__host__ __device__ double *
get_ref_cuda(double * v, const int n, const int no,
                       const int i) // return reference to logical element
{
  // no: original number of points
  // n : number of points at next coarser level (L-1) with  2^k+1 nodes
  // may not work for the last element!
  double *ref;
  if (i != n - 1) {
    ref = &v[(int)floor(((double)no - 2.0) / ((double)n - 2.0) * i)];
  } else if (i == n - 1) {
    ref = &v[no - 1];
  }
  return ref;
  //    return &v[floor(((no-2)/(n-2))*i ) ];
}

__host__ __device__ double *
get_ref_row_cuda(double * v, int ldv, const int n, const int no,
                       const int i) // return reference to logical element
{
  // no: original number of points
  // n : number of points at next coarser level (L-1) with  2^k+1 nodes
  // may not work for the last element!
  double *ref;
  if (i != n - 1) {
    ref = &v[(int)floor(((double)no - 2.0) / ((double)n - 2.0) * i)];
  } else if (i == n - 1) {
    ref = &v[no - 1];
  }
  return ref;
  //    return &v[floor(((no-2)/(n-2))*i ) ];
}

__host__ __device__ double *
get_ref_col_cuda(double * v, int ldv, const int n, const int no,
                       const int i) // return reference to logical element
{
  // no: original number of points
  // n : number of points at next coarser level (L-1) with  2^k+1 nodes
  // may not work for the last element!
  double *ref;
  if (i != n - 1) {
    ref = &v[((int)floor(((double)no - 2.0) / ((double)n - 2.0) * i)) * ldv];
  } else if (i == n - 1) {
    ref = &v[(no - 1) * ldv];
  }
  return ref;
  //    return &v[floor(((no-2)/(n-2))*i ) ];
}


__global__ void 
_pi_Ql_first_row_cuda(const int nrow,    const int ncol,
                      const int nr,      const int nc,
                      int * irow,        int * icolP,
                      double * dv,       int lddv,
                      double * coords_x, double * coords_y) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;

   
   //if (x < ncol-nc && y < nr) {
    for (int y = y0; y < nr; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < ncol-nc; x += blockDim.x * gridDim.x) {

      int r = irow[y];
      int c = icolP[x];
      //dv[mgard_common::get_index_cuda(lddv, r, c    )] ++;
      //printf ("thread (%d, %d) working on (%d, %d): %f\n", y, x, r, c, dv[mgard_common::get_index_cuda(lddv, r, c    )]);
      register double center = dv[mgard_common::get_index_cuda(lddv, r, c    )];
      register double left   = dv[mgard_common::get_index_cuda(lddv, r, c - 1)];
      register double right  = dv[mgard_common::get_index_cuda(lddv, r, c + 1)];
      register double h1     = mgard_common::_get_dist(coords_x, c - 1, c    );
      register double h2     = mgard_common::_get_dist(coords_x, c,     c + 1);
      // printf ("thread (%d, %d) working on (%d, %d): %f, left=%f, right=%f\n", y, x, r, c, dv[mgard_common::get_index_cuda(lddv, r, c    )], left, right);


      center -= (h2 * left + h1 * right) / (h1 + h2);
      //center -= (left + right)/2;
      //center -= 1;

      dv[mgard_common::get_index_cuda(lddv, r, c    )] = center;
    }
  }

}


__global__ void 
_pi_Ql_first_col_cuda(const int nrow,    const int ncol,
                      const int nr,      const int nc,
                      int * irowP,       int * icol,
                      double * dv,       int lddv,
                      double * coords_x, double * coords_y) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;

  //if (x < nc && y < nrow-nr) {
  for (int y = y0; y < nrow-nr; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x) {
      int r = irowP[y];
      int c = icol[x];
      //printf ("thread (%d, %d) working on (%d, %d): %f\n", y, x, r, c, dv[mgard_common::get_index_cuda(lddv, r, c    )]);
      register double center = dv[mgard_common::get_index_cuda(lddv, r,     c)];
      register double up   = dv[mgard_common::get_index_cuda(lddv,   r - 1, c)];
      register double down  = dv[mgard_common::get_index_cuda(lddv, r + 1, c)];
      register double h1     = mgard_common::_get_dist(coords_y, r - 1, r    );
      register double h2     = mgard_common::_get_dist(coords_y, r,     r + 1);

      center -= (h2 * up + h1 * down) / (h1 + h2);

      dv[mgard_common::get_index_cuda(lddv, r, c    )] = center;
    }
  }

}


__global__ void 
_pi_Ql_first_center_cuda(const int nrow,     const int ncol,
                         const int nr,       const int nc,
                         int * dirowP,       int * dicolP,
                         double * dv,        int lddv,
                         double * dcoords_x, double * dcoords_y) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;

  //if (x < ncol-nc && y < nrow-nr) {
  for (int y = y0; y < nrow-nr; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < ncol-nc; x += blockDim.x * gridDim.x) {
      int r = dirowP[y];
      int c = dicolP[x];
      //printf ("thread (%d, %d) working on (%d, %d): %f\n", y, x, r, c, dv[mgard_common::get_index_cuda(lddv, r, c    )]);
      register double center    = dv[mgard_common::get_index_cuda(lddv, r,     c    )];
      register double upleft    = dv[mgard_common::get_index_cuda(lddv, r - 1, c - 1)];
      register double upright   = dv[mgard_common::get_index_cuda(lddv, r - 1, c + 1)];
      register double downleft  = dv[mgard_common::get_index_cuda(lddv, r + 1, c - 1)];
      register double downright = dv[mgard_common::get_index_cuda(lddv, r + 1, c + 1)];


      register double x1 = 0.0;
      register double y1 = 0.0;

      register double x2 = mgard_common::_get_dist(dcoords_x, c - 1, c + 1);
      register double y2 = mgard_common::_get_dist(dcoords_y, r - 1, r + 1);

      register double x = mgard_common::_get_dist(dcoords_x, c, c + 1);
      register double y = mgard_common::_get_dist(dcoords_y, r, r + 1);

      double temp =
              mgard_common::interp_2d_cuda(upleft, downleft, upright, downright, x1, x2, y1, y2, x, y);

      center -= temp;

      dv[mgard_common::get_index_cuda(lddv, r, c    )] = center;
    }
  }

}



mgard_cuda_ret 
pi_Ql_first_cuda(const int nrow,     const int ncol,
                 const int nr,       const int nc,
                 int * dirow,        int * dicol,
                 int * dirowP,       int * dicolP,
                 double * dcoords_x, double * dcoords_y,
                 double * dv,        const int lddv) {  

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);


  int B = 16;
  int total_thread_x = ncol-nc;
  int total_thread_y = nr;
  int tbx = min(B, total_thread_x);
  int tby = min(B, total_thread_y);
  int gridx = ceil((float)total_thread_x/tbx);
  int gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "_pi_Ql_first_row_cuda " << std::endl;
  // std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
  // std::cout << "grid: " << gridx << ", " << gridy <<std::endl;
  _pi_Ql_first_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,      ncol,
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
  //   std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
  // std::cout << "grid: " << gridx << ", " << gridy <<std::endl;
  _pi_Ql_first_col_cuda<<<blockPerGrid2, threadsPerBlock2>>>(nrow,      ncol,
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
  _pi_Ql_first_center_cuda<<<blockPerGrid3, threadsPerBlock3>>>(nrow,      ncol,
                                                                nr,        nc,
                                                                dirowP,    dicolP,
                                                                dv,        lddv,
                                                                dcoords_x, dcoords_y);
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
_restriction_first_row_cuda(int nrow,       int ncol,
                            int row_stride, int * icolP, int nc,
                            double * dv,    int lddv,
                            double * dcoords_x) {
  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  //int y = threadIdx.y * stride;
  for (int x = idx; x < nrow; x += (blockDim.x * gridDim.x) * row_stride) {
    //printf("thread working on %d \n", x);
    double * vec = dv + x * lddv;
    for (int i = 0; i < ncol-nc; i++) {
      double h1 = mgard_common::get_h_cuda(dcoords_x, icolP[i] - 1, 1);
      double h2 = mgard_common::get_h_cuda(dcoords_x, icolP[i]    , 1);
      double hsum = h1 + h2;
      vec[icolP[i] - 1] += h2 * vec[icolP[i]] / hsum;
      vec[icolP[i] + 1] += h1 * vec[icolP[i]] / hsum;
    }

  }
}

mgard_cuda_ret 
restriction_first_row_cuda(int nrow,       int ncol, 
                           int row_stride, int * dicolP, int nc,
                           double * dv,    int lddv,
                           double * dcoords_x) {
  int B = 16;
  int total_thread = ceil((float)nrow/row_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  // std::cout << "thread block: " << tb << std::endl;
  // std::cout << "grid: " << grid << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _restriction_first_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                                                                 row_stride, dicolP, nc,
                                                                 dv,         lddv,
                                                                 dcoords_x);
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
_restriction_first_col_cuda(int nrow,       int ncol,
                            int * irowP,    int nr, int col_stride,
                            double * dv,    int lddv,
                            double * dcoords_y) {
  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  //int y = threadIdx.y * stride;
  for (int x = idx; x < ncol; x += (blockDim.x * gridDim.x) * col_stride) {
    //printf("thread working on %d \n", x);
    double * vec = dv + x;
    for (int i = 0; i < nrow-nr; i++) {
      double h1 = mgard_common::get_h_cuda(dcoords_y, irowP[i] - 1, 1);
      double h2 = mgard_common::get_h_cuda(dcoords_y, irowP[i]    , 1);
      double hsum = h1 + h2;
      vec[(irowP[i] - 1) * lddv] += h2 * vec[irowP[i] * lddv] / hsum;
      vec[(irowP[i] + 1) * lddv] += h1 * vec[irowP[i] * lddv] / hsum;
    }
  }
}

mgard_cuda_ret 
restriction_first_col_cuda(int nrow,       int ncol, 
                           int * dirowP, int nr, int col_stride,
                           double * dv,    int lddv,
                           double * dcoords_y) {
  int B = 16;
  int total_thread = ceil((float)ncol/col_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // std::cout << "thread block: " << tb << std::endl;
  // std::cout << "grid: " << grid << std::endl;

  _restriction_first_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                                                                 dirowP, nr, col_stride,
                                                                 dv,         lddv,
                                                                 dcoords_y);
  gpuErrchk(cudaGetLastError ()); 

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}




void restriction_first_cuda(std::vector<double> &v, std::vector<double> &coords,
                       int n, int no) {
  // calculate the result of restrictionion

  for (int i = 0; i < n - 1; ++i) // loop over the logical array
  {
    int i_logic = get_lindex_cuda(n, no, i);
    int i_logicP = get_lindex_cuda(n, no, i + 1);

    if (i_logicP != i_logic + 1) // next real memory location was jumped over,
                                 // so need to restriction
    {
      double h1 = mgard_common::get_h_cuda(coords.data(), i_logic, 1);
      double h2 = mgard_common::get_h_cuda(coords.data(), i_logic + 1, 1);
      double hsum = h1 + h2;
      // v[i_logic]  = 0.5*v[i_logic]  + 0.5*h2*v[i_logic+1]/hsum;
      // v[i_logicP] = 0.5*v[i_logicP] + 0.5*h1*v[i_logic+1]/hsum;
      v[i_logic] += h2 * v[i_logic + 1] / hsum;
      v[i_logicP] += h1 * v[i_logic + 1] / hsum;
    }
  }
}

__global__ void
_solve_tridiag_M_l_row_cuda(int nrow,        int ncol,
                             int nr,         int nc,
                             int row_stride, int col_stride,
                             int * dirow,    int * dicol, 
                             double * dv,    int lddv, 
                             double * dcoords_x) {
  int idx0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  //printf("thread %d, nr = %d\n", idx0, nr);
  double am, bm, h1, h2;
  double * coeff = new double[ncol];
  for (int idx = idx0; idx < nr; idx += (blockDim.x * gridDim.x) * row_stride) {
    //printf("thread %d, nr = %d, idx = %d\n", idx0, nr, idx);
    int r = dirow[idx];
    //printf("thread %d working on row %d \n", idx0, r);
    double * vec = dv + r * lddv;
    am = 2.0 * mgard_common::_get_dist(dcoords_x, dicol[0], dicol[col_stride]); //dicol[col_stride] - dicol[0]
    bm = mgard_common::_get_dist(dcoords_x, dicol[0], dicol[col_stride]) / am; //dicol[col_stride] - dicol[0]

    int counter = 1;
    coeff[0] = am;
    for (int i = col_stride; i < nc - 1; i += col_stride) {

      h1 = mgard_common::_get_dist(dcoords_x, dicol[i - col_stride], dicol[i]);
      h2 = mgard_common::_get_dist(dcoords_x, dicol[i], dicol[i + col_stride]);


      // h1 = dicol[i] - dicol[i - col_stride];
      // h2 = dicol[i + col_stride] - dicol[i];

      vec[dicol[i]] -= vec[dicol[i - col_stride]] * bm;

      am = 2.0 * (h1 + h2) - bm * h1;
      bm = h2 / am;

      coeff[counter] = am;
      ++counter;
    }
    h2 = mgard_common::_get_dist(dcoords_x, dicol[nc - 1 - col_stride], dicol[nc - 1]);
    // h2 = dicol[nc - 1] - dicol[nc - 1 - col_stride];


    am = 2.0 * h2 - bm * h2;

    vec[dicol[nc - 1]] -= vec[dicol[nc - 1 - col_stride]] * bm;
    coeff[counter] = am;

    vec[dicol[nc - 1]] /= am;
    --counter;

    for (int i = nc - 1 - col_stride; i >= 0; i -= col_stride) {
      h2 = mgard_common::_get_dist(dcoords_x, dicol[i], dicol[i + col_stride]);
      // h2 = dicol[i + col_stride] - dicol[i];
      vec[dicol[i]] =
        (vec[dicol[i]] - h2 * vec[dicol[i + col_stride]]) /
        coeff[counter];
      --counter;
    }
  }
  delete[] coeff;
}


mgard_cuda_ret
solve_tridiag_M_l_row_cuda(int nrow,       int ncol,
                           int nr,         int nc,
                           int row_stride, int col_stride,
                           int * dirow,    int * dicol,
                           double * dv,     int lddv, 
                           double * dcoords_x) {
  int B = 16;
  int total_thread = ceil((float)nr / row_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  // std::cout << "_solve_tridiag_M_l_row_cuda" << std::endl;
  // std::cout << "thread block: " << tb << std::endl;
  // std::cout << "grid: " << grid << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _solve_tridiag_M_l_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,   ncol,
                                                                 nr,     nc,
                                                                 row_stride, col_stride,
                                                                 dirow,  dicol,
                                                                 dv,     lddv, 
                                                                 dcoords_x);
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
_solve_tridiag_M_l_col_cuda(int nrow,        int ncol,
                             int nr,         int nc,
                             int row_stride, int col_stride,
                             int * dirow,    int * dicol,
                             double * dv,    int lddv, 
                             double * dcoords_y) {
  int idx0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  //printf("thread %d, nr = %d\n", idx0, nr);
  double am, bm, h1, h2;
  double * coeff = new double[nrow];
  for (int idx = idx0; idx < nc; idx += (blockDim.x * gridDim.x) * col_stride) {
    // printf("thread %d, nc = %d, idx = %d\n", idx0, nc, idx);
    int c = dicol[idx];
    // printf("thread %d working on col %d \n", idx0, c);
    double * vec = dv + c;
    am = 2.0 * mgard_common::_get_dist(dcoords_y, dirow[0], dirow[row_stride]); //dirow[row_stride] - dirow[0]
    bm = mgard_common::_get_dist(dcoords_y, dirow[0], dirow[row_stride]) / am; //dirow[row_stride] - dirow[0]

    
    int counter = 1;
    coeff[0] = am;
    
    for (int i = row_stride; i < nr - 1; i += row_stride) {
      h1 = mgard_common::_get_dist(dcoords_y, dirow[i - row_stride], dirow[i]);
      h2 = mgard_common::_get_dist(dcoords_y, dirow[i], dirow[i + row_stride]);

      // h1 = dirow[i] - dirow[i - row_stride];
      // h2 = dirow[i + row_stride] - dirow[i];
      // printf("thread %d working on col %d, vec[%d] = %f \n", idx0, c, dirow[i],  vec[dirow[i] * lddv]);
      vec[dirow[i] * lddv] -= vec[dirow[i - row_stride] * lddv] * bm;

      am = 2.0 * (h1 + h2) - bm * h1;
      bm = h2 / am;

      coeff[counter] = am;
      ++counter;

    }
    h2 = mgard_common::_get_dist(dcoords_y, dirow[nr - 1 - row_stride], dirow[nr - 1]);
    // h2 = get_h_l_cuda(dcoords_y, nr, nrow, nr - 1 - row_stride, row_stride);
    am = 2.0 * h2 - bm * h2;

    vec[dirow[nr - 1] * lddv] -= vec[dirow[nr - 1 - row_stride] * lddv] * bm;
    coeff[counter] = am;

    vec[dirow[nr - 1] * lddv] /= am;
    --counter;

    for (int i = nr - 1 - row_stride; i >= 0; i -= row_stride) {
      h2 = mgard_common::_get_dist(dcoords_y, dirow[i], dirow[i + row_stride]);
      // h2 = get_h_l_cuda(dcoords_y, nr, nrow, i, row_stride);
      vec[dirow[i] * lddv] =
        (vec[dirow[i] * lddv] - h2 * vec[dirow[i + row_stride] * lddv]) /
        coeff[counter];
      --counter;
    }
  }
  delete[] coeff;


}


mgard_cuda_ret
solve_tridiag_M_l_col_cuda(int nrow,       int ncol,
                           int nr,         int nc,
                           int row_stride, int col_stride,
                           int * dirow,    int * dicol,
                           double * dv,    int lddv, 
                           double * dcoords_y) {
  int B = 16;
  int total_thread = ceil((float)nc / col_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // std::cout << "thread block: " << tb << std::endl;
  // std::cout << "grid: " << grid << std::endl;

  _solve_tridiag_M_l_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                                 nr,         nc,
                                                                 row_stride, col_stride,
                                                                 dirow,      dicol,
                                                                 dv,         lddv, 
                                                                 dcoords_y);
  gpuErrchk(cudaGetLastError ()); 
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}




void solve_tridiag_M_l_cuda(const int l, std::vector<double> &v,
                       std::vector<double> &coords, int n, int no) {

  //  int my_level = nlevel - l;
  int stride = std::pow(2, l); // current stride

  double am, bm, h1, h2;
  am = 2.0 *
       get_h_l_cuda(coords.data(), n, no, 0, stride); // first element of upper diagonal U.

  //    bm = get_h_cuda(coords, 0, stride) / am;
  bm = get_h_l_cuda(coords.data(), n, no, 0, stride) / am;
  int nlevel = static_cast<int>(std::log2(n - 1));
  //    //std::cout  << nlevel;
  int nc = std::pow(2, nlevel - l) + 1;
  std::vector<double> coeff(v.size());
  int counter = 1;
  coeff.front() = am;

  ////std::cout  <<  am<< "\t"<< bm<<"\n";
  // forward sweep
  for (int i = stride; i < n - 1; i += stride) {
    h1 = get_h_l_cuda(coords.data(), n, no, i - stride, stride);
    h2 = get_h_l_cuda(coords.data(), n, no, i, stride);

    *get_ref_cuda(v.data(), n, no, i) -= *get_ref_cuda(v.data(), n, no, i - stride) * bm;

    am = 2.0 * (h1 + h2) - bm * h1;
    bm = h2 / am;

    coeff.at(counter) = am;
    ++counter;
  }

  h2 = get_h_l_cuda(coords.data(), n, no, n - 1 - stride, stride);
  am = 2.0 * h2 - bm * h2;

  //    *get_ref_cuda(v, n, no, n-1) -= *get_ref_cuda(v, n, no, n-1-stride)*bm;
  v.back() -= *get_ref_cuda(v.data(), n, no, n - 1 - stride) * bm;
  coeff.at(counter) = am;

  // backward sweep

  //    *get_ref_cuda(v, n, no, n-1) /= am;
  v.back() /= am;
  --counter;

  for (int i = n - 1 - stride; i >= 0; i -= stride) {
    h2 = get_h_l_cuda(coords.data(), n, no, i, stride);
    *get_ref_cuda(v.data(), n, no, i) =
        (*get_ref_cuda(v.data(), n, no, i) - h2 * (*get_ref_cuda(v.data(), n, no, i + stride))) /
        coeff.at(counter);

    //        *get_ref_cuda(v, n, no, i) = 3  ;

    --counter;
  }
}

__global__ void 
_add_level_l_cuda(int nrow,       int ncol, 
               int nr,          int nc,
               int row_stride, int col_stride,
               int * irow,     int * icol,
               double * dv,    int lddv, 
               double * dwork, int lddwork) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int idx_x = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;
    int idx_y = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
    //printf("x = %d, y = %d, stride = %d \n", x,y,stride);
    for (int y = idx_y; y < nr; y += blockDim.y * gridDim.y * row_stride) {
      for (int x = idx_x; x < nc; x += blockDim.x * gridDim.x * col_stride) {
        
        int r = irow[y];
        int c = icol[x];
        dv[get_idx(lddv, r, c)] += dwork[get_idx(lddwork, r, c)];
        //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
        //y += blockDim.y * gridDim.y * stride;
      }
        //x += blockDim.x * gridDim.x * stride;
    }
}

mgard_cuda_ret
add_level_l_cuda(int nrow,       int ncol, 
                 int nr,         int nc,
                 int row_stride, int col_stride,
                 int * dirow,    int * dicol,
                 double * dv,    int lddv, 
                 double * dwork, int lddwork) {

  int B = 16;
  int total_thread_x = nc/col_stride;
  int total_thread_y = nr/row_stride;
  int tbx = min(B, total_thread_x);
  int tby = min(B, total_thread_y);
  int gridx = ceil((float)total_thread_x/tbx);
  int gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
  // std::cout << "grid: " << gridx << ", " << gridy <<std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  _add_level_l_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                       nr,         nc,
                                                      row_stride, col_stride, 
                                                      dirow,      dicol,
                                                      dv,         lddv,
                                                      dwork,      lddwork);


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
_subtract_level_l_cuda(int nrow, int ncol, 
               int nr,           int nc,
               int row_stride,   int col_stride,
               int * irow,       int * icol,
               double * dv,      int lddv, 
               double * dwork,   int lddwork) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int idx_x = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;
    int idx_y = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
    //printf("x = %d, y = %d, stride = %d \n", x,y,stride);
    for (int y = idx_y; y < nr; y += blockDim.y * gridDim.y * row_stride) {
      for (int x = idx_x; x < nc; x += blockDim.x * gridDim.x * col_stride) {
        
        int r = irow[y];
        int c = icol[x];
        dv[get_idx(lddv, r, c)] -= dwork[get_idx(lddwork, r, c)];
        //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
        //y += blockDim.y * gridDim.y * stride;
      }
        //x += blockDim.x * gridDim.x * stride;
    }
}

mgard_cuda_ret 
subtract_level_l_cuda(int nrow,       int ncol, 
                      int nr,         int nc,
                      int row_stride, int col_stride,
                      int * dirow,    int * dicol,
                      double * dv,    int lddv, 
                      double * dwork, int lddwork) {

  int B = 16;
  int total_thread_x = nc/col_stride;
  int total_thread_y = nr/row_stride;
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

  _subtract_level_l_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                       nr,         nc,
                                                      row_stride, col_stride, 
                                                      dirow,      dicol,
                                                      dv,         lddv,
                                                      dwork,      lddwork);
  gpuErrchk(cudaGetLastError ()); 

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);

}



// void add_level_l_cuda(const int l, double *v, double *work, int nr, int nc, int nrow,
//                  int ncol) {
//   // v += work at level l

//   int stride = std::pow(2, l); // current stride

//   for (int irow = 0; irow < nr; irow += stride) {
//     int ir = get_lindex_cuda(nr, nrow, irow);
//     for (int jcol = 0; jcol < nc; jcol += stride) {
//       int jr = get_lindex_cuda(nc, ncol, jcol);
//       v[mgard_common::get_index_cuda(ncol, ir, jr)] +=
//           work[mgard_common::get_index_cuda(ncol, ir, jr)];
//     }
//   }
// }


void 
prep_2D_cuda(const int nrow,     const int ncol,
             const int nr,       const int nc, 
             int * dirow,        int * dicol,
             int * dirowP,       int * dicolP,
             double * dv,        int lddv, 
             double * dwork,     int lddwork,
             double * dcoords_x, double * dcoords_y) {

  // std::vector<double> row_vec (ncol);
  // std::vector<double> col_vec (nrow);
  // std::cout << "size of row_vec = " << ncol << std::endl;
  // std::cout << "size of col_vec = " << nrow << std::endl;
  // std::vector<double> work(nrow * ncol);
  // double * v = new double[nrow * ncol];
  // std::vector<double> coords_x(ncol), coords_y(nrow);
      
  // std::iota(std::begin(coords_x), std::end(coords_x), 0);
  // std::iota(std::begin(coords_y), std::end(coords_y), 0);


  // cudaMemcpy2DHelper(v, ncol  * sizeof(double), 
  //                        dv, lddv * sizeof(double), 
  //                        ncol * sizeof(double), nrow, 
  //                        D2H);

  // cudaMemcpy2DHelper(work.data(), ncol  * sizeof(double), 
  //                    dwork, lddwork * sizeof(double), 
  //                    ncol * sizeof(double), nrow, 
  //                    D2H);

  mgard_cuda_ret ret;

  double pi_Ql_first_cuda_time = 0.0;
  double copy_level_cuda_time = 0.0;
  double assign_num_level_l_cuda_time = 0.0;

  double mass_matrix_multiply_row_cuda_time = 0.0;
  double restriction_first_row_cuda_time = 0.0;
  double solve_tridiag_M_l_row_cuda_time = 0.0;
  
  double mass_matrix_multiply_col_cuda_time = 0.0;
  double restriction_first_col_cuda_time = 0.0;
  double solve_tridiag_M_l_col_cuda_time = 0.0;
  double add_level_l_cuda_time = 0.0;

  int l = 0;
  int row_stride = 1;
  int col_stride = 1;
  //int ldv = ncol;
  //int ldwork = ncol;

  // pi_Ql_first(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec, col_vec);
  ret = pi_Ql_first_cuda(nrow,      ncol,
                   nr,        nc, 
                   dirow,     dicol,
                   dirowP,    dicolP,
                   dcoords_x, dcoords_y,
                   dv,        lddv); //(I-\Pi)u this is the initial move to 2^k+1 nodes
  pi_Ql_first_cuda_time = ret.time;
  // mgard_cannon::copy_level(nrow, ncol, 0, v, work);
  ret = mgard_cannon::copy_level_cuda(nrow,       ncol, 
                                row_stride, col_stride,
                                dv,         lddv,
                                dwork,      lddwork);
  copy_level_cuda_time = ret.time;

  // assign_num_level_l(0, work.data(), 0.0, nr, nc, nrow, ncol);
  ret = assign_num_level_l_cuda(nrow,       ncol,
                          nr,         nc,
                          row_stride, col_stride,
                          dirow,      dicol,
                          dwork,      lddwork, 
                          0.0);
  assign_num_level_l_cuda_time = ret.time;

  row_stride = 1;
  col_stride = 1;
  ret = mgard_cannon::mass_matrix_multiply_row_cuda(nrow,       ncol,
                                              row_stride, col_stride,
                                              dwork,      lddwork,
                                              dcoords_x);
  mass_matrix_multiply_row_cuda_time = ret.time;

  ret = restriction_first_row_cuda(nrow,       ncol,
                             row_stride, dicolP, nc,
                             dwork,      lddwork,
                             dcoords_x);
  restriction_first_row_cuda_time = ret.time;

 //  col_stride = 1;
  ret = solve_tridiag_M_l_row_cuda(nrow,       ncol,
                             nr,         nc,
                             row_stride, col_stride,
                             dirow,      dicol, 
                             dwork,      lddwork, 
                             dcoords_x);
  solve_tridiag_M_l_row_cuda_time = ret.time;


  // for (int i = 0; i < nrow; ++i) {
  //   int ir = get_lindex_cuda(nr, nrow, i);
  //   for (int j = 0; j < ncol; ++j) {
  //     row_vec[j] = work[mgard_common::get_index_cuda(ncol, i, j)];
  //   }

  //   mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);

  //   restriction_first(row_vec, coords_x, nc, ncol);

  //   for (int j = 0; j < ncol; ++j) {
  //     work[mgard_common::get_index_cuda(ncol, i, j)] = row_vec[j];
  //   }
  // }

  // for (int i = 0; i < nr; ++i) {
  //   int ir = get_lindex_cuda(nr, nrow, i);
  //   for (int j = 0; j < ncol; ++j) {
  //     row_vec[j] = work[mgard_common::get_index_cuda(ncol, ir, j)];
  //   }

  //   mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol);

  //   for (int j = 0; j < ncol; ++j) {
  //     work[mgard_common::get_index_cuda(ncol, ir, j)] = row_vec[j];
  //   }
  // }

  //   //   //std::cout  << "recomposing-colsweep" << "\n";

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) // check if we have 1-D array..
  {
    row_stride = 1;
    col_stride = 1;
    ret = mgard_cannon::mass_matrix_multiply_col_cuda(nrow,       ncol,
                                                row_stride, col_stride,
                                                dwork,      lddwork,
                                                dcoords_y);
    mass_matrix_multiply_col_cuda_time = ret.time;


    ret = restriction_first_col_cuda(nrow,   ncol,
                               dirowP, nr, col_stride,
                               dwork,  lddwork,
                               dcoords_y);
    restriction_first_col_cuda_time = ret.time;

    // print_matrix(nrow, ncol, work.data(), ldwork);
    ret = solve_tridiag_M_l_col_cuda(nrow,       ncol,
                               nr,         nc,
                               row_stride, col_stride,
                               dirow,      dicol,
                               dwork,      lddwork, 
                               dcoords_y);
    solve_tridiag_M_l_col_cuda_time = ret.time;

    // for (int j = 0; j < ncol; ++j) {
    //        int jr  = get_lindex_cuda(nc,  ncol,  j);
    //   for (int i = 0; i < nrow; ++i) {
    //     col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, j)];
    //   }

    //   mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);

    //   mgard_gen::restriction_first(col_vec, coords_y, nr, nrow);

    //   for (int i = 0; i < nrow; ++i) {
    //     work[mgard_common::get_index_cuda(ncol, i, j)] = col_vec[i];
    //   }
    // }

    // for (int j = 0; j < nc; ++j) {
    //   int jr = get_lindex_cuda(nc, ncol, j);
    //   for (int i = 0; i < nrow; ++i) {
    //     col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, jr)];
    //   }

    //   mgard_gen::solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow);
    //   for (int i = 0; i < nrow; ++i) {
    //     work[mgard_common::get_index_cuda(ncol, i, jr)] = col_vec[i];
    //   }
    // }
 }
  // add_level_l(0, v, work.data(), nr, nc, nrow, ncol);
  row_stride = 1;
  col_stride = 1;
  ret = add_level_l_cuda(nrow,       ncol, 
                   nr,         nc, 
                   row_stride, col_stride, 
                   dirow,      dicol, 
                   dv,         lddv, 
                   dwork,      lddwork);
  add_level_l_cuda_time = ret.time;

  // cudaMemcpy2DHelper(dv, lddv * sizeof(double),
  //                  v, ncol  * sizeof(double), 
  //                  ncol * sizeof(double), nrow, 
  //                  H2D);

  // cudaMemcpy2DHelper(dwork, lddwork * sizeof(double),
  //                    work.data(), ncol  * sizeof(double), 
  //                    ncol * sizeof(double), nrow, 
  //                    H2D);
  std::ofstream timing_results;
  timing_results.open ("prep_2D_cuda.csv");
  timing_results << "pi_Ql_first_cuda_time," << pi_Ql_first_cuda_time << std::endl;
  timing_results << "copy_level_cuda_time," << copy_level_cuda_time << std::endl;
  timing_results << "assign_num_level_l_cuda_time," << assign_num_level_l_cuda_time << std::endl;

  timing_results << "mass_matrix_multiply_row_cuda_time," << mass_matrix_multiply_row_cuda_time << std::endl;
  timing_results << "restriction_first_row_cuda_time," << restriction_first_row_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_row_cuda_time," << solve_tridiag_M_l_row_cuda_time << std::endl;
  
  timing_results << "mass_matrix_multiply_col_cuda_time," << mass_matrix_multiply_col_cuda_time << std::endl;
  timing_results << "restriction_first_col_cuda_time," << restriction_first_col_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_col_cuda_time," << solve_tridiag_M_l_col_cuda_time << std::endl;
  timing_results << "add_level_l_cuda_time," << add_level_l_cuda_time << std::endl;
  timing_results.close();
}





__global__ void 
_pi_Ql_cuda(int nrow,           int ncol,
            int nr,             int nc,
            int row_stride,     int col_stride,
            int * irow,         int * icol,
            double * dv,        int lddv, 
            double * dcoords_x, double * dcoords_y) {

  int row_Cstride = row_stride * 2;
  int col_Cstride = col_stride * 2;
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_Cstride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_Cstride;
  
  // in most cases it only needs to iterate once unless the input is really large
  for (int y = y0; y + row_Cstride <= nr - 1; y += blockDim.y * gridDim.y * row_Cstride) {
    for (int x = x0; x + col_Cstride <= nc - 1; x += blockDim.x * gridDim.x * col_Cstride) {
      register double a00 = dv[get_idx(lddv, irow[y],             icol[x]             )];
      register double a01 = dv[get_idx(lddv, irow[y],             icol[x+col_stride]  )];
      register double a02 = dv[get_idx(lddv, irow[y],             icol[x+col_Cstride] )];
      register double a10 = dv[get_idx(lddv, irow[y+row_stride],  icol[x]             )];
      register double a11 = dv[get_idx(lddv, irow[y+row_stride],  icol[x+col_stride]  )];
      register double a12 = dv[get_idx(lddv, irow[y+row_stride],  icol[x+col_Cstride] )];
      register double a20 = dv[get_idx(lddv, irow[y+row_Cstride], icol[x]             )];
      register double a21 = dv[get_idx(lddv, irow[y+row_Cstride], icol[x+col_stride]  )];
      register double a22 = dv[get_idx(lddv, irow[y+row_Cstride], icol[x+col_Cstride] )];

      // printf("thread (%d, %d) working on v(%d, %d) = %f \n", y0, x0, irow[y], icol[x+col_stride], a01);

      int h1_col = mgard_common::_get_dist(dcoords_x, icol[x], icol[x + col_stride]);  //icol[x+col_stride]  - icol[x];
      int h2_col = mgard_common::_get_dist(dcoords_x, icol[x + col_stride], icol[x + col_Cstride]);  //icol[x+col_Cstride] - icol[x+col_stride];
      int hsum_col = h1_col + h2_col;
   
      int h1_row = mgard_common::_get_dist(dcoords_y, irow[y], irow[y + row_stride]);  //irow[y+row_stride]  - irow[y];
      int h2_row = mgard_common::_get_dist(dcoords_y, irow[y + row_stride], irow[y + row_Cstride]);  //irow[y+row_Cstride] - irow[y+row_stride];
      int hsum_row = h1_row + h2_row;
      //double ta01 = a01;
      a01 -= (h1_col * a02 + h2_col * a00) / hsum_col;
      //a21 -= (h1_col * a22 + h2_col * a20) / hsum_col;
       // printf("thread (%d, %d) working on v(%d, %d) = %f -> %f \n", y0, x0, irow[y], icol[x+col_stride], ta01, a01);
      
      a10 -= (h1_row * a20 + h2_row * a00) / hsum_row;
      //a12 -= (h1_row * a22 + h2_row * a02) / hsum_row;
     

      a11 -= 1.0 / (hsum_row * hsum_col) * (a00 * h2_col * h2_row + a02 * h1_col * h2_row + a20 * h2_col * h1_row + a22 * h1_col * h1_row);
      
      dv[get_idx(lddv, irow[y],             icol[x+col_stride]  )] = a01;
      dv[get_idx(lddv, irow[y+row_stride],  icol[x]             )] = a10;
      dv[get_idx(lddv, irow[y+row_stride],  icol[x+col_stride]  )] = a11;

      if (x + col_Cstride == nc - 1) {
        a12 -= (h1_row * a22 + h2_row * a02) / hsum_row;
        dv[get_idx(lddv, irow[y+row_stride],  icol[x+col_Cstride] )] = a12;
      }
      if (y + row_Cstride == nr - 1) {
        //double ta21=a21;
        a21 -= (h1_col * a22 + h2_col * a20) / hsum_col;
        dv[get_idx(lddv, irow[y+row_Cstride], icol[x+col_stride]  )] = a21;
         // printf("thread (%d, %d) working on v(%d, %d) = %f -> %f \n", y0, x0, irow[y+row_Cstride], icol[x+col_stride], ta21, a21);
      }
    }
  }

}

mgard_cuda_ret 
pi_Ql_cuda(int nrow,           int ncol,
           int nr,             int nc,
           int row_stride,     int col_stride,
           int * dirow,        int * dicol,
           double * dv,        int lddv, 
           double * dcoords_x, double * dcoords_y) {

  int B = 16;
  int total_thread_y = floor((double)nr/(row_stride * 2));
  int total_thread_x = floor((double)nc/(col_stride * 2));
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _pi_Ql_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                 nr,         nc,
                                                 row_stride, col_stride,
                                                 dirow,      dicol,
                                                 dv,         lddv,
                                                 dcoords_x,  dcoords_y);
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
_copy_level_l_cuda(int nrow,           int ncol,
                   int nr,             int nc,
                   int row_stride,     int col_stride,
                   int * irow,         int * icol,
                   double * dv,        int lddv,
                   double * dwork,     int lddwork) {
  
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;

  for (int y = y0; y < nr; y += blockDim.y * gridDim.y * row_stride) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x * col_stride) {
      dwork[get_idx(lddwork, irow[y], icol[x])] = dv[get_idx(lddv, irow[y], icol[x])];
    }
  }
}

mgard_cuda_ret 
copy_level_l_cuda(int nrow,       int ncol,
                  int nr,         int nc,
                  int row_stride, int col_stride,
                  int * dirow,    int * dicol,
                  double * dv,    int lddv,
                  double * dwork, int lddwork) {

  int B = 16;
  int total_thread_y = ceil((double)nr/(row_stride));
  int total_thread_x = ceil((double)nc/(col_stride));
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

  _copy_level_l_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                        nr,         nc,
                                                        row_stride, col_stride,
                                                        dirow,      dicol,
                                                        dv,         lddv,
                                                        dwork,      lddwork);

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
_assign_num_level_l_cuda(int nrow,           int ncol,
                         int nr,             int nc,
                         int row_stride,     int col_stride,
                         int * dirow,        int * dicol,
                         double * dv,        int lddv,
                         double num) {
  
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;

  for (int y = y0; y < nr; y += blockDim.y * gridDim.y * row_stride) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x * col_stride) {
      dv[get_idx(lddv, dirow[y], dicol[x])] = num;
    }
  }
}


mgard_cuda_ret 
assign_num_level_l_cuda(int nrow,           int ncol,
                        int nr,             int nc,
                        int row_stride,     int col_stride,
                        int * dirow,        int * dicol,
                        double * dv,        int lddv,
                        double num) {
  int B = 16;
  int total_thread_y = ceil((float)nr/(row_stride));
  int total_thread_x = ceil((float)nc/(col_stride));
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);


  // std::cout << "thread block: " << tby << ", " << tby << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _assign_num_level_l_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                              nr,         nc,
                                                              row_stride, col_stride,
                                                              dirow,      dicol,
                                                              dv,         lddv,
                                                              num);
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
_mass_mult_l_row_cuda(int nrow,       int ncol,
                      int nr,         int nc,
                      int row_stride, int col_stride,
                      int * dirow,    int * dicol,
                      double * dv,    int lddv,
                      double * dcoords_x) {
  int r0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  for (int r = r0; r < nr; r += (blockDim.x * gridDim.x) * row_stride) {
    // printf("thread %d is working on row %d\n", r0, dirow[r]);
    double * vec = dv + dirow[r] * lddv;
    double temp1, temp2;
    double h1, h2;
    temp1 = vec[dicol[0]];
    h1 = mgard_common::_get_dist(dcoords_x,  dicol[0], dicol[col_stride]); //dicol[col_stride] - dicol[0];
    

    vec[dicol[0]] = 2.0 * h1 * temp1 + h1 * vec[dicol[col_stride]];

    for (int i = col_stride; i <= nc - 1 - col_stride; i += col_stride) {
      temp2 = vec[dicol[i]];
      h1 = mgard_common::_get_dist(dcoords_x, dicol[i - col_stride], dicol[i]);
      h2 = mgard_common::_get_dist(dcoords_x, dicol[i], dicol[i + col_stride]);
      // printf("thread %d is working on h1 = %f, h2 = %f\n", r0, h1, h2);
      // h1 = dicol[i] - dicol[i - col_stride];
      // h2 = dicol[i + col_stride] - dicol[i];
      vec[dicol[i]] = h1 * temp1  + 2 * (h1 + h2) * temp2 + h2 * vec[dicol[i + col_stride]];
      temp1 = temp2;
    }
    vec[dicol[nc - 1]] = mgard_common::_get_dist(dcoords_x, dicol[nc - col_stride - 1], dicol[nc - 1]) * temp1 +
                        2 * mgard_common::_get_dist(dcoords_x, dicol[nc - col_stride - 1], dicol[nc - 1]) * vec[dicol[nc - 1]];
    // vec[dicol[nc - 1]] = (dicol[nc - 1] - dicol[nc - col_stride - 1]) * temp1 +
    //                     2 * (dicol[nc - 1] - dicol[nc - col_stride - 1]) * vec[dicol[nc - 1]];

  }
}

mgard_cuda_ret 
mass_mult_l_row_cuda(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     double * dv,    int lddv,
                     double * dcoords_x) {
 
  int B = 16;
  //int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread_x = ceil((float)nr/(row_stride));
  //int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  //int gridy = ceil(total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _mass_mult_l_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                           nr,         nc,
                                                           row_stride, col_stride,
                                                           dirow,      dicol,
                                                           dv,         lddv,
                                                           dcoords_x);
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
_mass_mult_l_col_cuda(int nrow,       int ncol,
                      int nr,         int nc,
                      int row_stride, int col_stride,
                      int * dirow,    int * dicol,
                      double * dv,    int lddv,
                      double * dcoords_y) {
  int c0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  //printf("thread1 %d working on col %d\n", c0, dicol[c0]);

  for (int c = c0; c < nc; c += (blockDim.x * gridDim.x) * col_stride) {
    double * vec = dv + dicol[c];
    //printf("thread %d working on col %d\n", c0, dicol[c]);
    double temp1, temp2;
    double h1, h2;
    temp1 = vec[dirow[0] * lddv];
    //printf("thread %d temp1 = %.6f\n", c0, temp1);
    h1 = mgard_common::_get_dist(dcoords_y,  dirow[0], dirow[row_stride]); //dirow[row_stride] - dirow[0];
    vec[dirow[0] * lddv] = 2.0 * h1 * temp1 + h1 * vec[dirow[row_stride] * lddv];
    // printf("thread %d vec[0] = %.6f\n", c0, vec[dirow[0] * lddv]);
    for (int i = row_stride; i <= nr - 1 - row_stride; i += row_stride) {
      temp2 = vec[dirow[i] * lddv];
      h1 = mgard_common::_get_dist(dcoords_y, dirow[i - row_stride], dirow[i]);
      h2 = mgard_common::_get_dist(dcoords_y, dirow[i], dirow[i + row_stride]);


      // h1 = dirow[i] - dirow[i - row_stride];
      // h2 = dirow[i + row_stride] - dirow[i];
      vec[dirow[i] * lddv] = h1 * temp1  + 2 * (h1 + h2) * temp2 + h2 * vec[dirow[i + row_stride] * lddv];
      temp1 = temp2;
    }
    vec[dirow[nr - 1] * lddv] = mgard_common::_get_dist(dcoords_y, dirow[nr - row_stride - 1], dirow[nr - 1]) * temp1 +
                        2 * mgard_common::_get_dist(dcoords_y, dirow[nr - row_stride - 1], dirow[nr - 1]) * vec[dirow[nr - 1] * lddv];


    // vec[dirow[nr - 1] * lddv] = (dirow[nr - 1] - dirow[nr - row_stride - 1]) * temp1 +
    //                     2 * (dirow[nr - 1] - dirow[nr - row_stride - 1]) * vec[dirow[nr - 1] * lddv];
  }
}

mgard_cuda_ret 
mass_mult_l_col_cuda(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     double * dv,    int lddv,
                     double * dcoords_y) {
  int B = 16;
  //int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread_x = ceil((float)nc/(col_stride));
  //int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  //int gridy = ceil(total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1); 

  // std::cout << "_mass_mult_l_col_cuda" << std::endl;
  // std::cout << "thread block: " << tbx  << std::endl;
  // std::cout << "grid: " << gridx  << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _mass_mult_l_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                           nr,         nc,
                                                           row_stride, col_stride,
                                                           dirow,      dicol,
                                                           dv,         lddv,
                                                           dcoords_y);
  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}


// __global__ void
// _restriction_l_row_cuda(int nrow,       int ncol,
//                       int nr,         int nc,
//                       int row_stride, int col_stride,
//                       int * dirow,    int * dicol,
//                       double * dv,    int lddv,
//                       double * dcoords_x) {
//   int col_Pstride = col_stride / 2;
//   int r0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
//   for (int r = r0; r < nr; r += (blockDim.x * gridDim.x) * row_stride) {
//     double * vec = dv + dirow[r] * lddv;
//     double h1 = mgard_common::_get_dist(dcoords_x, dicol[0], dicol[col_Pstride]); //dicol[col_Pstride] - dicol[0];
//     double h2 = mgard_common::_get_dist(dcoords_x, dicol[col_Pstride], dicol[col_stride]); //dicol[col_stride] - dicol[col_Pstride];
//     double hsum = h1 + h2;
//     vec[dicol[0]] += h2 * vec[dicol[col_Pstride]] / hsum;

//     for (int i = col_stride; i <= nc - col_stride; i += col_stride) {
//       vec[dicol[i]] += h1 * vec[dicol[i - col_Pstride]] / hsum;

//       h1 = mgard_common::_get_dist(dcoords_x, dicol[i], dicol[i + col_Pstride]);
//       h2 = mgard_common::_get_dist(dcoords_x, dicol[i + col_Pstride], dicol[i + col_stride]);


//       // h1 = dicol[i + col_Pstride] - dicol[i];
//       // h2 = dicol[i + col_stride] - dicol[i + col_Pstride];
//       hsum = h1 + h2;
//       vec[dicol[i]] += h2 * vec[dicol[i + col_Pstride]] / hsum;
//     }
//     vec[dicol[nc - 1]] += h1 * vec[dicol[nc - col_Pstride - 1]] / hsum;
//   }
//}


__global__ void
_restriction_l_row_cuda(int nrow,       int ncol,
                      int nr,         int nc,
                      int row_stride, int col_stride,
                      int * dirow,    int * dicol,
                      double * dv,    int lddv,
                      double * dcoords_x) {
  int col_Cstride = col_stride * 2;
  int r0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  for (int r = r0; r < nr; r += (blockDim.x * gridDim.x) * row_stride) {
    double * vec = dv + dirow[r] * lddv;
    double h1 = mgard_common::_get_dist(dcoords_x, dicol[0], dicol[col_stride]); //dicol[col_Pstride] - dicol[0];
    double h2 = mgard_common::_get_dist(dcoords_x, dicol[col_stride], dicol[col_Cstride]); //dicol[col_stride] - dicol[col_Pstride];
    double hsum = h1 + h2;
    vec[dicol[0]] += h2 * vec[dicol[col_stride]] / hsum;

    for (int i = col_Cstride; i <= nc - col_Cstride; i += col_Cstride) {
      vec[dicol[i]] += h1 * vec[dicol[i - col_stride]] / hsum;

      h1 = mgard_common::_get_dist(dcoords_x, dicol[i], dicol[i + col_stride]);
      h2 = mgard_common::_get_dist(dcoords_x, dicol[i + col_stride], dicol[i + col_Cstride]);


      // h1 = dicol[i + col_Pstride] - dicol[i];
      // h2 = dicol[i + col_stride] - dicol[i + col_Pstride];
      hsum = h1 + h2;
      vec[dicol[i]] += h2 * vec[dicol[i + col_stride]] / hsum;
    }
    vec[dicol[nc - 1]] += h1 * vec[dicol[nc - col_stride - 1]] / hsum;
  }
}

mgard_cuda_ret 
restriction_l_row_cuda(int nrow,       int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * dirow,    int * dicol,
                       double * dv,    int lddv,
                       double * dcoords_x) {

  int B = 16;
  //int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread_x = ceil((float)nr/(row_stride));
  //int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  //int gridy = ceil(total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);


  _restriction_l_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                             nr,         nc,
                                                             row_stride, col_stride,
                                                             dirow,      dicol,
                                                             dv,         lddv,
                                                             dcoords_x);
  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}


// __global__ void
// _restriction_l_col_cuda(int nrow,       int ncol,
//                       int nr,         int nc,
//                       int row_stride, int col_stride,
//                       int * dirow,    int * dicol,
//                       double * dv,    int lddv,
//                       double * dcoords_y) {
//   int row_Pstride = row_stride / 2;
//   int c0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
//   for (int c = c0; c < nc; c += (blockDim.x * gridDim.x) * col_stride) {
//     double * vec = dv + dicol[c];
//     double h1 = mgard_common::_get_dist(dcoords_y, dirow[0], dirow[row_Pstride]); //dirow[row_Pstride] - dirow[0];
//     double h2 = mgard_common::_get_dist(dcoords_y, dirow[row_Pstride], dirow[row_stride]); //dirow[row_stride] - dirow[row_Pstride];
//     double hsum = h1 + h2;
//     vec[dirow[0] * lddv] += h2 * vec[dirow[row_Pstride] * lddv] / hsum;

//     for (int i = row_stride; i <= nr - row_stride; i += row_stride) {
//       vec[dirow[i] * lddv] += h1 * vec[dirow[i - row_Pstride] * lddv] / hsum;
//       h1 = mgard_common::_get_dist(dcoords_y, dirow[i], dirow[i + row_Pstride]);
//       h2 = mgard_common::_get_dist(dcoords_y, dirow[i + row_Pstride], dirow[i + row_stride]);

//       // printf("dist_1[%d, %d] = %f\n", dirow[i], dirow[i + row_Pstride], h1);
//       // printf("dist_1[%d, %d] = %f\n", dirow[i + row_stride], dirow[i + row_Pstride], h2);
//       // h1 = dirow[i + row_Pstride] - dirow[i];
//       // h2 = dirow[i + row_stride] - dirow[i + row_Pstride];

//       // printf("dist_2[%d, %d] = %f\n", dirow[i], dirow[i + row_Pstride], h1);
//       // printf("dist_2[%d, %d] = %f\n", dirow[i + row_stride], dirow[i + row_Pstride], h2);
//       hsum = h1 + h2;
//       vec[dirow[i] * lddv] += h2 * vec[dirow[i + row_Pstride] * lddv] / hsum;
//     }
//     vec[dirow[nr - 1] * lddv] += h1 * vec[dirow[nr - row_Pstride - 1] * lddv] / hsum;
//   }
// }

__global__ void
_restriction_l_col_cuda(int nrow,       int ncol,
                      int nr,         int nc,
                      int row_stride, int col_stride,
                      int * dirow,    int * dicol,
                      double * dv,    int lddv,
                      double * dcoords_y) {
  int row_Cstride = row_stride * 2;
  int c0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  for (int c = c0; c < nc; c += (blockDim.x * gridDim.x) * col_stride) {
    double * vec = dv + dicol[c];
    double h1 = mgard_common::_get_dist(dcoords_y, dirow[0], dirow[row_stride]); //dirow[row_Pstride] - dirow[0];
    double h2 = mgard_common::_get_dist(dcoords_y, dirow[row_stride], dirow[row_Cstride]); //dirow[row_stride] - dirow[row_Pstride];
    double hsum = h1 + h2;
    vec[dirow[0] * lddv] += h2 * vec[dirow[row_stride] * lddv] / hsum;

    for (int i = row_Cstride; i <= nr - row_Cstride; i += row_Cstride) {
      vec[dirow[i] * lddv] += h1 * vec[dirow[i - row_stride] * lddv] / hsum;
      h1 = mgard_common::_get_dist(dcoords_y, dirow[i], dirow[i + row_stride]);
      h2 = mgard_common::_get_dist(dcoords_y, dirow[i + row_stride], dirow[i + row_Cstride]);

      // printf("dist_1[%d, %d] = %f\n", dirow[i], dirow[i + row_Pstride], h1);
      // printf("dist_1[%d, %d] = %f\n", dirow[i + row_stride], dirow[i + row_Pstride], h2);
      // h1 = dirow[i + row_Pstride] - dirow[i];
      // h2 = dirow[i + row_stride] - dirow[i + row_Pstride];

      // printf("dist_2[%d, %d] = %f\n", dirow[i], dirow[i + row_Pstride], h1);
      // printf("dist_2[%d, %d] = %f\n", dirow[i + row_stride], dirow[i + row_Pstride], h2);
      hsum = h1 + h2;
      vec[dirow[i] * lddv] += h2 * vec[dirow[i + row_stride] * lddv] / hsum;
    }
    vec[dirow[nr - 1] * lddv] += h1 * vec[dirow[nr - row_stride - 1] * lddv] / hsum;
  }
}

mgard_cuda_ret 
restriction_l_col_cuda(int nrow,       int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * dirow,    int * dicol,
                       double * dv,    int lddv,
                       double * dcoords_y) {
  int B = 16;
  //int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread_x = ceil((float)nc/(col_stride));
  //int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  //int gridy = ceil(total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);


  _restriction_l_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                             nr,         nc,
                                                             row_stride, col_stride,
                                                             dirow,      dicol,
                                                             dv,         lddv,
                                                             dcoords_y);
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
_prolongate_l_row_cuda(int nrow,       int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * dirow,    int * dicol,
                       double * dv,    int lddv,
                       double * coords_x) {

  int col_Pstride = col_stride / 2;
  int r0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  for (int r = r0; r < nr; r += (blockDim.x * gridDim.x) * row_stride) {
    double * vec = dv + dirow[r] * lddv;
    for (int i = col_stride; i < nc; i += col_stride) {
      double h1 = mgard_common::_get_dist(coords_x, dicol[i - col_stride], dicol[i - col_Pstride]);
      double h2 = mgard_common::_get_dist(coords_x, dicol[i - col_Pstride], dicol[i]);
      // double h1 = dicol[i - col_Pstride] - dicol[i - col_stride];
      // double h2 = dicol[i] - dicol[i - col_Pstride];
      double hsum = h1 + h2;
      vec[dicol[i - col_Pstride]] = (h2 * vec[dicol[i - col_stride]] + h1 * vec[dicol[i]]) / hsum;
    }
  }
}

mgard_cuda_ret 
prolongate_l_row_cuda(int nrow,       int ncol,
                      int nr,         int nc,
                      int row_stride, int col_stride,
                      int * dirow,    int * dicol,
                      double * dv,    int lddv,
                      double * dcoords_x) {
  int B = 16;
  //int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread_x = ceil((float)nr/(row_stride));
  //int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  //int gridy = ceil(total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _prolongate_l_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                            nr,         nc,
                                                            row_stride, col_stride,
                                                            dirow,      dicol,
                                                            dv,         lddv,
                                                            dcoords_x);
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
_prolongate_l_col_cuda(int nrow,       int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * dirow,    int * dicol,
                       double * dv,    int lddv,
                       double * coords_y) {

  int row_Pstride = row_stride / 2;
  int c0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  for (int c = c0; c < nc; c += (blockDim.x * gridDim.x) * col_stride) {
    double * vec = dv + dicol[c];
    for (int i = row_stride; i < nr; i += row_stride) {
      double h1 = mgard_common::_get_dist(coords_y, dirow[i - row_stride], dirow[i - row_Pstride]);
      double h2 = mgard_common::_get_dist(coords_y, dirow[i - row_Pstride], dirow[i]);
      // double h1 = dirow[i - row_Pstride] - dirow[i - row_stride];
      // double h2 = dirow[i] - dirow[i - row_Pstride];
      double hsum = h1 + h2;
      vec[dirow[i - row_Pstride] * lddv] = (h2 * vec[dirow[i - row_stride] * lddv] + h1 * vec[dirow[i] * lddv]) / hsum;
    }
  }
}

mgard_cuda_ret 
prolongate_l_col_cuda(int nrow,        int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * dirow,    int * dicol,
                       double * dv,    int lddv,
                       double * dcoords_y) {
  int B = 16;
  //int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread_x = ceil((double)ncol/(col_stride));
  //int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  //int gridy = ceil(total_thread_y/tby);
  int gridx = ceil((double)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _prolongate_l_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                            nr,         nc,
                                                            row_stride, col_stride,
                                                            dirow,      dicol,
                                                            dv,         lddv,
                                                            dcoords_y);
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
_prolongate_last_row_cuda(int nrow,       int ncol,
                          int nr,         int nc,
                          int row_stride, int col_stride,
                          int * dirow,    int * dicolP,
                          double * dv,    int lddv,
                          double * dcoords_x) {
  int r0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  //int y = threadIdx.y * stride;
  for (int r = r0; r < nr; r += (blockDim.x * gridDim.x) * row_stride) {
    // printf("thread %d working on row %d \n", r0, dirow[r]);
    double * vec = dv + dirow[r] * lddv;
    for (int i = 0; i < ncol-nc; i++) {
      double h1 = 1;//mgard_common::get_h_cuda(dcoords_x, icolP[i] - 1, 1);
      double h2 = 1;//mgard_common::get_h_cuda(dcoords_x, icolP[i]    , 1);
      double hsum = h1 + h2;
      // printf("thread %d working on vec = %f %f %f \n", r0, vec[dicolP[i] - 1], vec[dicolP[i]], vec[dicolP[i] + 1]);
      vec[dicolP[i]] = (h2 * vec[dicolP[i] - 1] + h1 * vec[dicolP[i] + 1]) / hsum;
      // printf("thread %d working on vec = %f \n", r0, vec[dicolP[i]]);
    }

  }
}

mgard_cuda_ret 
prolongate_last_row_cuda(int nrow,       int ncol, 
                         int nr,         int nc,
                         int row_stride, int col_stride,
                         int * dirow,    int * dicolP,
                         double * dv,    int lddv,
                         double * dcoords_x) {

  int B = 16;
  int total_thread_x = ceil((float)nr/row_stride);
  int tbx = min(B, total_thread_x);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  // std::cout << "thread block: " << tbx << std::endl;
  // std::cout << "grid: " << gridx << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _prolongate_last_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                               nr,         nc, 
                                                               row_stride, col_stride,
                                                               dirow,      dicolP,
                                                               dv,         lddv,
                                                               dcoords_x);
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
_prolongate_last_col_cuda(int nrow,       int ncol,
                          int nr,         int nc,
                          int row_stride, int col_stride,
                          int * dirowP,   int * dicol, 
                          double * dv,    int lddv,
                          double * dcoords_y) {
  int c0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  //int y = threadIdx.y * stride;
  //for (int c = c0; c < nc; c += (blockDim.x * gridDim.x) * col_stride) {
  for (int c = c0; c < ncol; c += (blockDim.x * gridDim.x) * col_stride) {
    //printf("thread working on %d \n", x);
    //double * vec = dv + dicol[c];
    double * vec = dv + c;
    for (int i = 0; i < nrow-nr; i++) {
      double h1 = 1; //mgard_common::get_h_cuda(dcoords_y, irowP[i] - 1, 1);
      double h2 = 1; //mgard_common::get_h_cuda(dcoords_y, irowP[i]    , 1);
      double hsum = h1 + h2;
      // printf("thread %d working on vec = %f %f %f \n", c0, vec[(dirowP[i] - 1)*lddv], vec[dirowP[i]*lddv], vec[(dirowP[i] + 1)*lddv]);
      vec[dirowP[i] * lddv] = (h2 * vec[(dirowP[i] - 1) * lddv] + h1 * vec[(dirowP[i] + 1) * lddv]) / hsum;
      // printf("thread %d working on vec = %f \n", c0, vec[dirowP[i] * lddv]);
    }
  }
}

mgard_cuda_ret 
prolongate_last_col_cuda(int nrow,       int ncol,
                         int nr,         int nc,
                         int row_stride, int col_stride,
                         int * dirowP,   int * dicol, 
                         double * dv,    int lddv,
                         double * dcoords_y) {
  
  int B = 16;
  //int total_thread_x = ceil((float)nc/col_stride);
  int total_thread_x = ceil((float)ncol/col_stride);
  int tbx = min(B, total_thread_x);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  // std::cout << "thread block: " << tbx << std::endl;
  // std::cout << "grid: " << gridx << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _prolongate_last_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                                                               nr,         nc,
                                                               row_stride, col_stride,
                                                               dirowP,     dicol,
                                                               dv,         lddv,
                                                               dcoords_y);
  gpuErrchk(cudaGetLastError ());  

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}




void 
refactor_2D_cuda(const int l_target,
                 const int nrow,     const int ncol,
                 const int nr,       const int nc, 
                 int * dirow,        int * dicol,
                 int * dirowP,       int * dicolP,
                 double * dv,        int lddv, 
                 double * dwork,     int lddwork,
                 double * dcoords_x, double * dcoords_y) {
  // refactor
  //    //std::cout  << "I am the general refactorer!" <<"\n";
  
  mgard_cuda_ret ret;

  double pi_Ql_cuda_time = 0.0;
  double copy_level_l_cuda_time = 0.0;
  double assign_num_level_l_cuda_time = 0.0;

  double mass_mult_l_row_cuda_time = 0.0;
  double restriction_l_row_cuda_time = 0.0;
  double solve_tridiag_M_l_row_cuda_time = 0.0;

  double mass_mult_l_col_cuda_time = 0.0;
  double restriction_l_col_cuda_time = 0.0;
  double solve_tridiag_M_l_col_cuda_time = 0.0;
  double add_level_cuda_time = 0.0;

  for (int l = 0; l < l_target; ++l) {
    int stride = std::pow(2, l); // current stride
    int Cstride = stride * 2;    // coarser stride

    // -> change funcs in pi_QL to use _l functions, otherwise distances are
    // wrong!!!
    // pi_Ql(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec,
    //       col_vec); // rename!. v@l has I-\Pi_l Q_l+1 u
    // print_matrix(nrow, ncol, v, ldv);
    int row_stride = stride;
    int col_stride = stride;
    ret = pi_Ql_cuda(nrow,            ncol,
               nr,              nc,
               row_stride,      col_stride,
               dirow,           dicol,
               dv,              lddv, 
               dcoords_x,       dcoords_y);
    pi_Ql_cuda_time += ret.time;

    // pi_Ql(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec,
    //       col_vec); // rename!. v@l has I-\Pi_l Q_l+1 u

    // copy_level_l(l, v, work.data(), nr, nc, nrow, ncol);
    row_stride = stride;
    col_stride = stride;
    ret = copy_level_l_cuda(nrow,       ncol,
                      nr,         nc,
                      row_stride, col_stride,
                      dirow,      dicol,
                      dv,         lddv, 
                      dwork,      lddwork);
    copy_level_l_cuda_time += ret.time;

    // assign_num_level_l(l + 1, work.data(), 0.0, nr, nc, nrow, ncol);
    row_stride = Cstride;
    col_stride = Cstride;
    ret = assign_num_level_l_cuda(nrow,       ncol,
                            nr,         nc,
                            row_stride, col_stride,
                            dirow,      dicol,
                            dwork,      lddwork, 
                            0.0);
    assign_num_level_l_cuda_time += ret.time;

    row_stride = 1;
    col_stride = stride;
    ret = mass_mult_l_row_cuda(nrow,       ncol,
                         nr,         nc,
                         row_stride, col_stride,
                         dirow,      dicol,
                         dwork,      lddwork,
                         dcoords_x);
    mass_mult_l_row_cuda_time += ret.time;


    row_stride = 1;
    col_stride = stride;
    ret = restriction_l_row_cuda(nrow,       ncol,
                           nr,         nc,
                           row_stride, col_stride,
                           dirow,      dicol,
                           dwork,      lddwork,
                           dcoords_x);
    restriction_l_row_cuda_time += ret.time;

    row_stride = 1;
    col_stride = Cstride;
    ret = solve_tridiag_M_l_row_cuda(nrow,       ncol,
                               nr,         nc,
                               row_stride, col_stride,
                               dirow,      dicol,
                               dwork,      lddwork,
                               dcoords_x);
    solve_tridiag_M_l_row_cuda_time += ret.time;

    // row-sweep
    // std::cout << "cpu: ";
    for (int i = 0; i < nr; ++i) {
      int ir = get_lindex_cuda(nr, nrow, i);
      // std::cout << ir << " ";
      for (int j = 0; j < ncol; ++j) {
        //row_vec[j] = work[mgard_common::get_index_cuda(ncol, ir, j)];
      }

      // mgard_gen::mass_mult_l(l, row_vec, coords_x, nc, ncol);

      // mgard_gen::restriction_l(l + 1, row_vec, coords_x, nc, ncol);

      // mgard_gen::solve_tridiag_M_l(l + 1, row_vec, coords_x, nc, ncol);

      for (int j = 0; j < ncol; ++j) {
        //work[mgard_common::get_index_cuda(ncol, ir, j)] = row_vec[j];
      }
    }
    // std::cout << std::endl;

    // column-sweep
    if (nrow > 1) // do this if we have an 2-dimensional array
    {
      // print_matrix(nrow, ncol, work.data(), ldwork);
      row_stride = stride;
      col_stride = Cstride;
      ret = mass_mult_l_col_cuda(nrow,       ncol,
                                 nr,         nc,
                                 row_stride, col_stride,
                                 dirow,      dicol,
                                 dwork,      lddwork,
                                 dcoords_y);
      mass_mult_l_col_cuda_time += ret.time;


      row_stride = stride;
      col_stride = Cstride;
      ret = restriction_l_col_cuda(nrow,       ncol,
                                   nr,         nc,
                                   row_stride, col_stride,
                                   dirow,      dicol,
                                   dwork, lddwork,
                                   dcoords_y);
      restriction_l_col_cuda_time += ret.time;

      row_stride = Cstride;
      col_stride = Cstride;
      ret = solve_tridiag_M_l_col_cuda(nrow,       ncol,
                                       nr,         nc,
                                       row_stride, col_stride,
                                       dirow,       dicol,
                                       dwork, lddwork,
                                       dcoords_y);
      solve_tridiag_M_l_col_cuda_time += ret.time;
      // std::cout << "cpu: ";
      for (int j = 0; j < nc; j += Cstride) {
        int jr = get_lindex_cuda(nc, ncol, j);
        // std::cout << jr << " ";
        for (int i = 0; i < nrow; ++i) {
          //col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, jr)];
        }


        // mgard_gen::mass_mult_l(l, col_vec, coords_y, nr, nrow);
        // mgard_gen::restriction_l(l + 1, col_vec, coords_y, nr, nrow);
        // mgard_gen::solve_tridiag_M_l(l + 1, col_vec, coords_y, nr, nrow);

        for (int i = 0; i < nrow; ++i) {
         // work[mgard_common::get_index_cuda(ncol, i, jr)] = col_vec[i];
        }
      }
      // std::cout<<std::endl;
    }

    // Solved for (z_l, phi_l) = (c_{l+1}, vl)
    // add_level_l(l + 1, v, work.data(), nr, nc, nrow, ncol);
    row_stride = Cstride;
    col_stride = Cstride;
    ret = add_level_l_cuda(nrow,       ncol, 
                     nr,         nc,
                     row_stride, col_stride,
                     dirow,      dicol,
                     dv,         lddv, 
                     dwork,      lddwork);
    add_level_cuda_time += ret.time;
  }

  std::ofstream timing_results;
  timing_results.open ("refactor_2D_cuda.csv");
  timing_results << "pi_Ql_cuda_time," << pi_Ql_cuda_time << std::endl;
  timing_results << "copy_level_l_cuda_time," << copy_level_l_cuda_time << std::endl;
  timing_results << "assign_num_level_l_cuda_time," << assign_num_level_l_cuda_time << std::endl;

  timing_results << "mass_mult_l_row_cuda_time," << mass_mult_l_row_cuda_time << std::endl;
  timing_results << "restriction_l_row_cuda_time," << restriction_l_row_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_row_cuda_time," << solve_tridiag_M_l_row_cuda_time << std::endl;

  timing_results << "mass_mult_l_col_cuda_time," << mass_mult_l_col_cuda_time << std::endl;
  timing_results << "restriction_l_col_cuda_time," << restriction_l_col_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_col_cuda_time," << solve_tridiag_M_l_col_cuda_time << std::endl;
  timing_results << "add_level_cuda_time," << add_level_cuda_time << std::endl;
  timing_results.close();
}

void 
recompose_2D_cuda(const int l_target,
                  const int nrow,     const int ncol,
                  const int nr,       const int nc, 
                  int * dirow,        int * dicol,
                  int * dirowP,       int * dicolP,
                  double * dv,        int lddv, 
                  double * dwork,     int lddwork,
                  double * dcoords_x, double * dcoords_y) {
 
  // recompose
  //    //std::cout  << "recomposing" << "\n";

  mgard_cuda_ret ret;
  double copy_level_l_cuda_time = 0.0;
  double assign_num_level_l_cuda_time = 0.0;

  double mass_mult_l_row_cuda_time = 0.0;
  double restriction_l_row_cuda_time = 0.0;
  double solve_tridiag_M_l_row_cuda_time = 0.0;

  double mass_mult_l_col_cuda_time = 0.0;
  double restriction_l_col_cuda_time = 0.0;
  double solve_tridiag_M_l_col_cuda_time = 0.0;

  double subtract_level_l_cuda_time = 0.0;
  double prolongate_l_row_cuda_time = 0.0;
  double prolongate_l_col_cuda_time = 0.0;

  for (int l = l_target; l > 0; --l) {

    int stride = std::pow(2, l); // current stride
    int Pstride = stride / 2;
    int Cstride = stride * 2;

    // copy_level_l(l - 1, v, work.data(), nr, nc, nrow, ncol);
    int row_stride = Pstride;
    int col_stride = Pstride;
    ret = copy_level_l_cuda(nrow,       ncol,
                      nr,         nc,
                      row_stride, col_stride,
                      dirow,      dicol,
                      dv,         lddv, 
                      dwork,      lddwork);
    copy_level_l_cuda_time = ret.time;

    // assign_num_level_l(l, work.data(), 0.0, nr, nc, nrow, ncol);
    row_stride = stride;
    col_stride = stride;
    ret = assign_num_level_l_cuda(nrow,       ncol,
                            nr,         nc,
                            row_stride, col_stride,
                            dirow,      dicol,
                            dwork,      lddwork, 
                            0.0);
    assign_num_level_l_cuda_time += ret.time;

    //        //std::cout  << "recomposing-rowsweep" << "\n";
    //  l = 0;
    // row-sweep
    row_stride = 1;
    col_stride = Pstride;
    ret = mass_mult_l_row_cuda(nrow,       ncol,
                         nr,         nc,
                         row_stride, col_stride,
                         dirow,      dicol,
                         dwork,      lddwork,
                         dcoords_x);
    mass_mult_l_row_cuda_time += ret.time;


    row_stride = 1;
    col_stride = Pstride;
    ret = restriction_l_row_cuda(nrow,       ncol,
                           nr,         nc,
                           row_stride, col_stride,
                           dirow,      dicol,
                           dwork,      lddwork,
                           dcoords_x);
    restriction_l_row_cuda_time += ret.time;


    row_stride = 1;
    col_stride = stride;
    ret = solve_tridiag_M_l_row_cuda(nrow,       ncol,
                               nr,         nc,
                               row_stride, col_stride,
                               dirow,      dicol,
                               dwork,      lddwork,
                               dcoords_x);
    solve_tridiag_M_l_row_cuda_time += ret.time;

    for (int i = 0; i < nr; ++i) {
      int ir = get_lindex_cuda(nr, nrow, i);
      for (int j = 0; j < ncol; ++j) {
        //row_vec[j] = work[mgard_common::get_index_cuda(ncol, ir, j)];
      }

      // mgard_gen::mass_mult_l(l - 1, row_vec, coords_x, nc, ncol);

      // mgard_gen::restriction_l(l, row_vec, coords_x, nc, ncol);

      // mgard_gen::solve_tridiag_M_l(l, row_vec, coords_x, nc, ncol);

      for (int j = 0; j < ncol; ++j) {
        //work[mgard_common::get_index_cuda(ncol, ir, j)] = row_vec[j];
      }
    }

    //   //std::cout  << "recomposing-colsweep" << "\n";

    // column-sweep, this is the slow one! Need something like column_copy
    if (nrow > 1) // check if we have 1-D array..
    {
      row_stride = Pstride;
      col_stride = stride;
      ret = mass_mult_l_col_cuda(nrow,       ncol,
                           nr,         nc,
                           row_stride, col_stride,
                           dirow,      dicol,
                           dwork,      lddwork,
                           dcoords_y);
      mass_mult_l_col_cuda_time += ret.time;


      row_stride = Pstride;
      col_stride = stride;
      ret = restriction_l_col_cuda(nrow,       ncol,
                             nr,         nc,
                             row_stride, col_stride,
                             dirow,       dicol,
                             dwork, lddwork,
                             dcoords_y);
      restriction_l_col_cuda_time += ret.time;

      row_stride = stride;
      col_stride = stride;
      ret = solve_tridiag_M_l_col_cuda(nrow,       ncol,
                                 nr,         nc,
                                 row_stride, col_stride,
                                 dirow,      dicol,
                                 dwork,      lddwork,
                                 dcoords_y);
      solve_tridiag_M_l_col_cuda_time += ret.time;

      for (int j = 0; j < nc; j += stride) {
        int jr = get_lindex_cuda(nc, ncol, j);
        for (int i = 0; i < nrow; ++i) {
         // col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, jr)];
        }

        // mgard_gen::mass_mult_l(l - 1, col_vec, coords_y, nr, nrow);

        // mgard_gen::restriction_l(l, col_vec, coords_y, nr, nrow);

        // mgard_gen::solve_tridiag_M_l(l, col_vec, coords_y, nr, nrow);

        for (int i = 0; i < nrow; ++i) {
         // work[mgard_common::get_index_cuda(ncol, i, jr)] = col_vec[i];
        }
      }
    }

    // subtract_level_l(l, work.data(), v, nr, nc, nrow, ncol); // do -(Qu - zl)
    row_stride = stride;
    col_stride = stride;
    ret = subtract_level_l_cuda(nrow,       ncol, 
                          nr,         nc,
                          row_stride, col_stride,
                          dirow,      dicol,
                          dwork,      lddwork,
                          dv,         lddv);
    subtract_level_l_cuda_time += ret.time;

    //        //std::cout  << "recomposing-rowsweep2" << "\n";

    //   //int Pstride = stride/2; //finer stride

    //   // row-sweep

    row_stride = stride;
    col_stride = stride;
    ret = prolongate_l_row_cuda(nrow,        ncol, 
                           nr,         nc,
                           row_stride, col_stride,
                           dirow,      dicol,
                           dwork,      lddwork,
                           dcoords_x);
    prolongate_l_row_cuda_time += ret.time;



    for (int i = 0; i < nr; i += stride) {
      int ir = get_lindex_cuda(nr, nrow, i);
      for (int j = 0; j < ncol; ++j) {
        //row_vec[j] = work[mgard_common::get_index_cuda(ncol, ir, j)];
      }

      // mgard_gen::prolongate_l(l, row_vec, coords_x, nc, ncol);

      for (int j = 0; j < ncol; ++j) {
        //work[mgard_common::get_index_cuda(ncol, ir, j)] = row_vec[j];
      }
    }

    //   //std::cout  << "recomposing-colsweep2" << "\n";
    // column-sweep, this is the slow one! Need something like column_copy
    if (nrow > 1) {
      row_stride = stride;
      col_stride = Pstride;
      ret = prolongate_l_col_cuda(nrow,        ncol, 
                             nr,         nc,
                             row_stride, col_stride,
                             dirow,      dicol,
                             dwork,      lddwork,
                             dcoords_y);
      prolongate_l_col_cuda_time += ret.time;

      for (int j = 0; j < nc; j += Pstride) {
        int jr = get_lindex_cuda(nc, ncol, j);
        for (int i = 0; i < nrow; ++i) // copy all rows
        {
          //col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, jr)];
        }

        // mgard_gen::prolongate_l(l, col_vec, coords_y, nr, nrow);

        for (int i = 0; i < nrow; ++i) {
          //work[mgard_common::get_index_cuda(ncol, i, jr)] = col_vec[i];
        }
      }
    }

    // assign_num_level_l(l, v, 0.0, nr, nc, nrow, ncol);
    row_stride = stride;
    col_stride = stride;
    ret = assign_num_level_l_cuda(nrow,       ncol,
                            nr,         nc,
                            row_stride, col_stride,
                            dirow,      dicol,
                            dv,         lddv, 
                            0.0);
    assign_num_level_l_cuda_time += ret.time;

    // subtract_level_l(l - 1, v, work.data(), nr, nc, nrow, ncol);

    row_stride = Pstride;
    col_stride = Pstride;
    ret = subtract_level_l_cuda(nrow,       ncol, 
                          nr,         nc,
                          row_stride, col_stride,
                          dirow,      dicol,
                          dv,         lddv,
                          dwork,      lddwork);
    subtract_level_l_cuda_time += ret.time;
  }

  std::ofstream timing_results;
  timing_results.open ("recompose_2D_cuda.csv");
  timing_results << "copy_level_l_cuda_time," << copy_level_l_cuda_time << std::endl;
  timing_results << "assign_num_level_l_cuda_time," << assign_num_level_l_cuda_time << std::endl;

  timing_results << "mass_mult_l_row_cuda_time," << mass_mult_l_row_cuda_time << std::endl;
  timing_results << "restriction_l_row_cuda_time," << restriction_l_row_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_row_cuda_time," << solve_tridiag_M_l_row_cuda_time << std::endl;

  timing_results << "mass_mult_l_col_cuda_time," << mass_mult_l_col_cuda_time << std::endl;
  timing_results << "restriction_l_col_cuda_time," << restriction_l_col_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_col_cuda_time," << solve_tridiag_M_l_col_cuda_time << std::endl;

  timing_results << "subtract_level_l_cuda_time," << subtract_level_l_cuda_time << std::endl;
  timing_results << "prolongate_l_row_cuda_time," << prolongate_l_row_cuda_time << std::endl;
  timing_results << "prolongate_l_col_cuda_time," << prolongate_l_col_cuda_time << std::endl;
  timing_results.close();

}


void 
postp_2D_cuda(const int nrow,     const int ncol,
              const int nr,       const int nc, 
              int * dirow,        int * dicol,
              int * dirowP,       int * dicolP,
              double * dv,        int lddv, 
              double * dwork,     int lddwork,
              double * dcoords_x, double * dcoords_y) {


  mgard_cuda_ret ret;
  double copy_level_cuda_time = 0.0;
  double assign_num_level_l_cuda_time = 0.0;

  double mass_matrix_multiply_row_cuda_time = 0.0;
  double restriction_first_row_cuda_time = 0.0;
  double solve_tridiag_M_l_row_cuda_time = 0.0;

  double mass_matrix_multiply_col_cuda_time = 0.0;
  double restriction_first_col_cuda_time = 0.0;
  double solve_tridiag_M_l_col_cuda_time = 0.0;

  double subtract_level_l_cuda_time = 0.0;
  double prolongate_last_row_cuda_time = 0.0;
  double prolongate_last_col_cuda_time = 0.0;

  double subtract_level_cuda_time = 0.0;

 // mgard_cannon::copy_level(nrow, ncol, 0, v, work);
  int row_stride = 1;
  int col_stride = 1;
  ret = mgard_cannon::copy_level_cuda(nrow,       ncol, 
                                row_stride, col_stride,
                                dv,         lddv,
                                dwork,      lddwork);
  copy_level_cuda_time += ret.time;

  // assign_num_level_l(0, work.data(), 0.0, nr, nc, nrow, ncol);

  row_stride = 1;
  col_stride = 1;
  ret = assign_num_level_l_cuda(nrow,       ncol,
                          nr,         nc,
                          row_stride, col_stride,
                          dirow,      dicol,
                          dwork,      lddwork,
                          0.0);
  assign_num_level_l_cuda_time += ret.time;

  row_stride = 1;
  col_stride = 1;
  ret = mgard_cannon::mass_matrix_multiply_row_cuda(nrow,       ncol, 
                                              row_stride, col_stride,
                                              dwork,      lddwork,
                                              dcoords_x);
  mass_matrix_multiply_row_cuda_time += ret.time;


  row_stride = 1;
  col_stride = 1;
  ret = restriction_first_row_cuda(nrow,       ncol, 
                             row_stride, dicolP, nc,
                             dwork,      lddwork,
                             dcoords_x);
  restriction_first_row_cuda_time += ret.time;

  row_stride = 1;
  col_stride = 1;
  ret = solve_tridiag_M_l_row_cuda(nrow,       ncol,
                             nr,         nc,
                             row_stride, col_stride,
                             dirow,      dicol,
                             dwork,      lddwork,
                             dcoords_x);
  solve_tridiag_M_l_row_cuda_time += ret.time;

  for (int i = 0; i < nrow; ++i) {
    int ir = get_lindex_cuda(nr, nrow, i);
    for (int j = 0; j < ncol; ++j) {
      //row_vec[j] = work[mgard_common::get_index_cuda(ncol, i, j)];
    }

    // mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);

    // restriction_first(row_vec, coords_x, nc, ncol);

    for (int j = 0; j < ncol; ++j) {
     // work[mgard_common::get_index_cuda(ncol, i, j)] = row_vec[j];
    }
  }

  for (int i = 0; i < nr; ++i) {
    int ir = get_lindex_cuda(nr, nrow, i);
    for (int j = 0; j < ncol; ++j) {
      //row_vec[j] = work[mgard_common::get_index_cuda(ncol, ir, j)];
    }

    // mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol);

    for (int j = 0; j < ncol; ++j) {
     // work[mgard_common::get_index_cuda(ncol, ir, j)] = row_vec[j];
    }
  }

  //   //   //std::cout  << "recomposing-colsweep" << "\n";

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) // check if we have 1-D array..
  {
    
    row_stride = 1;
    col_stride = 1;
    ret = mgard_cannon::mass_matrix_multiply_col_cuda(nrow,      ncol,
                                               row_stride, col_stride,
                                               dwork,      lddwork,
                                               dcoords_y);
    mass_matrix_multiply_col_cuda_time += ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = restriction_first_col_cuda(nrow,   ncol, 
                               dirowP, nr,   col_stride,
                               dwork,  lddwork,
                               dcoords_y);
    restriction_first_col_cuda_time += ret.time;

    row_stride = 1;
    col_stride = 1;
    ret = solve_tridiag_M_l_col_cuda(nrow,       ncol,
                               nr,         nc,
                               row_stride, col_stride,
                               dirow,      dicol,
                               dwork,      lddwork,
                               dcoords_y);
    solve_tridiag_M_l_col_cuda_time += ret.time;

    for (int j = 0; j < ncol; ++j) {
      int jr  = get_lindex_cuda(nc,  ncol,  j);
      for (int i = 0; i < nrow; ++i) {
       // col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, j)];
      }

      // mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);

      // mgard_gen::restriction_first(col_vec, coords_y, nr, nrow);

      for (int i = 0; i < nrow; ++i) {
       // work[mgard_common::get_index_cuda(ncol, i, j)] = col_vec[i];
      }
    }

    for (int j = 0; j < nc; ++j) {
      int jr = get_lindex_cuda(nc, ncol, j);
      for (int i = 0; i < nrow; ++i) {
        //col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, jr)];
      }

      // mgard_gen::solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow);
      for (int i = 0; i < nrow; ++i) {
        //work[mgard_common::get_index_cuda(ncol, i, jr)] = col_vec[i];
      }
    }
  }

  // subtract_level_l(0, work.data(), v, nr, nc, nrow, ncol); // do -(Qu - zl)
  row_stride = 1;
  col_stride = 1;
  ret = subtract_level_l_cuda(nrow,       ncol, 
                        nr,         nc,
                        row_stride, col_stride,
                        dirow,      dicol,
                        dwork,      lddwork,
                        dv,         lddv);
  subtract_level_l_cuda_time += ret.time;


  //        //std::cout  << "recomposing-rowsweep2" << "\n";

  //     //   //int Pstride = stride/2; //finer stride
  
  row_stride = 1;
  col_stride = 1;
  ret = prolongate_last_row_cuda(nrow,       ncol, 
                           nr,         nc,
                           row_stride, col_stride,
                           dirow,      dicolP,
                           dwork,      lddwork,
                           dcoords_x);
  prolongate_last_row_cuda_time += ret.time;

  //   //   // row-sweep
  // std::cout << "cpu: ";
  for (int i = 0; i < nr; ++i) {
    int ir = get_lindex_cuda(nr, nrow, i);
    // std::cout << ir << " ";
    for (int j = 0; j < ncol; ++j) {
      //row_vec[j] = work[mgard_common::get_index_cuda(ncol, ir, j)];
    }

    // mgard_gen::prolongate_last(row_vec, coords_x, nc, ncol);

    for (int j = 0; j < ncol; ++j) {
     // work[mgard_common::get_index_cuda(ncol, ir, j)] = row_vec[j];
    }
  }
  // std::cout << std::endl;

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) {
    // print_matrix(nrow, ncol, work.data(), ldwork);
    row_stride = 1;
    col_stride = 1;
    ret = prolongate_last_col_cuda(nrow,       ncol, 
                             nr,         nc,
                             row_stride, col_stride,
                             dirowP,     dicol,
                             dwork,      lddwork,
                             dcoords_y);
    prolongate_last_col_cuda_time += ret.time;
    // print_matrix(nrow, ncol, work.data(), ldwork);


    for (int j = 0; j < ncol; ++j) {
      int jr  = get_lindex_cuda(nc,  ncol,  j);
      for (int i = 0; i < nrow; ++i) // copy all rows
      {
       // col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, j)];
      }

      // mgard_gen::prolongate_last(col_vec, coords_y, nr, nrow);

      for (int i = 0; i < nrow; ++i) {
       // work[mgard_common::get_index_cuda(ncol, i, j)] = col_vec[i];
      }
    }
  }
  // print_matrix(nrow, ncol, work.data(), ldwork);
  


  // assign_num_level_l(0, v, 0.0, nr, nc, nrow, ncol);

  ret = assign_num_level_l_cuda(nrow,       ncol,
                          nr,         nc,
                          row_stride, col_stride,
                          dirow,      dicol,
                          dv,         lddv,
                          0.0);
  assign_num_level_l_cuda_time += ret.time;

  // mgard_cannon::subtract_level(nrow, ncol, 0, v, work.data());
  row_stride = 1;
  col_stride = 1;
  ret = mgard_cannon::subtract_level_cuda(nrow,       ncol, 
                                    row_stride, col_stride,
                                    dv,         lddv, 
                                    dwork,      lddwork); 
  subtract_level_cuda_time += ret.time;

  std::ofstream timing_results;
  timing_results.open ("postp_2D_cuda.csv");
  timing_results << "copy_level_cuda_time," << copy_level_cuda_time << std::endl;
  timing_results << "assign_num_level_l_cuda_time," << assign_num_level_l_cuda_time << std::endl;

  timing_results << "mass_matrix_multiply_row_cuda_time," << mass_matrix_multiply_row_cuda_time << std::endl;
  timing_results << "restriction_first_row_cuda_time," << restriction_first_row_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_row_cuda_time," << solve_tridiag_M_l_row_cuda_time << std::endl;

  timing_results << "mass_matrix_multiply_col_cuda_time," << mass_matrix_multiply_col_cuda_time << std::endl;
  timing_results << "restriction_first_col_cuda_time," << restriction_first_col_cuda_time << std::endl;
  timing_results << "solve_tridiag_M_l_col_cuda_time," << solve_tridiag_M_l_col_cuda_time << std::endl;

  timing_results << "subtract_level_l_cuda_time," << subtract_level_l_cuda_time << std::endl;
  timing_results << "prolongate_last_row_cuda_time," << prolongate_last_row_cuda_time << std::endl;
  timing_results << "prolongate_last_col_cuda_time," << prolongate_last_col_cuda_time << std::endl;

  timing_results << "subtract_level_cuda_time," << subtract_level_cuda_time << std::endl;
  timing_results.close();
}





} //end namespace mgard_gen

} //end namespace mard_2d