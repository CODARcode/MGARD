#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"

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



void copy_level_cuda(int nrow,       int ncol, 
                     int row_stride, int col_stride,
                     double * v,     int ldv,
                     double * work,  int ldwork) {

  // int stride = std::pow(2, l); // current stride
  // int row_stride = std::pow(2, l); // current stride
  // int col_stride = std::pow(2, l); // current stride

  int ldv = nrow;
  int ldwork = nrow;

  double * dv;
  int lddv;
  double * dwork;
  int lddwork;

  size_t dv_pitch;
  cudaMallocPitch(&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2D(dv, lddv * sizeof(double), 
               v,     ldv  * sizeof(double), 
               ncol * sizeof(double), nrow, 
               cudaMemcpyHostToDevice);

  size_t dwork_pitch;
  cudaMallocPitch(&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
  lddwork = dwork_pitch / sizeof(double);
  cudaMemcpy2D(dwork, lddwork * sizeof(double), 
               work,  ldwork  * sizeof(double), 
               ncol * sizeof(double), nrow, 
               cudaMemcpyHostToDevice);

  int B = 16;
  int total_thread_y = ceil((double)nrow/row_stride);
  int total_thread_x = ceil((double)ncol/col_stride);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil(total_thread_y/tbx);
  int gridx = ceil(total_thread_x/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  _copy_level_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                                                       row_stride, col_stride, 
                                                       dv,         lddv, 
                                                       dwork,      lddwork);
	gpuErrchk(cudaGetLastError ()); 

	cudaMemcpy2D(work,  ldwork * sizeof(double), 
               dwork, lddwork * sizeof(double),
               ncol * sizeof(double), nrow, 
               cudaMemcpyDeviceToHost);

  // for (int irow = 0; irow < nrow; irow += stride) {
  //   for (int jcol = 0; jcol < ncol; jcol += stride) {
  //     work[mgard_common::get_index_cuda(ncol, irow, jcol)] =
  //         v[mgard_common::get_index_cuda(ncol, irow, jcol)];
  //   }
  // }
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
    printf("thread %d working on %f\n", idx, temp1);
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


// void mass_matrix_multiply_cuda(const int l, std::vector<double> &v,
//                           const std::vector<double> &coords) {
// 	int stride = std::pow(2, l); // current stride
//   int n = v.size();
//   double temp1, temp2;

//   // Mass matrix times nodal value-vec
//   temp1 = v.front(); // save u(0) for later use
//   v.front() = 2.0 * mgard_common::get_h_cuda(coords.data(), 0, stride) * temp1 +
//               mgard_common::get_h_cuda(coords.data(), 0, stride) * v[stride];
//   for (int i = stride; i <= n - 1 - stride; i += stride) {
//     temp2 = v[i];
//     v[i] = mgard_common::get_h_cuda(coords.data(), i - stride, stride) * temp1 +
//            2 *
//                (mgard_common::get_h_cuda(coords.data(), i - stride, stride) +
//                 mgard_common::get_h_cuda(coords.data(), i, stride)) *
//                temp2 +
//            mgard_common::get_h_cuda(coords.data(), i, stride) * v[i + stride];
//     temp1 = temp2; // save u(n) for later use
//   }
//   v[n - 1] = mgard_common::get_h_cuda(coords.data(), n - stride - 1, stride) * temp1 +
//              2 * mgard_common::get_h_cuda(coords.data(), n - stride - 1, stride) * v[n - 1];
// }


void mass_matrix_multiply_row_cuda(int nrow,       int ncol, 
															     int row_stride, int col_stride,
                                   double * v,    int ldv,
                                   double * coords_x) {
  //print_matrix(nrow, ncol, v, nrow);
  double * dv;
  int lddv;

  size_t dv_pitch;
	cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
	lddv = dv_pitch / sizeof(double);
	cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
	             v,     ldv  * sizeof(double), 
	             ncol * sizeof(double), nrow, 
	             H2D);

	double * dcoords_x;
	cudaMallocHelper((void**)&dcoords_x, ncol * sizeof(double));
	cudaMemcpyHelper(dcoords_x, coords_x, ncol * sizeof(double), H2D);

  int B = 16;

	int total_thread = ceil((float)nrow/row_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  std::cout << "thread block: " << tb << std::endl;
  std::cout << "grid: " << grid << std::endl;

  _mass_matrix_multiply_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                              																			row_stride, col_stride,
                              																			dv,         lddv,
                              																			dcoords_x);
  gpuErrchk(cudaGetLastError ()); 

  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
			  					   dv,    lddv * sizeof(double), 
				             ncol * sizeof(double), nrow, 
				             D2H);
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


void mass_matrix_multiply_col_cuda(int nrow,       int ncol, 
															     int row_stride, int col_stride,
                                   double * v,    int ldv,
                                   double * coords_y) {
  //print_matrix(nrow, ncol, v, nrow);
  double * dv;
  int lddv;

  size_t dv_pitch;
	cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
	lddv = dv_pitch / sizeof(double);
	cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
	             v,     ldv  * sizeof(double), 
	             ncol * sizeof(double), nrow, 
	             H2D);

	double * dcoords_y;
	cudaMallocHelper((void**)&dcoords_y, nrow * sizeof(double));
	cudaMemcpyHelper(dcoords_y, coords_y, nrow * sizeof(double), H2D);

  int B = 16;

	int total_thread = ceil((float)ncol/col_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  std::cout << "thread block: " << tb << std::endl;
  std::cout << "grid: " << grid << std::endl;

  _mass_matrix_multiply_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                              																			row_stride, col_stride,
                              																			dv,         lddv,
                              																			dcoords_y);
  gpuErrchk(cudaGetLastError ()); 

  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
			  					   dv,    lddv * sizeof(double), 
				             ncol * sizeof(double), nrow, 
				             D2H);
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
_substract_level_cuda(int nrow,       int ncol, 
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

void substract_level_cuda(int nrow,       int ncol, 
                          int row_stride, int col_stride,
                          double * v,    int ldv, 
                          double * work, int ldwork) {
  double * dv;
  int lddv;

  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                      v,  ldv  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);

  double * dwork;
  int lddwork;

  size_t dwork_pitch;
  cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
  lddwork = dwork_pitch / sizeof(double);
  cudaMemcpy2DHelper(dwork, lddwork * sizeof(double), 
                      work,  ldwork  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);


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
  _substract_level_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                           row_stride, col_stride, 
                                                           dv,         lddv,
                                                           dwork,      lddwork);


  gpuErrchk(cudaGetLastError ()); 

  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
                     dv,    lddv * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);

  cudaMemcpy2DHelper(work,     ldwork  * sizeof(double), 
                     dwork,    lddwork * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);
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
_pi_Ql_first_row_cuda(const int nr, const int nc,
									   const int nrow, const int ncol,
									   int * irow, int * icolP,
									   double * coords_x, double * coords_y,
									   double * dv, int lddv) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < ncol-nc && y < nr) {
    int r = irow[y];
    int c = icolP[x];
    //printf ("thread (%d, %d) working on (%d, %d): %f\n", y, x, r, c, dv[mgard_common::get_index_cuda(lddv, r, c    )]);
    register double center = dv[mgard_common::get_index_cuda(lddv, r, c    )];
    register double left   = dv[mgard_common::get_index_cuda(lddv, r, c - 1)];
    register double right  = dv[mgard_common::get_index_cuda(lddv, r, c + 1)];
    register double h1     = mgard_common::get_dist_cuda(coords_x, c - 1, c    );
    register double h2     = mgard_common::get_dist_cuda(coords_x, c,     c + 1);

    center -= (h2 * left + h1 * right) / (h1 + h2);

    dv[mgard_common::get_index_cuda(lddv, r, c    )] = center;
	}

}


__global__ void 
_pi_Ql_first_col_cuda(const int nr, const int nc,
									   const int nrow, const int ncol,
									   int * irowP, int * icol,
									   double * coords_x, double * coords_y,
									   double * dv, int lddv) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < ncol && y < nrow-nr) {
    int r = irowP[y];
    int c = icol[x];
    //printf ("thread (%d, %d) working on (%d, %d): %f\n", y, x, r, c, dv[mgard_common::get_index_cuda(lddv, r, c    )]);
    register double center = dv[mgard_common::get_index_cuda(lddv, r,     c)];
    register double up   = dv[mgard_common::get_index_cuda(lddv,   r - 1, c)];
    register double down  = dv[mgard_common::get_index_cuda(lddv, r + 1, c)];
    register double h1     = mgard_common::get_dist_cuda(coords_x, r - 1, r    );
    register double h2     = mgard_common::get_dist_cuda(coords_x, r,     r + 1);

    center -= (h2 * up + h1 * down) / (h1 + h2);

    dv[mgard_common::get_index_cuda(lddv, r, c    )] = center;
	}

}


__global__ void 
_pi_Ql_first_center_cuda(const int nr, const int nc,
										   const int nrow, const int ncol,
										   int * dirowP, int * dicolP,
										   double * dcoords_x, double * dcoords_y,
										   double * dv, int lddv) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < ncol-nc && y < nrow-nr) {
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

    register double x2 = mgard_common::get_dist_cuda(dcoords_x, c - 1, c + 1);
    register double y2 = mgard_common::get_dist_cuda(dcoords_y, r - 1, r + 1);

    register double x = mgard_common::get_dist_cuda(dcoords_x, c, c + 1);
    register double y = mgard_common::get_dist_cuda(dcoords_y, r, r + 1);

    double temp =
            mgard_common::interp_2d_cuda(upleft, downleft, upright, downright, x1, x2, y1, y2, x, y);

    center -= temp;

    dv[mgard_common::get_index_cuda(lddv, r, c    )] = center;
	}

}



void pi_lminus1_first_cuda(std::vector<double> &v, const std::vector<double> &coords,
                      int n, int no) {

  for (int i = 0; i < n - 1; ++i) {
    int i_logic = get_lindex_cuda(n, no, i);
    int i_logicP = get_lindex_cuda(n, no, i + 1);

    if (i_logicP != i_logic + 1) {
      //          //std::cout  << i_logic +1 << "\t" << i_logicP<<"\n";
      double h1 = mgard_common::get_dist_cuda(coords.data(), i_logic, i_logic + 1);
      double h2 = mgard_common::get_dist_cuda(coords.data(), i_logic + 1, i_logicP);
      double hsum = h1 + h2;
      //printf("%f\n",v[i_logic + 1]);
      v[i_logic + 1] -= (h2 * v[i_logic] + h1 * v[i_logicP]) / hsum;

    }
  }
  //printf("\n");
}


void pi_Ql_first_cuda(const int nr, const int nc, const int nrow, const int ncol,
                 const int l, double *v, const std::vector<double> &coords_x,
                 const std::vector<double> &coords_y,
                 std::vector<double> &row_vec, std::vector<double> &col_vec) {
  // Restrict data to coarser level
  int ldv = nrow;

  //print_matrix(nrow, ncol, v, nrow);

  double * dv;
  int lddv;

  size_t dv_pitch;
	cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
	lddv = dv_pitch / sizeof(double);
	cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
	             v,     ldv  * sizeof(double), 
	             ncol * sizeof(double), nrow, 
	             H2D);

	double * dcoords_x;
	cudaMallocHelper((void**)&dcoords_x, ncol * sizeof(double));
	cudaMemcpyHelper(dcoords_x, coords_x.data(), ncol * sizeof(double), H2D);

	double * dcoords_y;
	cudaMallocHelper((void**)&dcoords_y, nrow * sizeof(double));
	cudaMemcpyHelper(dcoords_y, coords_y.data(), nrow * sizeof(double), H2D);	

	int * irow_idx  = new int[nr];
	int * irowP_idx = new int[nrow-nr];
	int irow_idx_ptr  = 0;
	int irowP_idx_ptr = 0;

	for (int irow = 0; irow < nr; irow++) {
		int irow_r = get_lindex_cuda(nr, nrow, irow);
		irow_idx[irow_idx_ptr] = irow_r;
		if (irow_idx_ptr > 0 && irow_idx[irow_idx_ptr - 1] != irow_idx[irow_idx_ptr] - 1) {
			irowP_idx[irowP_idx_ptr] = irow_idx[irow_idx_ptr] - 1;
			irowP_idx_ptr ++;
		} 
		irow_idx_ptr++;
	}

	// std::cout << "irow_idx: ";
	// for (int i = 0; i < nr; i ++) std::cout << irow_idx[i] << ", ";
	// std::cout << std::endl;

	// std::cout << "irowP_idx: ";
	// for (int i = 0; i < nrow-nr; i ++) std::cout << irowP_idx[i] << ", ";
	// std::cout << std::endl;


	int * icol_idx  = new int[nc];
	int * icolP_idx = new int[ncol-nc];
	int icol_idx_ptr  = 0;
	int icolP_idx_ptr = 0;

	for (int icol = 0; icol < nc; icol++) {
		int icol_r = get_lindex_cuda(nc, ncol, icol);
		icol_idx[icol_idx_ptr] = icol_r;
		if (icol_idx_ptr > 0 && icol_idx[icol_idx_ptr - 1] != icol_idx[icol_idx_ptr] - 1) {
			icolP_idx[icolP_idx_ptr] = icol_idx[icol_idx_ptr] - 1;
			icolP_idx_ptr ++;
		} 
		icol_idx_ptr++;
	}

	// std::cout << "icol_idx: ";
	// for (int i = 0; i < nc; i ++) std::cout << icol_idx[i] << ", ";
	// std::cout << std::endl;

	// std::cout << "icolP_idx: ";
	// for (int i = 0; i < ncol-nc; i ++) std::cout << icolP_idx[i] << ", ";
	// std::cout << std::endl;

	

	int * dirow_idx;
	cudaMallocHelper((void**)&dirow_idx, nr * sizeof(int));
	cudaMemcpyHelper(dirow_idx, irow_idx, nr * sizeof(int), H2D);

	int * dirowP_idx;
	cudaMallocHelper((void**)&dirowP_idx, (nrow-nr) * sizeof(int));
	cudaMemcpyHelper(dirowP_idx, irowP_idx, (nrow-nr) * sizeof(int), H2D);	


	int * dicol_idx;
	cudaMallocHelper((void**)&dicol_idx, nc * sizeof(int));
	cudaMemcpyHelper(dicol_idx, icol_idx, nc * sizeof(int), H2D);

	int * dicolP_idx;
	cudaMallocHelper((void**)&dicolP_idx, (ncol-nc) * sizeof(int));
	cudaMemcpyHelper(dicolP_idx, icolP_idx, (ncol-nc) * sizeof(int), H2D);	
  
	int B = 16;

	int total_thread_x = ncol-nc;
  int total_thread_y = nr;
  int tbx = min(B, total_thread_x);
  int tby = min(B, total_thread_y);
  int gridx = ceil((float)total_thread_x/tbx);
  int gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
  std::cout << "grid: " << gridx << ", " << gridy <<std::endl;
  _pi_Ql_first_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nr, nc,
																							         nrow, ncol,
																							   			 dirow_idx, dicolP_idx,
																							   			 dcoords_x, dcoords_y,
																							         dv, lddv);
  gpuErrchk(cudaGetLastError ()); 


  total_thread_x = nc;
  total_thread_y = nrow-nr;
  tbx = min(B, total_thread_x);
  tby = min(B, total_thread_y);
  gridx = ceil((float)total_thread_x/tbx);
  gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock2(tbx, tby);
  dim3 blockPerGrid2(gridx, gridy);
  _pi_Ql_first_col_cuda<<<blockPerGrid2, threadsPerBlock2>>>(nr, nc,
																									         nrow, ncol,
																									   			 dirowP_idx, dicol_idx,
																									   			 dcoords_x, dcoords_y,
																									         dv, lddv);
  gpuErrchk(cudaGetLastError ()); 


  total_thread_x = ncol-nc;
  total_thread_y = nrow-nr;
  tbx = min(B, total_thread_x);
  tby = min(B, total_thread_y);
  gridx = ceil((float)total_thread_x/tbx);
  gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock3(tbx, tby);
  dim3 blockPerGrid3(gridx, gridy);
  _pi_Ql_first_center_cuda<<<blockPerGrid3, threadsPerBlock3>>>(nr, nc,
                                                                nrow,       ncol,
                                                                dirowP_idx, dicolP_idx,
                                                                dcoords_x,  dcoords_y,
                                                                dv,         lddv);
  gpuErrchk(cudaGetLastError ()); 


  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
  					   dv,    lddv * sizeof(double), 
	             ncol * sizeof(double), nrow, 
	             D2H);




  int stride = 1; // current stride
  //  int Pstride = stride/2; //finer stride
  //    int Cstride = 2; // coarser stride

  for (int irow = 0; irow < nr; irow += stride) // Do the rows existing  in the coarser level
  {
    int irow_r = get_lindex_cuda(
        nr, nrow, irow); // get the real location of logical index irow

    for (int jcol = 0; jcol < ncol; ++jcol) {
      // int jcol_r = get_lindex_cuda(nc, ncol, jcol);
      // std::cerr << irow_r << "\t"<< jcol_r << "\n";

      row_vec[jcol] = v[mgard_common::get_index_cuda(ncol, irow_r, jcol)];
    }

    //pi_lminus1_first_cuda(row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      //            int jcol_r = get_lindex_cuda(nc, ncol, jcol);
      v[mgard_common::get_index_cuda(ncol, irow_r, jcol)] = row_vec[jcol];
    }

    // if( irP != ir +1) //are we skipping the next row?
    //   {
    //     ++irow;
    //   }
  }

  if (nrow > 1) {
    for (int jcol = 0; jcol < nc;
         jcol += stride) // Do the columns existing  in the coarser level
    {
      int jcol_r = get_lindex_cuda(nc, ncol, jcol);
      //            int jr  = get_lindex_cuda(nc, ncol, jcol);
      // int jrP = get_lindex_cuda(nc, ncol, jcol+1);

      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = v[mgard_common::get_index_cuda(ncol, irow, jcol_r)];
      }

      //pi_lminus1_first_cuda(col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        v[mgard_common::get_index_cuda(ncol, irow, jcol_r)] = col_vec[irow];
      }
    }
  }

  //        Now the new-new stuff
  for (int irow = 0; irow < nr - 1; ++irow) {
    int ir = get_lindex_cuda(nr, nrow, irow);
    int irP = get_lindex_cuda(nr, nrow, irow + 1);

    for (int jcol = 0; jcol < nc - 1; ++jcol) {
      int jr = get_lindex_cuda(nc, ncol, jcol);
      int jrP = get_lindex_cuda(nc, ncol, jcol + 1);

      if ((irP != ir + 1) &&
          (jrP != jr + 1)) // we skipped both a row and a column
      {

        double q11 = v[mgard_common::get_index_cuda(ncol, ir, jr)];
        double q12 = v[mgard_common::get_index_cuda(ncol, irP, jr)];
        double q21 = v[mgard_common::get_index_cuda(ncol, ir, jrP)];
        double q22 = v[mgard_common::get_index_cuda(ncol, irP, jrP)];

        double x1 = 0.0;
        double y1 = 0.0;

        double x2 = mgard_common::get_dist_cuda(coords_x.data(), jr, jrP);
        double y2 = mgard_common::get_dist_cuda(coords_y.data(), ir, irP);

        double x = mgard_common::get_dist_cuda(coords_x.data(), jr, jr + 1);
        double y = mgard_common::get_dist_cuda(coords_y.data(), ir, ir + 1);

        double temp =
            mgard_common::interp_2d_cuda(q11, q12, q21, q22, x1, x2, y1, y2, x, y);

        //v[mgard_common::get_index_cuda(ncol, ir + 1, jr + 1)] -= temp;
      }
    }
  }
}

// __global__ void 
// _assign_num_level_l_cuda(const int nr, const int nc,
// 										     const int nrow, const int ncol,
// 										     int * dirow, int * dicol,
// 										     double * dv, int lddv, double num) {

// 	int x = blockIdx.x * blockDim.x + threadIdx.x;
//   int y = blockIdx.y * blockDim.y + threadIdx.y;

//   if (x < nc && y < nr) {
//     int r = dirow[y];
//     int c = dicol[x];
//     dv[mgard_common::get_index_cuda(lddv, r,     c    )] = num;
//   }



// }


// void assign_num_level_l_cuda(const int l, double *v, double num, int nr, int nc,
//                         const int nrow, const int ncol) {
//   // set the value of nodal values at level l to number num

//   int stride = std::pow(2, l); // current stride


//   int ldv = nrow;

//   print_matrix(nrow, ncol, v, nrow);

//   double * dv;
//   int lddv;

//   size_t dv_pitch;
// 	cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
// 	lddv = dv_pitch / sizeof(double);
// 	cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
// 	             v,     ldv  * sizeof(double), 
// 	             ncol * sizeof(double), nrow, 
// 	             H2D);

// 	int * irow_idx  = new int[nr];
// 	int * irowP_idx = new int[nrow-nr];
// 	int irow_idx_ptr  = 0;
// 	int irowP_idx_ptr = 0;

// 	for (int irow = 0; irow < nr; irow++) {
// 		int irow_r = get_lindex_cuda(nr, nrow, irow);
// 		irow_idx[irow_idx_ptr] = irow_r;
// 		if (irow_idx_ptr > 0 && irow_idx[irow_idx_ptr - 1] != irow_idx[irow_idx_ptr] - 1) {
// 			irowP_idx[irowP_idx_ptr] = irow_idx[irow_idx_ptr] - 1;
// 			irowP_idx_ptr ++;
// 		} 
// 		irow_idx_ptr++;
// 	}

// 	std::cout << "irow_idx: ";
// 	for (int i = 0; i < nr; i ++) std::cout << irow_idx[i] << ", ";
// 	std::cout << std::endl;

// 	std::cout << "irowP_idx: ";
// 	for (int i = 0; i < nrow-nr; i ++) std::cout << irowP_idx[i] << ", ";
// 	std::cout << std::endl;


// 	int * icol_idx  = new int[nc];
// 	int * icolP_idx = new int[ncol-nc];
// 	int icol_idx_ptr  = 0;
// 	int icolP_idx_ptr = 0;

// 	for (int icol = 0; icol < nc; icol++) {
// 		int icol_r = get_lindex_cuda(nc, ncol, icol);
// 		icol_idx[icol_idx_ptr] = icol_r;
// 		if (icol_idx_ptr > 0 && icol_idx[icol_idx_ptr - 1] != icol_idx[icol_idx_ptr] - 1) {
// 			icolP_idx[icolP_idx_ptr] = icol_idx[icol_idx_ptr] - 1;
// 			icolP_idx_ptr ++;
// 		} 
// 		icol_idx_ptr++;
// 	}

// 	std::cout << "icol_idx: ";
// 	for (int i = 0; i < nc; i ++) std::cout << icol_idx[i] << ", ";
// 	std::cout << std::endl;

// 	std::cout << "icolP_idx: ";
// 	for (int i = 0; i < ncol-nc; i ++) std::cout << icolP_idx[i] << ", ";
// 	std::cout << std::endl;

	

// 	int * dirow_idx;
// 	cudaMallocHelper((void**)&dirow_idx, nr * sizeof(int));
// 	cudaMemcpyHelper(dirow_idx, irow_idx, nr * sizeof(int), H2D);

// 	int * dirowP_idx;
// 	cudaMallocHelper((void**)&dirowP_idx, (nrow-nr) * sizeof(int));
// 	cudaMemcpyHelper(dirowP_idx, irowP_idx, (nrow-nr) * sizeof(int), H2D);	


// 	int * dicol_idx;
// 	cudaMallocHelper((void**)&dicol_idx, nc * sizeof(int));
// 	cudaMemcpyHelper(dicol_idx, icol_idx, nc * sizeof(int), H2D);

// 	int * dicolP_idx;
// 	cudaMallocHelper((void**)&dicolP_idx, (ncol-nc) * sizeof(int));
// 	cudaMemcpyHelper(dicolP_idx, icolP_idx, (ncol-nc) * sizeof(int), H2D);	

// 	int B = 16;

// 	int total_thread_x = nc;
//   int total_thread_y = nr;
//   int tbx = min(B, total_thread_x);
//   int tby = min(B, total_thread_y);
//   int gridx = ceil((float)total_thread_x/tbx);
//   int gridy = ceil((float)total_thread_y/tby);
//   dim3 threadsPerBlock(tbx, tby);
//   dim3 blockPerGrid(gridx, gridy);


//   _assign_num_level_l_cuda<<<blockPerGrid, threadsPerBlock>>>(nr, nc,
//   																														nrow, ncol,
//   																														dirow_idx, dicol_idx,
//   																														dv, lddv, num);
//   gpuErrchk(cudaGetLastError ()); 



// 	cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
//   					   dv,    lddv * sizeof(double), 
// 	             ncol * sizeof(double), nrow, 
// 	             D2H);



//   // for (int irow = 0; irow < nr; irow += stride) {
//   //   int ir = get_lindex_cuda(nr, nrow, irow);
//   //   for (int jcol = 0; jcol < nc; jcol += stride) {
//   //     int jr = get_lindex_cuda(nc, ncol, jcol);
//   //     v[mgard_common::get_index_cuda(ncol, ir, jr)] = num;
//   //   }
//   // }
// }

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

void restriction_first_row_cuda(int nrow,       int ncol, 
															  int row_stride, int * icolP, int nc,
                                double * v,    int ldv,
                                double * coords_x) {
  //print_matrix(nrow, ncol, v, nrow);
  double * dv;
  int lddv;

  size_t dv_pitch;
	cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
	lddv = dv_pitch / sizeof(double);
	cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
	             v,     ldv  * sizeof(double), 
	             ncol * sizeof(double), nrow, 
	             H2D);

	double * dcoords_x;
	cudaMallocHelper((void**)&dcoords_x, ncol * sizeof(double));
	cudaMemcpyHelper(dcoords_x, coords_x, ncol * sizeof(double), H2D);

	int * dicolP;
	cudaMallocHelper((void**)&dicolP, (ncol-nc) * sizeof(int));
	cudaMemcpyHelper(dicolP, icolP, (ncol-nc) * sizeof(int), H2D);	

  int B = 16;

	int total_thread = ceil((float)nrow/row_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  std::cout << "thread block: " << tb << std::endl;
  std::cout << "grid: " << grid << std::endl;

  _restriction_first_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                          																			 row_stride, dicolP, nc,
                          																			 dv,         lddv,
                          																			 dcoords_x);
  gpuErrchk(cudaGetLastError ()); 

  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
			  					   dv,    lddv * sizeof(double), 
				             ncol * sizeof(double), nrow, 
				             D2H);
}


__global__ void
_restriction_first_col_cuda(int nrow,       int ncol,
														int * irowP, int nr, int col_stride,
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

void restriction_first_col_cuda(int nrow,       int ncol, 
															  int * irowP, int nr, int col_stride,
                                double * v,    int ldv,
                                double * coords_y) {
  //print_matrix(nrow, ncol, v, nrow);
  double * dv;
  int lddv;

  size_t dv_pitch;
	cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
	lddv = dv_pitch / sizeof(double);
	cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
	             v,     ldv  * sizeof(double), 
	             ncol * sizeof(double), nrow, 
	             H2D);

	double * dcoords_y;
	cudaMallocHelper((void**)&dcoords_y, nrow * sizeof(double));
	cudaMemcpyHelper(dcoords_y, coords_y, nrow * sizeof(double), H2D);

	int * dirowP;
	cudaMallocHelper((void**)&dirowP, (nrow-nr) * sizeof(int));
	cudaMemcpyHelper(dirowP, irowP, (nrow-nr) * sizeof(int), H2D);	

  int B = 16;

	int total_thread = ceil((float)ncol/col_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  std::cout << "thread block: " << tb << std::endl;
  std::cout << "grid: " << grid << std::endl;

  _restriction_first_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                          																			 dirowP, nr, col_stride,
                          																			 dv,         lddv,
                          																			 dcoords_y);
  gpuErrchk(cudaGetLastError ()); 

  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
			  					   dv,    lddv * sizeof(double), 
				             ncol * sizeof(double), nrow, 
				             D2H);
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
                             int * dirow,    int dicol, 
                             double * dv,    int lddv, 
                             double * dcoords_x) {
  int idx0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  //printf("thread %d, nr = %d\n", idx0, nr);
  double am, bm, h1, h2;
   for (int idx = idx0; idx < nr; idx += (blockDim.x * gridDim.x) * row_stride) {
   	//printf("thread %d, nr = %d, idx = %d\n", idx0, nr, idx);
    int r = dirow[idx];
    //printf("thread %d working on row %d \n", idx0, r);
    double * vec = dv + r * lddv;
    am = 2.0 * (dicol[stride] - dicol[0]);
    bm = (dicol[stride] - dicol[0]) / am;

    double * coeff = new double[ncol];
    int counter = 1;
    coeff[0] = am;
    for (int i = col_stride; i < nc - 1; i += col_stride) {
    	h1 = dicol[i] - dicol[i - col_stride];
    	h2 = dicol[i + col_stride] - dicol[i];

    	vec[dicol[i]] -= vec[dicol[i - col_stride]] * bm;

    	am = 2.0 * (h1 + h2) - bm * h1;
    	bm = h2 / am;

    	coeff[counter] = am;
    	++counter;
    }
    h2 = dicol[nc - 1] - dicol[nc - 1 - col_stride];
    am = 2.0 * h2 - bm * h2;

    vec[dicol[nc - 1]] -= vec[dicol[nc - 1 - col_stride]] * bm;
    coeff[counter] = am;

    vec[dicol[nc - 1]] /= am;
    --counter;

    for (int i = nc - 1 - col_stride; i >= 0; i -= col_stride) {
    	h2 = dicol[i + col_stride] - dicol[i];
    	vec[dicol[i]] =
        (*vec[dicol[i]] - h2 * vec[dicol[i + col_stride]]) /
        coeff[counter];
    	--counter;
    }
  }
}


void
solve_tridiag_M_l_row_cuda(int nrow,       int ncol,
                           int nr,         int nc,
                           int row_stride, int col_stride,
                           int * irow,     int * icol,
                           double * v,     int ldv, 
                           double * coords_x) {
	double * dv;
  int lddv;

  size_t dv_pitch;
	cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
	lddv = dv_pitch / sizeof(double);
	cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
	             			  v,  ldv  * sizeof(double), 
	             			 ncol * sizeof(double), nrow, 
	             			 H2D);

	double * dcoords_x;
	cudaMallocHelper((void**)&dcoords_x, ncol * sizeof(double));
	cudaMemcpyHelper(dcoords_x, coords_x, ncol * sizeof(double), H2D);

	int * dirow;
	cudaMallocHelper((void**)&dirow, nr * sizeof(int));
	cudaMemcpyHelper( dirow, irow, nr * sizeof(int), H2D);	

  int * dicol;
  cudaMallocHelper((void**)&dicol, nc * sizeof(int));
  cudaMemcpyHelper( dicol, icol, nc * sizeof(int), H2D);  

  int B = 16;

	int total_thread = nr / row_stride;
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  std::cout << "thread block: " << tb << std::endl;
  std::cout << "grid: " << grid << std::endl;

  _solve_tridiag_M_l_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,   ncol,
                             																		 nr,     nc,
                                                                 row_stride, col_stride,
                                                                 dirow,  dicol,
                            																     dv,     lddv, 
                                                                 dcoords_x);
  gpuErrchk(cudaGetLastError ()); 

  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
			  					   dv,    lddv * sizeof(double), 
				             ncol * sizeof(double), nrow, 
				             D2H);

}


__global__ void
_solve_tridiag_M_l_col_cuda(int nrow,        int ncol,
                             int nr,         int nc,
                             int row_stride, int col_stride,
                             int * dicol,    int * dirow,
                             double * dv,    int lddv, 
                             double * dcoords_y) {
  int idx0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  //printf("thread %d, nr = %d\n", idx0, nr);
  double am, bm, h1, h2;
   for (int idx = idx0; idx < nc; idx += (blockDim.x * gridDim.x) * col_stride) {
    //printf("thread %d, nr = %d, idx = %d\n", idx0, nr, idx);
    int c = dicol[idx];
    //printf("thread %d working on row %d \n", idx0, r);
    double * vec = dv + c;
    am = 2.0 * (dirow[row_stride] - dirow[0]);
    bm = (dirow[row_stride] - dirow[0]) / am;

    double * coeff = new double[nrow];
    int counter = 1;
    coeff[0] = am;
    for (int i = row_stride; i < nr - 1; i += row_stride) {
      h1 = dirow[i] - dirow[i - row_stride];
      h2 = dirow[i + row_stride] - dirow[i];

      vec[dirow[i] * lddv] -= vec[dirow[i - row_stride] * lddv] * bm;

      am = 2.0 * (h1 + h2) - bm * h1;
      bm = h2 / am;

      coeff[counter] = am;
      ++counter;

    }
    h2 = get_h_l_cuda(dcoords_y, nr, nrow, nr - 1 - row_stride, row_stride);
    am = 2.0 * h2 - bm * h2;

    vec[dirow[nr - 1] * lddv] -= vec[dirow[nr - 1 - row_stride] * lddv] * bm;
    coeff[counter] = am;

    vec[dirow[nr - 1] * lddv] /= am;
    --counter;

    for (int i = nr - 1 - row_stride; i >= 0; i -= row_stride) {
      h2 = get_h_l_cuda(dcoords_y, nr, nrow, i, row_stride);
      vec[dirow[i] * lddv] =
        (vec[dirow[i] * lddv] - h2 * vec[dirow[i + row_stride] * lddv]) /
        coeff[counter];
      --counter;
    }
   }

}


void
solve_tridiag_M_l_col_cuda(int nrow,       int ncol,
                           int nr,         int nc,
                           int row_stride, int col_stride,
                           int * irow,     int * icol,
                           double * v,     int ldv, 
                           double * coords_y) {
  double * dv;
  int lddv;

  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                      v,  ldv  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);

  double * dcoords_y;
  cudaMallocHelper((void**)&dcoords_y, nrow * sizeof(double));
  cudaMemcpyHelper(dcoords_y, coords_y, nrow * sizeof(double), H2D);

  int * dirow;
  cudaMallocHelper((void**)&dirow, nr * sizeof(int));
  cudaMemcpyHelper( dirow, irow, nr * sizeof(int), H2D);  

  int * dicol;
  cudaMallocHelper((void**)&dicol, nc * sizeof(int));
  cudaMemcpyHelper( dicol, icol, nc * sizeof(int), H2D);  

  int B = 16;

  int total_thread = nc / col_stride;
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  std::cout << "thread block: " << tb << std::endl;
  std::cout << "grid: " << grid << std::endl;

  _solve_tridiag_M_l_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                                 nr,         nc,
                                                                 row_stride, col_stride,
                                                                 dirow,      dicol,
                                                                 dv,         lddv, 
                                                                 dcoords_y);
  gpuErrchk(cudaGetLastError ()); 

  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
                     dv,    lddv * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);

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

void add_level_l_cuda(int nrow,       int ncol, 
               int nr,          int nc,
               int row_stride, int col_stride,
               int * irow,     int * icol,
               double * v,    int ldv, 
               double * work, int ldwork) {
  double * dv;
  int lddv;

  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                      v,  ldv  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);

  double * dwork;
  int lddwork;

  size_t dwork_pitch;
  cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
  lddwork = dwork_pitch / sizeof(double);
  cudaMemcpy2DHelper(dwork, lddwork * sizeof(double), 
                      work,  ldwork  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);

  int * dicol;
  cudaMallocHelper((void**)&dicol, nc * sizeof(int));
  cudaMemcpyHelper( dicol, icol, nc * sizeof(int), H2D);  

  int * dirow;
  cudaMallocHelper((void**)&dirow, nr * sizeof(int));
  cudaMemcpyHelper( dirow, irow, nr * sizeof(int), H2D);  


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
  _add_level_l_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                       nr,         nc,
                                                      row_stride, col_stride, 
                                                      dirow,      dicol,
                                                      dv,         lddv,
                                                      dwork,      lddwork);


  gpuErrchk(cudaGetLastError ()); 

  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
                     dv,    lddv * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);

  cudaMemcpy2DHelper(work,     ldwork  * sizeof(double), 
                     dwork,    lddwork * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);
}


__global__ void 
_subtract_level_l_cuda(int nrow,       int ncol, 
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
        dv[get_idx(lddv, r, c)] -= dwork[get_idx(lddwork, r, c)];
        //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
        //y += blockDim.y * gridDim.y * stride;
      }
        //x += blockDim.x * gridDim.x * stride;
    }
}

void subtract_level_l_cuda(int nrow,       int ncol, 
               int nr,          int nc,
               int row_stride, int col_stride,
               int * irow,     int * icol,
               double * v,    int ldv, 
               double * work, int ldwork) {
  double * dv;
  int lddv;

  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                      v,  ldv  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);

  double * dwork;
  int lddwork;

  size_t dwork_pitch;
  cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
  lddwork = dwork_pitch / sizeof(double);
  cudaMemcpy2DHelper(dwork, lddwork * sizeof(double), 
                      work,  ldwork  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);

  int * dicol;
  cudaMallocHelper((void**)&dicol, nc * sizeof(int));
  cudaMemcpyHelper( dicol, icol, nc * sizeof(int), H2D);  

  int * dirow;
  cudaMallocHelper((void**)&dirow, nr * sizeof(int));
  cudaMemcpyHelper( dirow, irow, nr * sizeof(int), H2D);  


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
  _subtract_level_l_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                       nr,         nc,
                                                      row_stride, col_stride, 
                                                      dirow,      dicol,
                                                      dv,         lddv,
                                                      dwork,      lddwork);


  gpuErrchk(cudaGetLastError ()); 

  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
                     dv,    lddv * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);

  cudaMemcpy2DHelper(work,     ldwork  * sizeof(double), 
                     dwork,    lddwork * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);
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


void prep_2D_cuda(const int nr, const int nc, const int nrow, const int ncol,
             const int l_target, double *v, std::vector<double> &work,
             std::vector<double> &coords_x, std::vector<double> &coords_y,
             std::vector<double> &row_vec, std::vector<double> &col_vec) {

	int * irow  = new int[nr];
	int * irowP = new int[nrow-nr];
	int irow_ptr  = 0;
	int irowP_ptr = 0;

	for (int i = 0; i < nr; i++) {
		int irow_r = get_lindex_cuda(nr, nrow, i);
		irow[irow_ptr] = irow_r;
		if (irow_ptr > 0 && irow[irow_ptr - 1] != irow[irow_ptr] - 1) {
			irowP[irowP_ptr] = irow[irow_ptr] - 1;
			irowP_ptr ++;
		} 
		irow_ptr++;
	}

	std::cout << "irow: ";
	for (int i = 0; i < nr; i++) std::cout << irow[i] << ", ";
	std::cout << std::endl;

	std::cout << "irowP: ";
	for (int i = 0; i < nrow-nr; i++) std::cout << irowP[i] << ", ";
	std::cout << std::endl;


	int * icol  = new int[nc];
	int * icolP = new int[ncol-nc];
	int icol_ptr  = 0;
	int icolP_ptr = 0;

	for (int i = 0; i < nc; i++) {
		int icol_r = get_lindex_cuda(nc, ncol, i);
		icol[icol_ptr] = icol_r;
		if (icol_ptr > 0 && icol[icol_ptr - 1] != icol[icol_ptr] - 1) {
			icolP[icolP_ptr] = icol[icol_ptr] - 1;
			icolP_ptr ++;
		} 
		icol_ptr++;
	}

	std::cout << "icol: ";
	for (int i = 0; i < nc; i++) std::cout << icol[i] << ", ";
	std::cout << std::endl;

	std::cout << "icolP: ";
	for (int i = 0; i < ncol-nc; i++) std::cout << icolP[i] << ", ";
	std::cout << std::endl;



  int l = 0;
  int row_stride = 1;
  int col_stride = 1;
  int ldv = nrow;
  int ldwork = nrow;

  //    int stride = 1;
  pi_Ql_first_cuda(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec,
              col_vec); //(I-\Pi)u this is the initial move to 2^k+1 nodes

  mgard_cannon::copy_level_cuda(nrow, ncol, 
                                row_stride, col_stride,
                                v, ldv,
                                work.data(), ldwork);

  //assign_num_level_l_cuda(0, work.data(), 0.0, nr, nc, nrow, ncol);
  assign_num_level_l_cuda(nrow,        ncol,
                          nr,          nr,
                          row_stride,  col_stride,
                          irow,        icol,
                          work.data(), ldwork, 
                          0.0);


  row_stride = 1;
  col_stride = 1;
 
  mgard_cannon::mass_matrix_multiply_row_cuda(nrow,       ncol,
  																						row_stride, col_stride,
  																						work.data(), ldwork,
  																						coords_x.data());

  restriction_first_row_cuda(nrow, ncol,
		  											 row_stride, icolP, nc,
		  											 work.data(), ldwork,
		  											 coords_x.data());

  col_stride = 1;
	solve_tridiag_M_l_row_cuda(nrow,       ncol,
                             nr,         nc,
                             row_stride, col_stride,
                             irow,       icol, 
                             work.data(), ldwork, 
                             coords_x.data());

  for (int i = 0; i < nrow; ++i) {
    //        int ir = get_lindex_cuda(nr, nrow, irow);
    for (int j = 0; j < ncol; ++j) {
      row_vec[j] = work[mgard_common::get_index_cuda(ncol, i, j)];
    }

    //mgard_cannon::mass_matrix_multiply_cuda(0, row_vec, coords_x);

    //restriction_first_cuda(row_vec, coords_x, nc, ncol);

    for (int j = 0; j < ncol; ++j) {
      work[mgard_common::get_index_cuda(ncol, i, j)] = row_vec[j];
    }
  }

  for (int i = 0; i < nr; ++i) {
    int ir = get_lindex_cuda(nr, nrow, i);
    for (int j = 0; j < ncol; ++j) {
      row_vec[j] = work[mgard_common::get_index_cuda(ncol, ir, j)];
    }

    //mgard_gen::solve_tridiag_M_l_cuda(0, row_vec, coords_x, nc, ncol);

    for (int j = 0; j < ncol; ++j) {
      work[mgard_common::get_index_cuda(ncol, ir, j)] = row_vec[j];
    }
  }

  //   //   //std::cout  << "recomposing-colsweep" << "\n";

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) // check if we have 1-D array..
  {
  	row_stride = 1;
  	col_stride = 1;
  	mgard_cannon::mass_matrix_multiply_col_cuda(nrow,       ncol,
  																						row_stride, col_stride,
  																						work.data(), ldwork,
  																						coords_y.data());

  	restriction_first_col_cuda(nrow, ncol,
		  											 irowP, nr, col_stride,
		  											 work.data(), ldwork,
		  											 coords_y.data());

    solve_tridiag_M_l_col_cuda(nrow,       ncol,
                              nr,          nc,
                              row_stride,  col_stride,
                              irow,        icol,
                              work.data(), ldwork, 
                              coords_y.data());


  	
    for (int j = 0; j < ncol; ++j) {
      //      int jr  = get_lindex_cuda(nc,  ncol,  jcol);
      for (int i = 0; i < nrow; ++i) {
        col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, j)];
      }

      //mgard_cannon::mass_matrix_multiply_cuda(0, col_vec, coords_y);

      //mgard_gen::restriction_first_cuda(col_vec, coords_y, nr, nrow);

      for (int i = 0; i < nrow; ++i) {
        work[mgard_common::get_index_cuda(ncol, i, j)] = col_vec[i];
      }
    }

    for (int j = 0; j < nc; ++j) {
      int jr = get_lindex_cuda(nc, ncol, j);
      for (int i = 0; i < nrow; ++i) {
        col_vec[i] = work[mgard_common::get_index_cuda(ncol, i, jr)];
      }

      //mgard_gen::solve_tridiag_M_l_cuda(0, col_vec, coords_y, nr, nrow);
      for (int i = 0; i < nrow; ++i) {
        work[mgard_common::get_index_cuda(ncol, i, jr)] = col_vec[i];
      }
    }
  }
  //add_level_l_cuda(0, v, work.data(), nr, nc, nrow, ncol);
  row_stride = 1;
  col_stride = 1;
  add_level_l_cuda(nrow, ncol, nc, nr, row_stride, col_stride, 
                   irow, icol, v, ldv, work.data(), ldwork);
}



__global__ void 
_pi_Ql_cuda(int nrow,           int ncol,
            int nr,             int nr,
            int row_stride,     int col_stride,
            int * irow,         int * icol,
            double * dv,        int lddv, 
            double * dcoords_x, double * dcoords_y) {

  int row_Cstride = row_stride * 2;
  int col_Cstride = col_stride * 2;
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_Cstride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_Cstride;
  
  // in most cases it only needs to iterate once unless the input is really large
  for (int y = y0; y < nr; y += blockDim.y * gridDim.y *  row_Cstride) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x *  col_Cstride) {
      register double a00 = dv[get_idx(lddv, irow[y],             icol[x]             )];
      register double a01 = dv[get_idx(lddv, irow[y],             icol[x+col_stride]  )];
      register double a02 = dv[get_idx(lddv, irow[y],             icol[x+col_Cstride] )];
      register double a10 = dv[get_idx(lddv, irow[y+row_stride],  icol[x]             )];
      register double a11 = dv[get_idx(lddv, irow[y+row_stride],  icol[x+col_stride]  )];
      register double a12 = dv[get_idx(lddv, irow[y+row_stride],  icol[x+col_Cstride] )];
      register double a20 = dv[get_idx(lddv, irow[y+Crow_stride], icol[x]             )];
      register double a21 = dv[get_idx(lddv, irow[y+Crow_stride], icol[x+col_stride]  )];
      register double a22 = dv[get_idx(lddv, irow[y+Crow_stride], icol[x+col_Cstride] )];

      int h1_col = icol[x+col_stride]  - icol[x];
      int h2_col = icol[x+col_Cstride] - icol[x+col_stride];
      int hsum_col = h1_col + h2_col;
   
      int h1_row = irow[x+row_stride]  - irow[x];
      int h2_row = icol[x+row_Cstride] - icol[x+row_stride];
      int hsum_row = h1_row + h2_row;

      a01 -= (h1_col * a02 + h2_col * a00) / hsum_col;
      //a21 -= (h1_col * a22 + h2_col * a20) / hsum_col;

      a10 -= (h1_row * a20 + h2_row * a00) / hsum_row;
      //a12 -= (h1_row * a22 + h2_row * a02) / hsum_row;

      a11 -= 1.0 / (hsum_row * hsum_col) * (a00 * h2_col * h2_row + a02 * h1_col * h2_row + a20 * h2_col * h1_row + a22 * h1_col * h1_row);
      

      if (x + col_Cstride = nc - 1) {
        a12 -= (h1_row * a22 + h2_row * a02) / hsum_row;
      }
      if (y + row_Cstride = nr - 1) {
        a21 -= (h1_col * a22 + h2_col * a20) / hsum_col;
      }
    }
  }

}

void 
pi_Ql_cuda(int nrow,           int ncol,
           int nr,             int nr,
           int row_stride,     int col_stride,
           int * irow,         int * icol,
           double * v,        int ldv, 
           double * coords_x, double * coords_y) {

  double * dv;
  int lddv;
  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                      v,  ldv  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);

  double * dcoords_x;
  cudaMallocHelper((void**)&dcoords_x, ncol * sizeof(double));
  cudaMemcpyHelper(dcoords_x, coords_x, ncol * sizeof(double), H2D);

  double * dcoords_y;
  cudaMallocHelper((void**)&dcoords_y, nrow * sizeof(double));
  cudaMemcpyHelper(dcoords_y, coords_y, nrow * sizeof(double), H2D);

  int * dirow;
  cudaMallocHelper((void**)&dirow, nr * sizeof(int));
  cudaMemcpyHelper( dirow, irow, nr * sizeof(int), H2D);  

  int * dicol;
  cudaMallocHelper((void**)&dicol, nc * sizeof(int));
  cudaMemcpyHelper( dicol, icol, nc * sizeof(int), H2D);  

  int B = 16;
  int total_thread_y = floor((double)nrow/(row_stride * 2));
  int total_thread_x = floor((double)ncol/(col_stride * 2));
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil(total_thread_y/tby);
  int gridx = ceil(total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  _pi_Ql_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                 nr,         nc,
                                                 row_stride, col_stride,
                                                 dirow,      dicol,
                                                 dv,         lddv,
                                                 dcoords_x,  dcoords_y);
  gpuErrchk(cudaGetLastError ());
  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
                     dv,    lddv * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);
}

__global__ void 
_copy_level_l_cuda(int nrow,           int ncol,
                   int nr,             int nr,
                   int row_stride,     int col_stride,
                   int * irow,         int * icol,
                   double * dv,        int lddv,
                   double * dwork,     int ldwork) {
  
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;

  for (int y = y0; y < nr; y += blockDim.y * gridDim.y * row_stride) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x * col_stride) {
      dwork[get_idx(lddv, irow[y], icol[y])] = dv[get_idx(lddwork, irow[y], icol[y])];
    }
  }
}

void 
copy_level_l_cuda(int nrow,           int ncol,
                  int nr,             int nr,
                  int row_stride,     int col_stride,
                  int * irow,         int * icol,
                  double * v,        int ldv,
                  double * dwork,     int ldwork) {
  double * dv;
  int lddv;
  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                      v,  ldv  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  double * dwork;
  int lddwork;
  size_t dwork_pitch;
  cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
  lddwork = dwork_pitch / sizeof(double);
  cudaMemcpy2DHelper(dwork, lddwork * sizeof(double), 
                      work,  ldwork  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  int * dirow;
  cudaMallocHelper((void**)&dirow, nr * sizeof(int));
  cudaMemcpyHelper( dirow, irow, nr * sizeof(int), H2D);  

  int * dicol;
  cudaMallocHelper((void**)&dicol, nc * sizeof(int));
  cudaMemcpyHelper( dicol, icol, nc * sizeof(int), H2D);  

  int B = 16;
  int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread_x = ceil((double)ncol/(col_stride));
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil(total_thread_y/tby);
  int gridx = ceil(total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  _copy_level_l_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                   nr,         nc,
                                                   row_stride, col_stride,
                                                   dirow,      dicol,
                                                   dv,         lddv,
                                                   dwork,      ldwork);

  gpuErrchk(cudaGetLastError ());
  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
                     dv,    lddv * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);
}


__global__ void 
_assign_num_level_l_cuda(int nrow,           int ncol,
                         int nr,             int nr,
                         int row_stride,     int col_stride,
                         int * irow,         int * icol,
                         double * dv,        int lddv,
                         double num) {
  
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;

  for (int y = y0; y < nr; y += blockDim.y * gridDim.y * row_stride) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x * col_stride) {
      dv[get_idx(lddv, irow[y], icol[y])] = num;
    }
  }
}


void 
assign_num_level_l_cuda(int nrow,           int ncol,
                        int nr,             int nr,
                        int row_stride,     int col_stride,
                        int * irow,         int * icol,
                        double * v,        int ldv,
                        double num) {
  double * dv;
  int lddv;
  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                      v,  ldv  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  double * dwork;
  int lddwork;
  size_t dwork_pitch;
  cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
  lddwork = dwork_pitch / sizeof(double);
  cudaMemcpy2DHelper(dwork, lddwork * sizeof(double), 
                      work,  ldwork  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  int * dirow;
  cudaMallocHelper((void**)&dirow, nr * sizeof(int));
  cudaMemcpyHelper( dirow, irow, nr * sizeof(int), H2D);  

  int * dicol;
  cudaMallocHelper((void**)&dicol, nc * sizeof(int));
  cudaMemcpyHelper( dicol, icol, nc * sizeof(int), H2D);  

  int B = 16;
  int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread_x = ceil((double)ncol/(col_stride));
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil(total_thread_y/tby);
  int gridx = ceil(total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  _assign_num_level_l_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                              nr,         nc,
                                                              row_stride, col_stride,
                                                              dirow,      dicol,
                                                              dv,         lddv,
                                                              num);

  gpuErrchk(cudaGetLastError ());
  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
                     dv,    lddv * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);
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
    double * vec = dv + r * lddv;
    double temp1, temp2;
    double h1, h2;
    temp1 = vec[dicol[0]];
    h1 = dicol[col_stride] - dicol[0];
    vec[dicol[0]] = 2.0 * h1 * temp1 + h1 * v[dicol[col_stride]];

    for (int i = col_stride; i <= nc - 1 - col_stride; i += col_stride) {
      temp2 = vec[dicol[i]];
      h1 = dicol[i] - dicol[i - col_stride];
      h2 = dicol[i + col_stride] - dicol[i];
      vec[dicol[i]] = h1 * temp1  + 2 * (h1 + h2) * temp2 + h2 * vec[dicol[i + col_stride]];
      temp1 = temp2;
    }
    vec[dicol[nc - 1]] = (dicol[nc - 1] - dicol[nc - col_stride - 1]) * temp1 +
                        2 * (dicol[nc - 1] - dicol[nc - col_stride - 1]) * vec[dicol[nc - 1]];
  }
}

void mass_mult_l_row_cuda(int nrow,       int ncol,
                          int nr,         int nc,
                          int row_stride, int col_stride,
                          int * irow,     int * icol,
                          double * v,     int ldv,
                          double * coords_x) {
  double * dv;
  int lddv;
  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                      v,  ldv  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  double * dwork;
  int lddwork;
  size_t dwork_pitch;
  cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
  lddwork = dwork_pitch / sizeof(double);
  cudaMemcpy2DHelper(dwork, lddwork * sizeof(double), 
                      work,  ldwork  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  int * dirow;
  cudaMallocHelper((void**)&dirow, nr * sizeof(int));
  cudaMemcpyHelper( dirow, irow, nr * sizeof(int), H2D);  

  int * dicol;
  cudaMallocHelper((void**)&dicol, nc * sizeof(int));
  cudaMemcpyHelper( dicol, icol, nc * sizeof(int), H2D);  

  double * dcoords_x;
  cudaMallocHelper((void**)&dcoords_x, ncol * sizeof(double));
  cudaMemcpyHelper(dcoords_x, coords_x, ncol * sizeof(double), H2D);

  int B = 16;
  //int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread = ceil((double)nrow/(row_stride));
  //int tby = min(B, total_thread_y);
  int tb = min(B, total_thread);
  //int gridy = ceil(total_thread_y/tby);
  int grid = ceil((double)total_thread/tb);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  _mass_mult_l_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                           nr,         nc,
                                                           row_stride, col_stride,
                                                           dirow,      dicol,
                                                           dv,         lddv,
                                                           dcoords_x);
  gpuErrchk(cudaGetLastError ());
  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
                     dv,    lddv * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);
}

__global__ void
_mass_mult_l_col_cuda(int nrow,       int ncol,
                      int nr,         int nc,
                      int row_stride, int col_stride,
                      int * dirow,    int * dicol,
                      double * dv,    int lddv,
                      double * dcoords_y) {
  int c0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  for (int c = c0; c < nc; c += (blockDim.x * gridDim.x) * col_stride) {
    double * vec = dv + c;
    double temp1, temp2;
    double h1, h2;
    temp1 = vec[dirow[0]];
    h1 = dirow[row_stride] - dirow[0];
    vec[dirow[0] * lddv] = 2.0 * h1 * temp1 + h1 * v[dirow[row_stride] * lddv];

    for (int i = row_stride; i <= nr - 1 - row_stride; i += row_stride) {
      temp2 = vec[dirow[i] * lddv];
      h1 = dirow[i] - dirow[i - row_stride];
      h2 = dirow[i + row_stride] - dirow[i];
      vec[dirow[i] * lddv] = h1 * temp1  + 2 * (h1 + h2) * temp2 + h2 * vec[dirow[i + row_stride] * lddv];
      temp1 = temp2;
    }
    vec[dirow[nr - 1] * lddv] = (dirow[nr - 1] - dirow[nr - row_stride - 1]) * temp1 +
                        2 * (dirow[nr - 1] - dirow[nr - row_stride - 1]) * vec[dirow[nr - 1] * lddv];
  }
}

void mass_mult_l_col_cuda(int nrow,       int ncol,
                          int nr,         int nc,
                          int row_stride, int col_stride,
                          int * irow,     int * icol,
                          double * v,     int ldv,
                          double * coords_y) {
  double * dv;
  int lddv;
  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                      v,  ldv  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  double * dwork;
  int lddwork;
  size_t dwork_pitch;
  cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
  lddwork = dwork_pitch / sizeof(double);
  cudaMemcpy2DHelper(dwork, lddwork * sizeof(double), 
                      work,  ldwork  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  int * dirow;
  cudaMallocHelper((void**)&dirow, nr * sizeof(int));
  cudaMemcpyHelper( dirow, irow, nr * sizeof(int), H2D);  

  int * dicol;
  cudaMallocHelper((void**)&dicol, nc * sizeof(int));
  cudaMemcpyHelper( dicol, icol, nc * sizeof(int), H2D);  

  double * dcoords_y;
  cudaMallocHelper((void**)&dcoords_y, nrow * sizeof(double));
  cudaMemcpyHelper(dcoords_y, coords_y, nrow * sizeof(double), H2D);

  int B = 16;
  //int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread = ceil((double)ncol/(col_stride));
  //int tby = min(B, total_thread_y);
  int tb = min(B, total_thread);
  //int gridy = ceil(total_thread_y/tby);
  int grid = ceil((double)total_thread/tb);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  _mass_mult_l_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                           nr,         nc,
                                                           row_stride, col_stride,
                                                           dirow,      dicol,
                                                           dv,         lddv,
                                                           dcoords_y);
  gpuErrchk(cudaGetLastError ());
  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
                     dv,    lddv * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);
}


__global__ void
_restriction_l_row_cuda(int nrow,       int ncol,
                      int nr,         int nc,
                      int row_stride, int col_stride,
                      int * dirow,    int * dicol,
                      double * dv,    int lddv,
                      double * dcoords_x) {
  int col_Pstride = col_stride / 2;
  int r0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  for (int r = r0; r < nr; r += (blockDim.x * gridDim.x) * row_stride) {
    double * vec = dv + r * lddv;
    double h1 = dicol[col_Pstride] - dicol[0];
    double h2 = dicol[col_stride] - dicol[col_Pstride];
    double hsum = h1 + h2;
    vec[dicol[0]] += h2 * (vec[col_Pstride]) / hsum;

    for (int i = col_stride; i <= nc - col_stride; i += col_stride) {
      vec[dicol[i]] += h1 * vec[dicol[i - col_Pstride]] / hsum;
      h1 = dicol[i + col_Pstride] - dicol[i];
      h2 = dicol[i + col_stride] - dicol[i + col_Pstride];
      hsum = h1 + h2;
      vec[dicol[i]] += h2 * vec[dicol[i + col_Pstride]] / hsum;
    }
    vec[dicol[nc - 1]] += h1 * vec[dicol[nc - col_Pstride - 1]] / hsum;
  }
}

void 
restriction_l_row_cuda(int nrow,       int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * irow,     int * icol,
                       double * v,     int ldv,
                       double * coords_x) {
  double * dv;
  int lddv;
  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                      v,  ldv  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  double * dwork;
  int lddwork;
  size_t dwork_pitch;
  cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
  lddwork = dwork_pitch / sizeof(double);
  cudaMemcpy2DHelper(dwork, lddwork * sizeof(double), 
                      work,  ldwork  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  int * dirow;
  cudaMallocHelper((void**)&dirow, nr * sizeof(int));
  cudaMemcpyHelper( dirow, irow, nr * sizeof(int), H2D);  

  int * dicol;
  cudaMallocHelper((void**)&dicol, nc * sizeof(int));
  cudaMemcpyHelper( dicol, icol, nc * sizeof(int), H2D);  

  double * dcoords_x;
  cudaMallocHelper((void**)&dcoords_x, ncol * sizeof(double));
  cudaMemcpyHelper(dcoords_x, coords_x, ncol * sizeof(double), H2D);

  int B = 16;
  //int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread = ceil((double)nrow/(row_stride));
  //int tby = min(B, total_thread_y);
  int tb = min(B, total_thread);
  //int gridy = ceil(total_thread_y/tby);
  int grid = ceil((double)total_thread/tb);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  _restriction_l_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                           nr,         nc,
                                                           row_stride, col_stride,
                                                           dirow,      dicol,
                                                           dv,         lddv,
                                                           dcoords_x);
  gpuErrchk(cudaGetLastError ());
  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
                     dv,    lddv * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);
}


__global__ void
_restriction_l_col_cuda(int nrow,       int ncol,
                      int nr,         int nc,
                      int row_stride, int col_stride,
                      int * dirow,    int * dicol,
                      double * dv,    int lddv,
                      double * dcoords_y) {
  int row_Pstride = row_stride / 2;
  int c0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  for (int c = c0; c < nc; c += (blockDim.x * gridDim.x) * col_stride) {
    double * vec = dv + c;
    double h1 = dirow[row_Pstride] - dirow[0];
    double h2 = dirow[row_stride] - dirow[row_Pstride];
    double hsum = h1 + h2;
    vec[dirow[0] * lddv] += h2 * (vec[row_Pstride] * lddv) / hsum;

    for (int i = row_stride; i <= nr - row_stride; i += row_stride) {
      vec[dirow[i] * lddv] += h1 * vec[dirow[i - row_Pstride] * lddv] / hsum;
      h1 = dirow[i + row_Pstride] - dirow[i];
      h2 = dirow[i + row_stride] - dirow[i + row_Pstride];
      hsum = h1 + h2;
      vec[dirow[i] * lddv] += h2 * vec[dirow[i + row_Pstride] * lddv] / hsum;
    }
    vec[dirow[nc - 1] * lddv] += h1 * vec[dirow[nc - row_Pstride - 1] * lddv] / hsum;
  }
}

void 
restriction_l_col_cuda(int nrow,       int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * irow,     int * icol,
                       double * v,     int ldv,
                       double * coords_y) {
  double * dv;
  int lddv;
  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                      v,  ldv  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  double * dwork;
  int lddwork;
  size_t dwork_pitch;
  cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
  lddwork = dwork_pitch / sizeof(double);
  cudaMemcpy2DHelper(dwork, lddwork * sizeof(double), 
                      work,  ldwork  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  int * dirow;
  cudaMallocHelper((void**)&dirow, nr * sizeof(int));
  cudaMemcpyHelper( dirow, irow, nr * sizeof(int), H2D);  

  int * dicol;
  cudaMallocHelper((void**)&dicol, nc * sizeof(int));
  cudaMemcpyHelper( dicol, icol, nc * sizeof(int), H2D);  

  double * dcoords_y;
  cudaMallocHelper((void**)&dcoords_y, nrow * sizeof(double));
  cudaMemcpyHelper(dcoords_y, coords_y, nrow * sizeof(double), H2D);

  int B = 16;
  //int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread = ceil((double)ncol/(col_stride));
  //int tby = min(B, total_thread_y);
  int tb = min(B, total_thread);
  //int gridy = ceil(total_thread_y/tby);
  int grid = ceil((double)total_thread/tb);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  _restriction_l_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                             nr,         nc,
                                                             row_stride, col_stride,
                                                             dirow,      dicol,
                                                             dv,         lddv,
                                                             dcoords_y);
  gpuErrchk(cudaGetLastError ());
  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
                     dv,    lddv * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);
}


__global__ void
_prolongate_l_row_cuda(int nrow, int ncol,
                       int nr,   int nc,
                       int row_stride, int col_stride,
                       int * dirow,    int * dicol,
                       double * dv,    int lddv,
                       double * coords_x) {

  int col_Pstride = col_stride / 2;
  int r0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  for (int r = r0; r < nr; r += (blockDim.x * gridDim.x) * row_stride) {
    double * vec = dv + r * lddv;
    for (int i = col_stride; i < nc; i += col_stride) {
      double h1 = dicol[i - col_Pstride] - dicol[i - col_stride];
      double h2 = dicol[i] - dicol[i - col_Pstride];
      double hsum = h1 + h2;
      vec[dicol[i - col_Pstride]] = (h2 * vec[dicol[i - col_stride]] + h1 * vec[dicol[i]]) / hsum;
    }
  }
}

void 
prolongate_l_row_cuda(int nrow,       int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * irow,     int * icol,
                       double * v,     int ldv,
                       double * coords_x) {
  double * dv;
  int lddv;
  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                      v,  ldv  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  double * dwork;
  int lddwork;
  size_t dwork_pitch;
  cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
  lddwork = dwork_pitch / sizeof(double);
  cudaMemcpy2DHelper(dwork, lddwork * sizeof(double), 
                      work,  ldwork  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  int * dirow;
  cudaMallocHelper((void**)&dirow, nr * sizeof(int));
  cudaMemcpyHelper( dirow, irow, nr * sizeof(int), H2D);  

  int * dicol;
  cudaMallocHelper((void**)&dicol, nc * sizeof(int));
  cudaMemcpyHelper( dicol, icol, nc * sizeof(int), H2D);  

  double * dcoords_x;
  cudaMallocHelper((void**)&dcoords_x, ncol * sizeof(double));
  cudaMemcpyHelper(dcoords_x, coords_x, ncol * sizeof(double), H2D);

  int B = 16;
  //int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread = ceil((double)nrow/(row_stride));
  //int tby = min(B, total_thread_y);
  int tb = min(B, total_thread);
  //int gridy = ceil(total_thread_y/tby);
  int grid = ceil((double)total_thread/tb);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  _prolongate_l_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                           nr,         nc,
                                                           row_stride, col_stride,
                                                           dirow,      dicol,
                                                           dv,         lddv,
                                                           dcoords_x);
  gpuErrchk(cudaGetLastError ());
  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
                     dv,    lddv * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);
}

__global__ void
_prolongate_l_col_cuda(int nrow, int ncol,
                       int nr,   int nc,
                       int row_stride, int col_stride,
                       int * dirow,    int * dicol,
                       double * dv,    int lddv,
                       double * coords_y) {

  int row_Pstride = row_stride / 2;
  int c0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  for (int c = c0; c < nc; c += (blockDim.x * gridDim.x) * col_stride) {
    double * vec = dv + c;
    for (int i = row_stride; i < nr; i += row_stride) {
      double h1 = dirow[i - row_Pstride] - dirow[i - row_stride];
      double h2 = dirow[i] - dirow[i - row_Pstride];
      double hsum = h1 + h2;
      vec[dirow[i - row_Pstride] * lddv] = (h2 * vec[dirow[i - row_stride] * lddv] + h1 * vec[dirow[i] * lddv]) / hsum;
    }
  }
}

void 
prolongate_l_col_cuda(int nrow,       int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * irow,     int * icol,
                       double * v,     int ldv,
                       double * coords_y) {
  double * dv;
  int lddv;
  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                      v,  ldv  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  double * dwork;
  int lddwork;
  size_t dwork_pitch;
  cudaMallocPitchHelper((void**)&dwork, &dwork_pitch, ncol * sizeof(double), nrow);
  lddwork = dwork_pitch / sizeof(double);
  cudaMemcpy2DHelper(dwork, lddwork * sizeof(double), 
                      work,  ldwork  * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     H2D);
  int * dirow;
  cudaMallocHelper((void**)&dirow, nr * sizeof(int));
  cudaMemcpyHelper( dirow, irow, nr * sizeof(int), H2D);  

  int * dicol;
  cudaMallocHelper((void**)&dicol, nc * sizeof(int));
  cudaMemcpyHelper( dicol, icol, nc * sizeof(int), H2D);  

  double * dcoords_y;
  cudaMallocHelper((void**)&dcoords_y, nrow * sizeof(double));
  cudaMemcpyHelper(dcoords_y, coords_y, nrow * sizeof(double), H2D);

  int B = 16;
  //int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread = ceil((double)ncol/(col_stride));
  //int tby = min(B, total_thread_y);
  int tb = min(B, total_thread);
  //int gridy = ceil(total_thread_y/tby);
  int grid = ceil((double)total_thread/tb);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  _prolongate_l_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
                                                             nr,         nc,
                                                             row_stride, col_stride,
                                                             dirow,      dicol,
                                                             dv,         lddv,
                                                             dcoords_y);
  gpuErrchk(cudaGetLastError ());
  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
                     dv,    lddv * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);
}



__global__ void
_prolongate_last_row_cuda(int nrow,       int ncol,
                          int row_stride, int * icolP, int nc,
                          double * dv,    int lddv,
                          double * dcoords_x) {
  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  //int y = threadIdx.y * stride;
  for (int x = idx; x < nrow; x += (blockDim.x * gridDim.x) * row_stride) {
    //printf("thread working on %d \n", x);
    double * vec = dv + x * lddv;
    for (int i = 0; i < ncol-nc; i++) {
      double h1 = 1;//mgard_common::get_h_cuda(dcoords_x, icolP[i] - 1, 1);
      double h2 = 1;//mgard_common::get_h_cuda(dcoords_x, icolP[i]    , 1);
      double hsum = h1 + h2;
      vec[icolP[i]] = (h2 * vec[icolP[i] - 1] + h1 * vec[icolP[i] + 1]) / hsum;
    }

  }
}

void prolongate_last_row_cuda(int nrow,       int ncol, 
                                int row_stride, int * icolP, int nc,
                                double * v,    int ldv,
                                double * coords_x) {
  //print_matrix(nrow, ncol, v, nrow);
  double * dv;
  int lddv;

  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
               v,     ldv  * sizeof(double), 
               ncol * sizeof(double), nrow, 
               H2D);

  double * dcoords_x;
  cudaMallocHelper((void**)&dcoords_x, ncol * sizeof(double));
  cudaMemcpyHelper(dcoords_x, coords_x, ncol * sizeof(double), H2D);

  int * dicolP;
  cudaMallocHelper((void**)&dicolP, (ncol-nc) * sizeof(int));
  cudaMemcpyHelper(dicolP, icolP, (ncol-nc) * sizeof(int), H2D);  

  int B = 16;

  int total_thread = ceil((float)nrow/row_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  std::cout << "thread block: " << tb << std::endl;
  std::cout << "grid: " << grid << std::endl;

  _prolongate_last_row_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                                                                 row_stride, dicolP, nc,
                                                                 dv,         lddv,
                                                                 dcoords_x);
  gpuErrchk(cudaGetLastError ()); 

  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
                     dv,    lddv * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);
}



__global__ void
_prolongate_last_col_cuda(int nrow,       int ncol,
                          int * irowP, int nr, int col_stride,
                          double * dv,    int lddv,
                          double * dcoords_y) {
  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  //int y = threadIdx.y * stride;
  for (int x = idx; x < ncol; x += (blockDim.x * gridDim.x) * col_stride) {
    //printf("thread working on %d \n", x);
    double * vec = dv + x;
    for (int i = 0; i < nrow-nr; i++) {
      double h1 = 1; //mgard_common::get_h_cuda(dcoords_y, irowP[i] - 1, 1);
      double h2 = 1; //mgard_common::get_h_cuda(dcoords_y, irowP[i]    , 1);
      double hsum = h1 + h2;
      vec[irowP[i] * lddv] = (h2 * vec[(irowP[i] - 1) * lddv] + h1 * vec[(irowP[i] + 1) * lddv]) / hsum;
    }
  }
}

void prolongate_last_col_cuda(int nrow,       int ncol, 
                              int * irowP, int nr, int col_stride,
                              double * v,    int ldv,
                              double * coords_y) {
  //print_matrix(nrow, ncol, v, nrow);
  double * dv;
  int lddv;

  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
               v,     ldv  * sizeof(double), 
               ncol * sizeof(double), nrow, 
               H2D);

  double * dcoords_y;
  cudaMallocHelper((void**)&dcoords_y, nrow * sizeof(double));
  cudaMemcpyHelper(dcoords_y, coords_y, nrow * sizeof(double), H2D);

  int * dirowP;
  cudaMallocHelper((void**)&dirowP, (nrow-nr) * sizeof(int));
  cudaMemcpyHelper(dirowP, irowP, (nrow-nr) * sizeof(int), H2D);  

  int B = 16;

  int total_thread = ceil((float)ncol/col_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  std::cout << "thread block: " << tb << std::endl;
  std::cout << "grid: " << grid << std::endl;

  _prolongate_last_col_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol, 
                                                                 dirowP, nr, col_stride,
                                                                 dv,         lddv,
                                                                 dcoords_y);
  gpuErrchk(cudaGetLastError ()); 

  cudaMemcpy2DHelper(v,     ldv  * sizeof(double), 
                     dv,    lddv * sizeof(double), 
                     ncol * sizeof(double), nrow, 
                     D2H);
}




void refactor_2D_cuda(const int nr, const int nc, const int nrow, const int ncol,
                 const int l_target, double *v, std::vector<double> &work,
                 std::vector<double> &coords_x, std::vector<double> &coords_y,
                 std::vector<double> &row_vec, std::vector<double> &col_vec) {
  
  int ldv = ncol;
  int ldwork = ncol;

  int row_stride;
  int col_stride;

  int * irow  = new int[nr];
  int * irowP = new int[nrow-nr];
  int irow_ptr  = 0;
  int irowP_ptr = 0;

  for (int i = 0; i < nr; i++) {
    int irow_r = get_lindex_cuda(nr, nrow, i);
    irow[irow_ptr] = irow_r;
    if (irow_ptr > 0 && irow[irow_ptr - 1] != irow[irow_ptr] - 1) {
      irowP[irowP_ptr] = irow[irow_ptr] - 1;
      irowP_ptr ++;
    } 
    irow_ptr++;
  }

  std::cout << "irow: ";
  for (int i = 0; i < nr; i++) std::cout << irow[i] << ", ";
  std::cout << std::endl;

  std::cout << "irowP: ";
  for (int i = 0; i < nrow-nr; i++) std::cout << irowP[i] << ", ";
  std::cout << std::endl;


  int * icol  = new int[nc];
  int * icolP = new int[ncol-nc];
  int icol_ptr  = 0;
  int icolP_ptr = 0;

  for (int i = 0; i < nc; i++) {
    int icol_r = get_lindex_cuda(nc, ncol, i);
    icol[icol_ptr] = icol_r;
    if (icol_ptr > 0 && icol[icol_ptr - 1] != icol[icol_ptr] - 1) {
      icolP[icolP_ptr] = icol[icol_ptr] - 1;
      icolP_ptr ++;
    } 
    icol_ptr++;
  }

  std::cout << "icol: ";
  for (int i = 0; i < nc; i++) std::cout << icol[i] << ", ";
  std::cout << std::endl;

  std::cout << "icolP: ";
  for (int i = 0; i < ncol-nc; i++) std::cout << icolP[i] << ", ";
  std::cout << std::endl;
  

  // refactor
  //    //std::cout  << "I am the general refactorer!" <<"\n";
  for (int l = 0; l < l_target; ++l) {
    int stride = std::pow(2, l); // current stride
    int Cstride = stride * 2;    // coarser stride

    // -> change funcs in pi_QL to use _l functions, otherwise distances are
    // wrong!!!
    // pi_Ql(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec,
    //       col_vec); // rename!. v@l has I-\Pi_l Q_l+1 u
    row_stride = stride;
    col_stride = stride;
    pi_Ql_cuda(nrow,            ncol,
               nr,              nr,
               row_stride,      col_stride,
               irow,            icol,
               v,               ldv, 
               coords_x.data(), coords_y.data());

    //copy_level_l(l, v, work.data(), nr, nc, nrow, ncol);
    row_stride = stride;
    col_stride = stride;
    copy_level_l_cuda(nrow,        ncol,
                      nr,          nr,
                      row_stride,  col_stride,
                      irow,        icol,
                      v,           ldv, 
                      work.data(), work);

    //assign_num_level_l(l + 1, work.data(), 0.0, nr, nc, nrow, ncol);
    row_stride = Cstride;
    col_stride = Cstride;
    assign_num_level_l_cuda(nrow,        ncol,
                            nr,          nr,
                            row_stride,  col_stride,
                            irow,        icol,
                            work.data(), ldwork, 
                            0.0);

    row_stride = 1;
    col_stride = stride;
    mass_mult_l_row_cuda(nrow,        ncol,
                         nr,          nc,
                         row_stride,  col_stride,
                         irow,        icol,
                         work.data(), ldwork,
                         coords_x.data());


    row_stride = 1;
    col_stride = Cstride;
    restriction_l_row_cuda(nrow,       ncol,
                           nr,         nc,
                           row_stride, col_stride,
                           irow,       icol,
                           work.data(), ldwork,
                           coords_x.data());

    row_stride = 1;
    col_stride = Cstride;
    solve_tridiag_M_l_row_cuda(nrow,       ncol,
                               nr,         nc,
                               row_stride, col_stride,
                               irow,       icol,
                               v, ldv, 
                               coords_x.data());

    // row-sweep
    for (int irow = 0; irow < nr; ++irow) {
      int ir = get_lindex_cuda(nr, nrow, irow);
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[mgard_common::get_index_cuda(ncol, ir, jcol)];
      }

      //mgard_gen::mass_mult_l(l, row_vec, coords_x, nc, ncol);

      //mgard_gen::restriction_l(l + 1, row_vec, coords_x, nc, ncol);

      //mgard_gen::solve_tridiag_M_l(l + 1, row_vec, coords_x, nc, ncol);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        work[mgard_common::get_index_cuda(ncol, ir, jcol)] = row_vec[jcol];
      }
    }

    // column-sweep
    if (nrow > 1) // do this if we have an 2-dimensional array
    {
      row_stride = stride;
      col_stride = Cstride;
      mass_mult_l_col_cuda(nrow,        ncol,
                           nr,          nc,
                           row_stride,  col_stride,
                           irow,        icol,
                           work.data(), ldwork,
                           coords_x.data());


      row_stride = Cstride;
      col_stride = Cstride;
      restriction_l_col_cuda(nrow,       ncol,
                             nr,         nc,
                             row_stride, col_stride,
                             irow,       icol,
                             work.data(), ldwork,
                             coords_x.data());

      row_stride = Cstride;
      col_stride = Cstride;
      solve_tridiag_M_l_col_cuda(nrow,       ncol,
                                 nr,         nc,
                                 row_stride, col_stride,
                                 irow,       icol,
                                 v, ldv, 
                                 coords_x.data());
      for (int jcol = 0; jcol < nc; jcol += Cstride) {
        int jr = get_lindex_cuda(nc, ncol, jcol);
        for (int irow = 0; irow < nrow; ++irow) {
          col_vec[irow] = work[mgard_common::get_index_cuda(ncol, irow, jr)];
        }

        // mgard_gen::mass_mult_l(l, col_vec, coords_y, nr, nrow);
        // mgard_gen::restriction_l(l + 1, col_vec, coords_y, nr, nrow);
        // mgard_gen::solve_tridiag_M_l(l + 1, col_vec, coords_y, nr, nrow);

        for (int irow = 0; irow < nrow; ++irow) {
          work[mgard_common::get_index_cuda(ncol, irow, jr)] = col_vec[irow];
        }
      }
    }

    // Solved for (z_l, phi_l) = (c_{l+1}, vl)
    //add_level_l(l + 1, v, work.data(), nr, nc, nrow, ncol);

    add_level_l_cuda(nrow,        ncol, 
                     nr,          nc,
                     row_stride,  col_stride,
                     irow,        icol,
                     v,           ldv, 
                     work.data(), ldwork)


  }
}

void recompose_2D_cuda(const int nr, const int nc, const int nrow, const int ncol,
                  const int l_target, double *v, std::vector<double> &work,
                  std::vector<double> &coords_x, std::vector<double> &coords_y,
                  std::vector<double> &row_vec, std::vector<double> &col_vec) {
  int ldv = ncol;
  int ldwork = ncol;

  int row_stride;
  int col_stride;

  int * irow  = new int[nr];
  int * irowP = new int[nrow-nr];
  int irow_ptr  = 0;
  int irowP_ptr = 0;

  for (int i = 0; i < nr; i++) {
    int irow_r = get_lindex_cuda(nr, nrow, i);
    irow[irow_ptr] = irow_r;
    if (irow_ptr > 0 && irow[irow_ptr - 1] != irow[irow_ptr] - 1) {
      irowP[irowP_ptr] = irow[irow_ptr] - 1;
      irowP_ptr ++;
    } 
    irow_ptr++;
  }

  std::cout << "irow: ";
  for (int i = 0; i < nr; i++) std::cout << irow[i] << ", ";
  std::cout << std::endl;

  std::cout << "irowP: ";
  for (int i = 0; i < nrow-nr; i++) std::cout << irowP[i] << ", ";
  std::cout << std::endl;


  int * icol  = new int[nc];
  int * icolP = new int[ncol-nc];
  int icol_ptr  = 0;
  int icolP_ptr = 0;

  for (int i = 0; i < nc; i++) {
    int icol_r = get_lindex_cuda(nc, ncol, i);
    icol[icol_ptr] = icol_r;
    if (icol_ptr > 0 && icol[icol_ptr - 1] != icol[icol_ptr] - 1) {
      icolP[icolP_ptr] = icol[icol_ptr] - 1;
      icolP_ptr ++;
    } 
    icol_ptr++;
  }

  std::cout << "icol: ";
  for (int i = 0; i < nc; i++) std::cout << icol[i] << ", ";
  std::cout << std::endl;

  std::cout << "icolP: ";
  for (int i = 0; i < ncol-nc; i++) std::cout << icolP[i] << ", ";
  std::cout << std::endl;


  // recompose
  //    //std::cout  << "recomposing" << "\n";
  for (int l = l_target; l > 0; --l) {

    int stride = std::pow(2, l); // current stride
    int Pstride = stride / 2;

    //copy_level_l(l - 1, v, work.data(), nr, nc, nrow, ncol);
    row_stride = Pstride;
    col_stride = Pstride;
    copy_level_l_cuda(nrow,        ncol,
                      nr,          nr,
                      row_stride,  col_stride,
                      irow,        icol,
                      v,           ldv, 
                      work.data(), work);

    //assign_num_level_l(l, work.data(), 0.0, nr, nc, nrow, ncol);
    row_stride = stride;
    col_stride = stride;
    assign_num_level_l_cuda(nrow,        ncol,
                            nr,          nr,
                            row_stride,  col_stride,
                            irow,        icol,
                            work.data(), ldwork, 
                            0.0);

    //        //std::cout  << "recomposing-rowsweep" << "\n";
    //  l = 0;
    // row-sweep
    row_stride = 1;
    col_stride = stride;
    mass_mult_l_row_cuda(nrow,        ncol,
                         nr,          nc,
                         row_stride,  col_stride,
                         irow,        icol,
                         work.data(), ldwork,
                         coords_x.data());


    row_stride = 1;
    col_stride = Cstride;
    restriction_l_row_cuda(nrow,       ncol,
                           nr,         nc,
                           row_stride, col_stride,
                           irow,       icol,
                           work.data(), ldwork,
                           coords_x.data());

    row_stride = 1;
    col_stride = stride;
    solve_tridiag_M_l_row_cuda(nrow,        ncol,
                               nr,          nc,
                               row_stride,  col_stride,
                               irow,        icol,
                               work.data(), ldwork,
                               coords_x.data());

    for (int irow = 0; irow < nr; ++irow) {
      int ir = get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
      }

      // mgard_gen::mass_mult_l(l - 1, row_vec, coords_x, nc, ncol);

      // mgard_gen::restriction_l(l, row_vec, coords_x, nc, ncol);

      // mgard_gen::solve_tridiag_M_l(l, row_vec, coords_x, nc, ncol);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol];
      }
    }

    //   //std::cout  << "recomposing-colsweep" << "\n";

    // column-sweep, this is the slow one! Need something like column_copy
    if (nrow > 1) // check if we have 1-D array..
    {
      row_stride = Pstride;
      col_stride = stride;
      mass_mult_l_col_cuda(nrow,        ncol,
                           nr,          nc,
                           row_stride,  col_stride,
                           irow,        icol,
                           work.data(), ldwork,
                           coords_x.data());


      row_stride = stride;
      col_stride = stride;
      restriction_l_col_cuda(nrow,       ncol,
                             nr,         nc,
                             row_stride, col_stride,
                             irow,       icol,
                             work.data(), ldwork,
                             coords_x.data());

      row_stride = stride;
      col_stride = stride;
      solve_tridiag_M_l_col_cuda(nrow,        ncol,
                                 nr,          nc,
                                 row_stride,  col_stride,
                                 irow,        icol,
                                 work.data(), ldwork,
                                 coords_x.data());
      for (int jcol = 0; jcol < nc; jcol += stride) {
        int jr = get_lindex(nc, ncol, jcol);
        for (int irow = 0; irow < nrow; ++irow) {
          col_vec[irow] = work[mgard_common::get_index(ncol, irow, jr)];
        }

        // mgard_gen::mass_mult_l(l - 1, col_vec, coords_y, nr, nrow);

        // mgard_gen::restriction_l(l, col_vec, coords_y, nr, nrow);

        // mgard_gen::solve_tridiag_M_l(l, col_vec, coords_y, nr, nrow);

        for (int irow = 0; irow < nrow; ++irow) {
          work[mgard_common::get_index(ncol, irow, jr)] = col_vec[irow];
        }
      }
    }

    //subtract_level_l(l, work.data(), v, nr, nc, nrow, ncol); // do -(Qu - zl)
    row_stride = stride;
    col_stride = stride;
    subtract_level_l_cuda(nrow,        ncol, 
                     nr,          nc,
                     row_stride,  col_stride,
                     irow,        icol,
                     work.data(), ldwork,
                     v,           ldv );

    //        //std::cout  << "recomposing-rowsweep2" << "\n";

    //   //int Pstride = stride/2; //finer stride

    //   // row-sweep

    row_stride = stride;
    col_stride = stride;
    prolongate_l_row_cuda(nrow,        ncol, 
                           nr,          nc,
                           row_stride,  col_stride,
                           irow,        icol,
                           work.data(), ldwork,
                           coords_x.data());



    for (int irow = 0; irow < nr; irow += stride) {
      int ir = get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
      }

      //mgard_gen::prolongate_l(l, row_vec, coords_x, nc, ncol);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol];
      }
    }

    //   //std::cout  << "recomposing-colsweep2" << "\n";
    // column-sweep, this is the slow one! Need something like column_copy
    if (nrow > 1) {
      row_stride = stride;
      col_stride = Pstride;
      prolongate_l_col_cuda(nrow,        ncol, 
                             nr,          nc,
                             row_stride,  col_stride,
                             irow,        icol,
                             work.data(), ldwork,
                             coords_y.data());

      for (int jcol = 0; jcol < nc; jcol += Pstride) {
        int jr = get_lindex(nc, ncol, jcol);
        for (int irow = 0; irow < nrow; ++irow) // copy all rows
        {
          col_vec[irow] = work[mgard_common::get_index(ncol, irow, jr)];
        }

        //mgard_gen::prolongate_l(l, col_vec, coords_y, nr, nrow);

        for (int irow = 0; irow < nrow; ++irow) {
          work[mgard_common::get_index(ncol, irow, jr)] = col_vec[irow];
        }
      }
    }

    //assign_num_level_l(l, v, 0.0, nr, nc, nrow, ncol);
    row_stride = stride;
    col_stride = stride;
    assign_num_level_l_cuda(nrow,        ncol,
                            nr,          nr,
                            row_stride,  col_stride,
                            irow,        icol,
                            v, ldv, 
                            0.0);

    //subtract_level_l(l - 1, v, work.data(), nr, nc, nrow, ncol);

    row_stride = Pstride;
    col_stride = Pstride;
    substract_level_l_cuda(nrow,        ncol, 
                     nr,          nc,
                     row_stride,  col_stride,
                     irow,        icol,
                     v,           ldv
                     work.data(), ldwork,)
  }
  //    //std::cout  << "last step" << "\n";
}


void postp_2D_cuda(const int nr, const int nc, const int nrow, const int ncol,
              const int l_target, double *v, std::vector<double> &work,
              std::vector<double> &coords_x, std::vector<double> &coords_y,
              std::vector<double> &row_vec, std::vector<double> &col_vec) {
  int ldv = ncol;
  int ldwork = ncol;

  int row_stride;
  int col_stride;

  int * irow  = new int[nr];
  int * irowP = new int[nrow-nr];
  int irow_ptr  = 0;
  int irowP_ptr = 0;

  for (int i = 0; i < nr; i++) {
    int irow_r = get_lindex_cuda(nr, nrow, i);
    irow[irow_ptr] = irow_r;
    if (irow_ptr > 0 && irow[irow_ptr - 1] != irow[irow_ptr] - 1) {
      irowP[irowP_ptr] = irow[irow_ptr] - 1;
      irowP_ptr ++;
    } 
    irow_ptr++;
  }

  std::cout << "irow: ";
  for (int i = 0; i < nr; i++) std::cout << irow[i] << ", ";
  std::cout << std::endl;

  std::cout << "irowP: ";
  for (int i = 0; i < nrow-nr; i++) std::cout << irowP[i] << ", ";
  std::cout << std::endl;


  int * icol  = new int[nc];
  int * icolP = new int[ncol-nc];
  int icol_ptr  = 0;
  int icolP_ptr = 0;

  for (int i = 0; i < nc; i++) {
    int icol_r = get_lindex_cuda(nc, ncol, i);
    icol[icol_ptr] = icol_r;
    if (icol_ptr > 0 && icol[icol_ptr - 1] != icol[icol_ptr] - 1) {
      icolP[icolP_ptr] = icol[icol_ptr] - 1;
      icolP_ptr ++;
    } 
    icol_ptr++;
  }

  std::cout << "icol: ";
  for (int i = 0; i < nc; i++) std::cout << icol[i] << ", ";
  std::cout << std::endl;

  std::cout << "icolP: ";
  for (int i = 0; i < ncol-nc; i++) std::cout << icolP[i] << ", ";
  std::cout << std::endl;

 //mgard_cannon::copy_level(nrow, ncol, 0, v, work);
  row_stride = 1;
  col_stride = 1;
  copy_level_cuda(nrow,        ncol, 
                  row_stride,  col_stride,
                  v,           ldv,
                  work.data(), ldwork);

  //assign_num_level_l(0, work.data(), 0.0, nr, nc, nrow, ncol);

  row_stride = 1;
  col_stride = 1;
  assign_num_level_l_cuda(nrow,        ncol,
                          nr,          nr,
                          row_stride,  col_stride,
                          irow,        icol,
                          work.data(), ldwork
                          0.0);

  row_stride = 1;
  col_stride = 1;
  mass_matrix_multiply_row_cuda(nrow,        ncol, 
                                row_stride,  col_stride,
                                work.data(), ldwork
                                coords_x);
  row_stride = 1;
  col_stride = 1;
  restriction_first_row_cuda(nrow,        ncol, 
                             row_stride,  icolP, nc,
                             work.data(), ldwork
                             coords_x);

  row_stride = 1;
  col_stride = 1;
  solve_tridiag_M_l_row_cuda(nrow,        ncol,
                             nr,          nc,
                             row_stride,  col_stride,
                             irow,        icol,
                             work.data(), ldwork
                             coords_x);

  for (int irow = 0; irow < nrow; ++irow) {
    //        int ir = get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard_common::get_index(ncol, irow, jcol)];
    }

    //mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);

    //restriction_first(row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard_common::get_index(ncol, irow, jcol)] = row_vec[jcol];
    }
  }

  for (int irow = 0; irow < nr; ++irow) {
    int ir = get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
    }

    //mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  //   //   //std::cout  << "recomposing-colsweep" << "\n";

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) // check if we have 1-D array..
  {
    
    row_stride = 1;
    col_stride = 1;
    mass_mult_l_col_cuda(nrow,        ncol,
                         nr,          nc,
                         row_stride,  col_stride,
                         irow,        icol,
                         work.data(), ldwork
                         coords_y);

    row_stride = 1;
    col_stride = 1;
    restriction_first_col_cuda(nrow,        ncol, 
                               irowP,       nr,   col_stride,
                               work.data(), ldwork
                               coords_y);

    row_stride = 1;
    col_stride = 1;
    solve_tridiag_M_l_col_cuda(nrow,        ncol,
                               nr,          nc,
                               row_stride,  col_stride,
                               irow,        icol,
                               work.data(), ldwork
                               coords_y);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      //      int jr  = get_lindex(nc,  ncol,  jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard_common::get_index(ncol, irow, jcol)];
      }

      //mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);

      //mgard_gen::restriction_first(col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard_common::get_index(ncol, irow, jcol)] = col_vec[irow];
      }
    }

    for (int jcol = 0; jcol < nc; ++jcol) {
      int jr = get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard_common::get_index(ncol, irow, jr)];
      }

      //mgard_gen::solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow);
      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard_common::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }
  }

  //subtract_level_l(0, work.data(), v, nr, nc, nrow, ncol); // do -(Qu - zl)
  row_stride = 1;
  col_stride = 1;
  subtract_level_l_cuda(nrow,        ncol, 
                        nr,          nc,
                        row_stride,  col_stride,
                        irow,        icol,
                        work.data(), ldwork,
                        v,           ldv);


  //        //std::cout  << "recomposing-rowsweep2" << "\n";

  //     //   //int Pstride = stride/2; //finer stride
  
  row_stride = 1;
  col_stride = 1;
  prolongate_last_row_cuda(nrow,       ncol, 
                           row_stride, icolP, nc,
                           v,          ldv,
                           coords_x);

  //   //   // row-sweep
  for (int irow = 0; irow < nr; ++irow) {
    int ir = get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
    }

    mgard_gen::prolongate_last(row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) {

    row_stride = 1;
    col_stride = 1;
    prolongate_last_col_cuda(nrow,       ncol, 
                             irowP, nr, col_stride,
                             v,     ldv,
                             coords_y);


    for (int jcol = 0; jcol < ncol; ++jcol) {
      // int jr  = get_lindex(nc,  ncol,  jcol);
      for (int irow = 0; irow < nrow; ++irow) // copy all rows
      {
        col_vec[irow] = work[mgard_common::get_index(ncol, irow, jcol)];
      }

      //mgard_gen::prolongate_last(col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard_common::get_index(ncol, irow, jcol)] = col_vec[irow];
      }
    }
  }

  


  //assign_num_level_l(0, v, 0.0, nr, nc, nrow, ncol);

  assign_num_level_l_cuda(nrow,        ncol,
                          nr,          nr,
                          row_stride,  col_stride,
                          irow,        icol,
                          work.data(), ldwork
                          0.0);

  //mgard_cannon::subtract_level(nrow, ncol, 0, v, work.data());
  row_stride = 1;
  col_stride = 1;
  mgard_cannon::substract_level_cuda(nrow,       ncol, 
                                     row_stride, col_stride,
                                     v,          ldv, 
                                     work,       ldwork); 
}





} //end namespace mgard_gen

} //end namespace mard_2d