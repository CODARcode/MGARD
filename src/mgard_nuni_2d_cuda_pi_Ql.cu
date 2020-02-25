#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>
#include <cmath>

namespace mgard_2d {
namespace mgard_gen {  

__global__ void 
_pi_Ql_cuda_sm(int nr,           int nc,
             int row_stride,   int col_stride,
             int * irow,       int * icol,
             double * dv,      int lddv, 
             double * ddist_x, double * ddist_y) {

  register int c0 = blockIdx.x * blockDim.x;
  register int c0_stride = c0 * col_stride;
  register int r0 = blockIdx.y * blockDim.y;
  register int r0_stride = r0 * row_stride;

  register int c_sm = threadIdx.x;
  register int r_sm = threadIdx.y;

  extern __shared__ double sm[]; // size: (blockDim.x + 1) * (blockDim.y + 1)
  int ldsm = blockDim.x + 1;
  double * v_sm = sm;
  double * dist_x_sm = sm + (blockDim.x + 1) * (blockDim.y + 1);
  double * dist_y_sm = dist_x_sm + nc;

  for (int r = r0; r < nr - 1; r += blockDim.y * gridDim.y) {
    for (int c = c0; c < nc - 1; c += blockDim.x * gridDim.x) {
      /* Load data */
      if (c + c_sm < nc)
      v_sm[r_sm * ldsm + c_sm] = dv[(r + r_sm) * row_stride * lddv + (c + c_sm) * col_stride];
      if (r_sm == 0) {
        v_sm[blockDim.y * ldsm + c_sm] = dv[blockDim.y * row_stride * lddv + (c + c_sm) * col_stride];
        dist_x_sm[c_sm] = ddist_x[c + c_sm];
        dist_y_sm[c_sm] = ddist_y[c + c_sm];
      }
      if (c_sm == 0) {
        v_sm[r_sm * ldsm + blockDim.x] = dv[(r + r_sm) * row_stride * lddv + blockDim.x * col_stride];
      }
      if (r_sm == 0 && c_sm == 0) {
        v_sm[blockDim.y * ldsm + blockDim.x] = dv[blockDim.y * row_stride * lddv + blockDim.x * col_stride];
      }
      __syncthreads();

      /* Compute */
      if (r_sm % 2 == 0 && c_sm % 2 != 0) {
        double h1 = dist_x_sm[c_sm - 1];
        double h2 = dist_x_sm[c_sm + 1];
        v_sm[r_sm * ldsm + c_sm] -= (h2 * v_sm[r_sm * ldsm + (c_sm - 1)] + 
                                     h1 * v_sm[r_sm * ldsm + (c_sm + 1)])/
                                    (h1 + h2);
      } 
      if (r_sm % 2 != 0 && c_sm % 2 == 0) {
        double h1 = dist_y_sm[r_sm - 1];
        double h2 = dist_y_sm[r_sm + 1];
        v_sm[r_sm * ldsm + c_sm] -= (h2 * v_sm[(r_sm - 1) * ldsm + c_sm] +
                                     h1 * v_sm[(r_sm + 1) * ldsm + c_sm])/
                                    (h1 + h2);
      } 
      if (r_sm % 2 != 0 && c_sm % 2 != 0) {
        double h1_col = dist_x_sm[c_sm - 1];
        double h2_col = dist_x_sm[c_sm + 1];
        double h1_row = dist_y_sm[r_sm - 1];
        double h2_row = dist_y_sm[r_sm + 1];
        v_sm[r_sm * ldsm + c_sm] -= (v_sm[(r_sm - 1) * ldsm + (c_sm - 1)] * h2_col * h2_row +
                                     v_sm[(r_sm - 1) * ldsm + (c_sm + 1)] * h1_col * h2_row + 
                                     v_sm[(r_sm + 1) * ldsm + (c_sm - 1)] * h2_col * h1_row + 
                                     v_sm[(r_sm + 1) * ldsm + (c_sm + 1)] * h1_col * h1_row)/
                                    ((h1_col + h2_col) * (h1_row + h2_row));
      }
      /* extra computaion for global boarder */

      /* Store results */

    }
  }


}
}