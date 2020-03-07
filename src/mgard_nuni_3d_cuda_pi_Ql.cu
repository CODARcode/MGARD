#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>
#include <cmath>

namespace mgard_gen {

__global__ void 
_pi_Ql_cuda_cpt_sm(int nf, int nr, int nc,
                   int fib_stride, int row_stride, int col_stride,
                   double * dv, int lddv1, int lddv2, 
                   double * ddist_x, double * ddist_y, double * ddist_z) {
  register int c0 = blockIdx.x * blockDim.x;
  register int c0_stride = c0 * col_stride;
  register int r0 = blockIdx.y * blockDim.y;
  register int r0_stride = r0 * row_stride;
  register int f0 = blockIdx.z * blockDim.z;
  register int f0_stride = f0 * fib_stride;

  register int total_fib = ceil((double)nf/(fib_stride));
  register int total_row = ceil((double)nr/(row_stride));
  register int total_col = ceil((double)nc/(col_stride));

  register int c_sm = threadIdx.x;
  register int r_sm = threadIdx.y;
  register int f_sm = threadIdx.z;

  extern __shared__ double sm[]; // size: (blockDim.x + 1) * (blockDim.y + 1) * (blockDim.z + 1)
  int ldsm1 = blockDim.x + 1;
  int ldsm2 = blockDim.y + 1;
  double * dist_x_sm = sm + (blockDim.x + 1) * (blockDim.y + 1) * (blockDim.z + 1);
  double * dist_y_sm = dist_x_sm + blockDim.x;
  double * dist_z_sm = dist_y_sm + blockDim.y;

  for (int f = f0; r < total_fib - 1; f += blockDim.z * gridDim.z) {
    for (int r = r0; r < total_row - 1; r += blockDim.y * gridDim.y) {
      for (int c = c0; c < total_col - 1; c += blockDim.x * gridDim.x) {
        /* Load v */
        if (c + c_sm < total_col && r + r_sm < total_row && f + f_sm < total_fib) {
          // load cubic
          v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm] = 
            dv[(f + f_sm) * fib_stride * lddv1 * lddv2 + (r + r_sm) * row_stride * lddv1 + (c + c_sm) * col_stride];
          // load extra surfaces
          if (c_sm == 0 && c + blockDim.x < total_col) {
            v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + blockDim.x] = 
              dv[(f + f_sm) * fib_stride * lddv1 * lddv2 + (r + r_sm) * row_stride * lddv1 + (c + blockDim.x) * col_stride];
          }
          if (r_sm == 0 && r + blockDim.y < total_row) {
            v_sm[f_sm * ldsm1 * ldsm2 + blockDim.y * ldsm1 + c_sm] = 
              dv[(f + f_sm) * fib_stride * lddv1 * lddv2 + (r + blockDim.y) * row_stride * lddv1 + (c + c_sm) * col_stride];
          }
          if (f_sm == 0 && f + blockDim.z < total_fib) {
            v_sm[blockDim.z * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm] = 
              dv[(f + blockDim.z) * fib_stride * lddv1 * lddv2 + (r + r_sm) * row_stride * lddv1 + (c + c_sm) * col_stride];
          }
          // load extra edges
          if (r_sm == 0 && c_sm == 0 && r + blockDim.y < total_row && c + blockDim.x < total_col) {
            v_sm[f_sm * ldsm1 * ldsm2 + blockDim.y * ldsm1 + blockDim.x] = 
              dv[(f + f_sm) * fib_stride * lddv1 * lddv2 + (r + blockDim.y) * row_stride * lddv1 + (c + blockDim.x) * col_stride];
          }
          if (f_sm == 0 && c_sm == 0 && f + blockDim.z < total_fib && c + blockDim.x < total_col) {
            v_sm[blockDim.z * ldsm1 * ldsm2 + r_sm * ldsm1 + blockDim.x] = 
              dv[(f + blockDim.z) * fib_stride * lddv1 * lddv2 + (r + r_sm) * row_stride * lddv1 + (c + blockDim.x) * col_stride];
          }
          if (r_sm == 0 && f_sm == 0 && r + blockDim.y < total_row && f + blockDim.z < total_fib) {
            v_sm[blockDim.z * ldsm1 * ldsm2 + blockDim.y * ldsm1 + c_sm] = 
              dv[(f + blockDim.z) * fib_stride * lddv1 * lddv2 + (r + blockDim.y) * row_stride * lddv1 + (c + c_sm) * col_stride];
          }
          // load extra vertex
          if (r_sm == 0 && c_sm == 0 && f_sm == 0 && r + blockDim.y < total_row && c + blockDim.x < total_col && f + blockDim.z < total_fib) {
            v_sm[blockDim.z * ldsm1 * ldsm2 + blockDim.y * ldsm1 + blockDim.x] = 
              dv[(f + blockDim.z) * fib_stride * lddv1 * lddv2 + (r + blockDim.y) * row_stride * lddv1 + (c + blockDim.x) * col_stride];
          }
        }
        // load dist
        if (r_sm == 0 && f_sm == 0 && c + c_sm < total_col) {
          dist_x_sm[c_sm] = ddist_x[c + c_sm];
        }
        if (c_sm == 0 && f_sm == 0 && r + r_sm < total_row) {
          dist_y_sm[r_sm] = ddist_y[r + r_sm];
        }
        if (c_sm == 0 && r_sm == 0 && f + f_sm < total_fib) {
          dist_z_sm[f_sm] = ddist_z[f + f_sm];
        }
        __syncthreads();

        /* Compute */
        // edges
        if (c_sm % 2 != 0 && r_sm % 2 == 0 && f_sm % 2 == 0) {
          double h1 = dist_x_sm[c_sm - 1];
          double h2 = dist_x_sm[c_sm];
          v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm] -= 
            (h2 * v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + (c_sm - 1)] + 
             h1 * v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + (c_sm + 1)])/ 
            (h1 + h2);
          dv[(f + f_sm) * fib_stride * lddv1 * lddv2 + (r + r_sm) * row_stride * lddv1 + (c + c_sm) * col_stride] =
            v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm];
        }

        if (c_sm % 2 == 0 && r_sm % 2 != 0 && f_sm % 2 == 0) {
          double h1 = dist_y_sm[r_sm - 1];
          double h2 = dist_y_sm[r_sm];
          v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm] -= 
            (h2 * v_sm[f_sm * ldsm1 * ldsm2 + (r_sm - 1) * ldsm1 + c_sm] + 
             h1 * v_sm[f_sm * ldsm1 * ldsm2 + (r_sm + 1) * ldsm1 + c_sm])/ 
            (h1 + h2);
          dv[(f + f_sm) * fib_stride * lddv1 * lddv2 + (r + r_sm) * row_stride * lddv1 + (c + c_sm) * col_stride] =
            v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm];
        }

        if (c_sm % 2 == 0 && r_sm % 2 == 0 && f_sm % 2 != 0) {
          double h1 = dist_z_sm[f_sm - 1];
          double h2 = dist_z_sm[f_sm];
          v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm] -= 
            (h2 * v_sm[(f_sm - 1) * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm] + 
             h1 * v_sm[(f_sm + 1) * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm])/ 
            (h1 + h2);
          dv[(f + f_sm) * fib_stride * lddv1 * lddv2 + (r + r_sm) * row_stride * lddv1 + (c + c_sm) * col_stride] =
            v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm];
        }

        // surfaces
        if (c_sm % 2 != 0 && r_sm % 2 != 0 && f_sm % 2 == 0) {
          double h1_col = dist_x_sm[c_sm - 1];
          double h2_col = dist_x_sm[c_sm];
          double h1_row = dist_y_sm[r_sm - 1];
          double h2_row = dist_y_sm[r_sm];
          v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm] -= 
            (v_sm[f_sm * ldsm1 * ldsm2 + (r_sm - 1) * ldsm1 + (c_sm - 1)] * h2_row * h2_col + 
             v_sm[f_sm * ldsm1 * ldsm2 + (r_sm - 1) * ldsm1 + (c_sm + 1)] * h2_row * h1_col + 
             v_sm[f_sm * ldsm1 * ldsm2 + (r_sm + 1) * ldsm1 + (c_sm - 1)] * h1_row * h2_col +
             v_sm[f_sm * ldsm1 * ldsm2 + (r_sm + 1) * ldsm1 + (c_sm + 1)] * h1_row * h1_col)/ 
            ((h1_col + h2_col) * (h1_row + h2_row));
          dv[(f + f_sm) * fib_stride * lddv1 * lddv2 + (r + r_sm) * row_stride * lddv1 + (c + c_sm) * col_stride] =
            v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm];
        }

        if (c_sm % 2 != 0 && r_sm % 2 == 0 && f_sm % 2 != 0) {
          double h1_col = dist_x_sm[c_sm - 1];
          double h2_col = dist_x_sm[c_sm];
          double h1_fib = dist_z_sm[f_sm - 1];
          double h2_fib = dist_z_sm[f_sm];
          v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm] -= 
            (v_sm[(f_sm - 1) * ldsm1 * ldsm2 + r_sm * ldsm1 + (c_sm - 1)] * h2_fib * h2_col + 
             v_sm[(f_sm - 1) * ldsm1 * ldsm2 + r_sm * ldsm1 + (c_sm + 1)] * h2_fib * h1_col + 
             v_sm[(f_sm + 1) * ldsm1 * ldsm2 + r_sm * ldsm1 + (c_sm - 1)] * h1_fib * h2_col +
             v_sm[(f_sm + 1) * ldsm1 * ldsm2 + r_sm * ldsm1 + (c_sm + 1)] * h1_fib * h1_col)/ 
            ((h1_col + h2_col) * (h1_fib + h2_fib));
          dv[(f + f_sm) * fib_stride * lddv1 * lddv2 + (r + r_sm) * row_stride * lddv1 + (c + c_sm) * col_stride] =
            v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm];
        }

        if (c_sm % 2 == 0 && r_sm % 2 != 0 && f_sm % 2 != 0) {
          double h1_row = dist_y_sm[r_sm - 1];
          double h2_row = dist_y_sm[r_sm];
          double h1_fib = dist_z_sm[f_sm - 1];
          double h2_fib = dist_z_sm[f_sm];
          v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm] -= 
            (v_sm[(f_sm - 1) * ldsm1 * ldsm2 + (r_sm - 1) * ldsm1 + c_sm] * h2_fib * h2_row + 
             v_sm[(f_sm - 1) * ldsm1 * ldsm2 + (r_sm + 1) * ldsm1 + c_sm] * h2_fib * h1_row + 
             v_sm[(f_sm + 1) * ldsm1 * ldsm2 + (r_sm - 1) * ldsm1 + c_sm] * h1_fib * h2_row +
             v_sm[(f_sm + 1) * ldsm1 * ldsm2 + (r_sm + 1) * ldsm1 + c_sm] * h1_fib * h1_row)/ 
            ((h1_col + h2_col) * (h1_fib + h2_fib));
          dv[(f + f_sm) * fib_stride * lddv1 * lddv2 + (r + r_sm) * row_stride * lddv1 + (c + c_sm) * col_stride] =
            v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm];
        }

        // core
        if (c_sm % 2 != 0 && r_sm % 2 != 0 && f_sm % 2 != 0) {
          double h1_col = dist_x_sm[c_sm - 1];
          double h2_col = dist_x_sm[c_sm];
          double h1_row = dist_y_sm[r_sm - 1];
          double h2_row = dist_y_sm[r_sm];
          double h1_fib = dist_z_sm[f_sm - 1];
          double h2_fib = dist_z_sm[f_sm];
          double x00 = (h2_col * v_sm[(f_sm - 1) * ldsm1 * ldsm2 + (r_sm - 1) * ldsm1 + (c_sm - 1)] +
                        h1_col * v_sm[(f_sm - 1) * ldsm1 * ldsm2 + (r_sm - 1) * ldsm1 + (c_sm + 1)]) /
                       (h2_col + h1_col);
          double x01 = (h2_col * v_sm[(f_sm - 1) * ldsm1 * ldsm2 + (r_sm + 1) * ldsm1 + (c_sm - 1)] +
                        h1_col * v_sm[(f_sm - 1) * ldsm1 * ldsm2 + (r_sm + 1) * ldsm1 + (c_sm + 1)]) /
                       (h2_col + h1_col);
          double x10 = (h2_col * v_sm[(f_sm + 1) * ldsm1 * ldsm2 + (r_sm - 1) * ldsm1 + (c_sm - 1)] +
                        h1_col * v_sm[(f_sm + 1) * ldsm1 * ldsm2 + (r_sm - 1) * ldsm1 + (c_sm + 1)]) /
                       (h2_col + h1_col);
          double x11 = (h2_col * v_sm[(f_sm + 1) * ldsm1 * ldsm2 + (r_sm + 1) * ldsm1 + (c_sm - 1)] +
                        h1_col * v_sm[(f_sm + 1) * ldsm1 * ldsm2 + (r_sm + 1) * ldsm1 + (c_sm + 1)]) /
                       (h2_col + h1_col);
          double y0  = (h2_row * x00 + h1_row * x01) / (h2_row + h1_row);
          double y1  = (h2_row * x10 + h1_row * x11) / (h2_row + h1_row);
          double z   = (h2_fib * y0 + h1_fib * y1) / (h2_fib + h1_fib);
          dv[(f + f_sm) * fib_stride * lddv1 * lddv2 + (r + r_sm) * row_stride * lddv1 + (c + c_sm) * col_stride] = z;
        }

        /* extra computaion for global boarder */
        //edge
        if (c + blockDim.x == total_col - 1 && r + blockDim.y == total_row - 1) {
          if (f_sm % 2 != 0 && c_sm == 0 && r_sm == 0) {
            double h1 = dist_z_sm[f_sm - 1];
            double h2 = dist_z_sm[f_sm];
            v_sm[f_sm * ldsm1 * ldsm2 + blockDim.y * ldsm1 + blockDim.x] -= 
              (h2 * v_sm[(f_sm - 1) * ldsm1 * ldsm2 + blockDim.y * ldsm1 + blockDim.x] + 
               h1 * v_sm[(f_sm + 1) * ldsm1 * ldsm2 + blockDim.y * ldsm1 + blockDim.x])/ 
              (h1 + h2);
            dv[(f + f_sm) * fib_stride * lddv1 * lddv2 + (r + blockDim.y) * row_stride * lddv1 + (c + blockDim.x) * col_stride] =
              v_sm[f_sm * ldsm1 * ldsm2 + blockDim.y * ldsm1 + blockDim.x];
          }
        }
        if (c + blockDim.x == total_col - 1 && f + blockDim.z == total_fib - 1) {
          if (r_sm % 2 != 0 && c_sm == 0 && f_sm == 0) {
            double h1 = dist_y_sm[r_sm - 1];
            double h2 = dist_y_sm[r_sm];
            v_sm[blockDim.z * ldsm1 * ldsm2 + r_sm * ldsm1 + blockDim.x] -= 
              (h2 * v_sm[blockDim.z * ldsm1 * ldsm2 + (r_sm - 1) * ldsm1 + blockDim.x] + 
               h1 * v_sm[blockDim.z * ldsm1 * ldsm2 + (r_sm + 1) * ldsm1 + blockDim.x])/ 
              (h1 + h2);
            dv[(f + blockDim.z) * fib_stride * lddv1 * lddv2 + (r + r_sm) * row_stride * lddv1 + (c + blockDim.x) * col_stride] =
              v_sm[blockDim.z * ldsm1 * ldsm2 + r_sm * ldsm1 + blockDim.x];
          }
        }

        if (r + blockDim.y == total_row - 1 && f + blockDim.z == total_fib - 1) {
          if (c_sm % 2 != 0 && r_sm == 0 && f_sm == 0) {
            double h1 = dist_x_sm[c_sm - 1];
            double h2 = dist_x_sm[c_sm];
            v_sm[blockDim.z * ldsm1 * ldsm2 + blockDim.y * ldsm1 + c_sm] -= 
              (h2 * v_sm[blockDim.z * ldsm1 * ldsm2 + blockDim.y * ldsm1 + (c_sm - 1)] + 
               h1 * v_sm[blockDim.z * ldsm1 * ldsm2 + blockDim.y * ldsm1 + (c_sm + 1)])/ 
              (h1 + h2);
            dv[(f + blockDim.z) * fib_stride * lddv1 * lddv2 + (r + blockDim.y) * row_stride * lddv1 + (c + c_sm) * col_stride] =
              v_sm[blockDim.z * ldsm1 * ldsm2 + blockDim.y * ldsm1 + c_sm];
          }
        }

        // surface
        if (c + blockDim.x == total_col - 1) {
          if (c_sm == 0 && r_sm % 2 != 0 && f_sm % 2 != 0) {
            double h1_row = dist_y_sm[r_sm - 1];
            double h2_row = dist_y_sm[r_sm];
            double h1_fib = dist_z_sm[f_sm - 1];
            double h2_fib = dist_z_sm[f_sm];
            v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + blockDim.x] -= 
              (v_sm[(f_sm - 1) * ldsm1 * ldsm2 + (r_sm - 1) * ldsm1 + blockDim.x] * h2_fib * h2_row + 
               v_sm[(f_sm - 1) * ldsm1 * ldsm2 + (r_sm + 1) * ldsm1 + blockDim.x] * h2_fib * h1_row + 
               v_sm[(f_sm + 1) * ldsm1 * ldsm2 + (r_sm - 1) * ldsm1 + blockDim.x] * h1_fib * h2_row +
               v_sm[(f_sm + 1) * ldsm1 * ldsm2 + (r_sm + 1) * ldsm1 + blockDim.x] * h1_fib * h1_row)/ 
              ((h1_col + h2_col) * (h1_fib + h2_fib));
            dv[(f + f_sm) * fib_stride * lddv1 * lddv2 + (r + r_sm) * row_stride * lddv1 + (c + blockDim.x) * col_stride] =
              v_sm[f_sm * ldsm1 * ldsm2 + r_sm * ldsm1 + blockDim.x];

          }
        }

        if (r + blockDim.y == total_row - 1) {
          if (r_sm == 0 && c_sm % 2 != 0 && f_sm % 2 != 0) {
            double h1_col = dist_x_sm[c_sm - 1];
            double h2_col = dist_x_sm[c_sm];
            double h1_fib = dist_z_sm[f_sm - 1];
            double h2_fib = dist_z_sm[f_sm];
            v_sm[f_sm * ldsm1 * ldsm2 + blockDim.y * ldsm1 + c_sm] -= 
              (v_sm[(f_sm - 1) * ldsm1 * ldsm2 + blockDim.y * ldsm1 + (c_sm - 1)] * h2_fib * h2_col + 
               v_sm[(f_sm - 1) * ldsm1 * ldsm2 + blockDim.y * ldsm1 + (c_sm + 1)] * h2_fib * h1_col + 
               v_sm[(f_sm + 1) * ldsm1 * ldsm2 + blockDim.y * ldsm1 + (c_sm - 1)] * h1_fib * h2_col +
               v_sm[(f_sm + 1) * ldsm1 * ldsm2 + blockDim.y * ldsm1 + (c_sm + 1)] * h1_fib * h1_col)/ 
              ((h1_col + h2_col) * (h1_fib + h2_fib));
            dv[(f + f_sm) * fib_stride * lddv1 * lddv2 + (r + blockDim.y) * row_stride * lddv1 + (c + c_sm) * col_stride] =
              v_sm[f_sm * ldsm1 * ldsm2 + blockDim.y * ldsm1 + c_sm];

          }
        }

        if (f + blockDim.z == total_fib - 1) {
          if (f_sm == 0 && c_sm % 2 != 0 && r_sm % 2 != 0) {
            double h1_col = dist_x_sm[c_sm - 1];
            double h2_col = dist_x_sm[c_sm];
            double h1_row = dist_y_sm[r_sm - 1];
            double h2_row = dist_y_sm[r_sm];
            v_sm[blockDim.z * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm] -= 
              (v_sm[blockDim.z * ldsm1 * ldsm2 + (r_sm - 1) * ldsm1 + (c_sm - 1)] * h2_row * h2_col + 
               v_sm[blockDim.z * ldsm1 * ldsm2 + (r_sm - 1) * ldsm1 + (c_sm + 1)] * h2_row * h1_col + 
               v_sm[blockDim.z * ldsm1 * ldsm2 + (r_sm + 1) * ldsm1 + (c_sm - 1)] * h1_row * h2_col +
               v_sm[blockDim.z * ldsm1 * ldsm2 + (r_sm + 1) * ldsm1 + (c_sm + 1)] * h1_row * h1_col)/ 
              ((h1_col + h2_col) * (h1_fib + h2_fib));
            dv[(f + blockDim.z) * fib_stride * lddv1 * lddv2 + (r + r_sm) * row_stride * lddv1 + (c + c_sm) * col_stride] =
              v_sm[blockDim.z * ldsm1 * ldsm2 + r_sm * ldsm1 + c_sm];
          }
        }
      }
    }
  }
}


mgard_cuda_ret 
pi_Ql_cuda_cpt_sm(int nf, int nr, int nc,
                  int fib_stride, int row_stride, int col_stride,
                  double * dv, int lddv1, int lddv2, 
                  double * ddist_x, double * ddist_y, double * ddist_z,
                  int B) {
 
  int total_fib = ceil((double)nf/(fib_stride));
  int total_row = ceil((double)nr/(row_stride));
  int total_col = ceil((double)nc/(col_stride));
  int total_thread_z = total_fib - 1;
  int total_thread_y = total_row - 1;
  int total_thread_x = total_col - 1;

  int tbz = min(B, total_thread_z);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);


  size_t sm_size = ((B+1) * (B+1) * (B+1) + 3 * B) * sizeof(double);

  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);

  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx<< std::endl;



  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _pi_Ql_cuda_cpt_sm<<<blockPerGrid, threadsPerBlock, sm_size>>>(nf, nr, nc,
                                                                 fib_stride, row_stride, col_stride,
                                                                 dv, lddv1, lddv2, 
                                                                 ddist_x, ddist_y, ddist_z);


  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}




void pi_Ql3D(const int nr, const int nc, const int nf, const int nrow,
             const int ncol, const int nfib, const int l, double *v,
             const std::vector<double> &coords_x,
             const std::vector<double> &coords_y,
             const std::vector<double> &coords_z, std::vector<double> &row_vec,
             std::vector<double> &col_vec, std::vector<double> &fib_vec) {

  int stride = std::pow(2, l); // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride * 2; // coarser stride

  //  std::vector<double> row_vec(ncol), col_vec(nrow)   ;

  for (int kfib = 0; kfib < nf; kfib += Cstride) {
    int kf = get_lindex(nf, nfib,
                        kfib); // get the real location of logical index irow
    for (int irow = 0; irow < nr;
         irow += Cstride) // Do the rows existing  in the coarser level
    {
      int ir = get_lindex(nr, nrow,
                          irow); // get the real location of logical index irow
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = v[mgard_common::get_index3(ncol, nfib, ir, jcol, kf)];
      }

      pi_lminus1_l(l, row_vec, coords_x, nc, ncol);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        v[mgard_common::get_index3(ncol, nfib, ir, jcol, kf)] = row_vec[jcol];
      }
    }
  }

  if (nrow > 1) {
    for (int kfib = 0; kfib < nf; kfib += Cstride) {
      int kf = get_lindex(nf, nfib, kfib);
      for (int jcol = 0; jcol < nc;
           jcol += Cstride) // Do the columns existing  in the coarser level
      {
        int jr = get_lindex(nc, ncol, jcol);
        for (int irow = 0; irow < nrow; ++irow) {
          //                int irow_r = get_lindex(nr, nrow, irow);
          col_vec[irow] = v[mgard_common::get_index3(ncol, nfib, irow, jr, kf)];
        }
        pi_lminus1_l(l, col_vec, coords_y, nr, nrow);
        for (int irow = 0; irow < nrow; ++irow) {
          v[mgard_common::get_index3(ncol, nfib, irow, jr, kf)] = col_vec[irow];
        }
      }
    }
  }

  if (nfib > 1) {
    for (int irow = 0; irow < nr;
         irow += Cstride) // Do the columns existing  in the coarser level
    {
      int ir = get_lindex(nr, nrow,
                          irow); // get the real location of logical index irow
      for (int jcol = 0; jcol < nc; jcol += Cstride) {
        int jr = get_lindex(nc, ncol, jcol);
        for (int kfib = 0; kfib < nfib; ++kfib) {
          fib_vec[kfib] = v[mgard_common::get_index3(ncol, nfib, ir, jr, kfib)];
        }
        pi_lminus1_l(l, fib_vec, coords_z, nf, nfib);
        for (int kfib = 0; kfib < nfib; ++kfib) {
          v[mgard_common::get_index3(ncol, nfib, ir, jr, kfib)] = fib_vec[kfib];
        }
      }
    }
  }

  //        Now the new-new stuff, xy-plane
  for (int kfib = 0; kfib < nf; kfib += Cstride) {
    int kf = get_lindex(nf, nfib, kfib);
    for (int irow = stride; irow < nr; irow += Cstride) {
      int ir1 = get_lindex(nr, nrow, irow - stride);
      int ir = get_lindex(nr, nrow, irow);
      int ir2 = get_lindex(nr, nrow, irow + stride);

      for (int jcol = stride; jcol < nc; jcol += Cstride) {

        int jr1 = get_lindex(nc, ncol, jcol - stride);
        int jr = get_lindex(nc, ncol, jcol);
        int jr2 = get_lindex(nc, ncol, jcol + stride);

        double q11 = v[mgard_common::get_index3(ncol, nfib, ir1, jr1, kf)];
        double q12 = v[mgard_common::get_index3(ncol, nfib, ir2, jr1, kf)];
        double q21 = v[mgard_common::get_index3(ncol, nfib, ir1, jr2, kf)];
        double q22 = v[mgard_common::get_index3(ncol, nfib, ir2, jr2, kf)];

        double x1 = 0.0; // relative coordinate axis centered at irow - Cstride,
                         // jcol - Cstride
        double y1 = 0.0;
        double x2 = mgard_common::get_dist(coords_x, jr1, jr2);
        double y2 = mgard_common::get_dist(coords_y, ir1, ir2);

        double x = mgard_common::get_dist(coords_x, jr1, jr);
        double y = mgard_common::get_dist(coords_y, ir1, ir);
        double temp =
            mgard_common::interp_2d(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
        //              //std::cout  << temp <<"\n";
        v[mgard_common::get_index3(ncol, nfib, ir, jr, kf)] -= temp;
      }
    }
  }

  // // //        Now the new-new stuff, xz-plane
  for (int irow = 0; irow < nr; irow += Cstride) {
    int irr = get_lindex(nr, nrow, irow);
    for (int jcol = stride; jcol < nc; jcol += Cstride) {
      int ir1 = get_lindex(nc, ncol, jcol - stride);
      int ir = get_lindex(nc, ncol, jcol);
      int ir2 = get_lindex(nc, ncol, jcol + stride);

      for (int kfib = stride; kfib < nf; kfib += Cstride) {
        int jr1 = get_lindex(nf, nfib, kfib - stride);
        int jr = get_lindex(nf, nfib, kfib);
        int jr2 = get_lindex(nf, nfib, kfib + stride);

        double q11 = v[mgard_common::get_index3(ncol, nfib, irr, ir1, jr1)];
        double q12 = v[mgard_common::get_index3(ncol, nfib, irr, ir2, jr1)];
        double q21 = v[mgard_common::get_index3(ncol, nfib, irr, ir1, jr2)];
        double q22 = v[mgard_common::get_index3(ncol, nfib, irr, ir2, jr2)];

        double x1 = 0.0; // relative coordinate axis centered at irow - Cstride,
                         // jcol - Cstride
        double y1 = 0.0;
        double x2 = mgard_common::get_dist(coords_z, jr1, jr2);
        double y2 = mgard_common::get_dist(coords_x, ir1, ir2);

        double x = mgard_common::get_dist(coords_z, jr1, jr);
        double y = mgard_common::get_dist(coords_x, ir1, ir);
        double temp =
            mgard_common::interp_2d(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
        //              //std::cout  << temp <<"\n";
        v[mgard_common::get_index3(ncol, nfib, irr, ir, jr)] -= temp;
      }
    }
  }

  //     //        Now the new-new stuff, yz-plane
  for (int jcol = 0; jcol < nc; jcol += Cstride) {
    int jrr = get_lindex(nc, ncol, jcol);
    for (int irow = stride; irow < nr; irow += Cstride) {
      int ir1 = get_lindex(nr, nrow, irow - stride);
      int ir = get_lindex(nr, nrow, irow);
      int ir2 = get_lindex(nr, nrow, irow + stride);

      for (int kfib = stride; kfib < nf; kfib += Cstride) {
        int jr1 = get_lindex(nf, nfib, kfib - stride);
        int jr = get_lindex(nf, nfib, kfib);
        int jr2 = get_lindex(nf, nfib, kfib + stride);

        double q11 = v[mgard_common::get_index3(ncol, nfib, ir1, jrr, jr1)];
        double q12 = v[mgard_common::get_index3(ncol, nfib, ir2, jrr, jr1)];
        double q21 = v[mgard_common::get_index3(ncol, nfib, ir1, jrr, jr2)];
        double q22 = v[mgard_common::get_index3(ncol, nfib, ir2, jrr, jr2)];

        double x1 = 0.0; // relative coordinate axis centered at irow - Cstride,
                         // jcol - Cstride
        double y1 = 0.0;
        double x2 = mgard_common::get_dist(coords_z, jr1, jr2);
        double y2 = mgard_common::get_dist(coords_y, ir1, ir2);

        double x = mgard_common::get_dist(coords_z, jr1, jr);
        double y = mgard_common::get_dist(coords_y, ir1, ir);
        double temp =
            mgard_common::interp_2d(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
        //              //std::cout  << temp <<"\n";
        v[mgard_common::get_index3(ncol, nfib, ir, jrr, jr)] -= temp;
      }
    }
  }

  // ///    new-new-new stuff

  for (int irow = stride; irow < nr; irow += Cstride) {
    int ir1 = get_lindex(nr, nrow, irow - stride);
    int ir = get_lindex(nr, nrow, irow);
    int ir2 = get_lindex(nr, nrow, irow + stride);

    for (int jcol = stride; jcol < nc; jcol += Cstride) {
      int jr1 = get_lindex(nc, ncol, jcol - stride);
      int jr = get_lindex(nc, ncol, jcol);
      int jr2 = get_lindex(nc, ncol, jcol + stride);

      for (int kfib = stride; kfib < nf; kfib += Cstride) {

        int kr1 = get_lindex(nf, nfib, kfib - stride);
        int kr = get_lindex(nf, nfib, kfib);
        int kr2 = get_lindex(nf, nfib, kfib + stride);

        double x1 = 0.0;
        double y1 = 0.0;
        double z1 = 0.0;

        double x2 = mgard_common::get_dist(coords_x, jr1, jr2);
        double y2 = mgard_common::get_dist(coords_y, ir1, ir2);
        double z2 = mgard_common::get_dist(coords_z, kr1, kr2);

        double x = mgard_common::get_dist(coords_x, jr1, jr);
        double y = mgard_common::get_dist(coords_y, ir1, ir);
        double z = mgard_common::get_dist(coords_z, kr1, kr);

        double q000 = v[mgard_common::get_index3(ncol, nfib, ir1, jr1, kr1)];
        double q100 = v[mgard_common::get_index3(ncol, nfib, ir1, jr2, kr1)];
        double q110 = v[mgard_common::get_index3(ncol, nfib, ir1, jr2, kr2)];

        double q010 = v[mgard_common::get_index3(ncol, nfib, ir1, jr1, kr2)];

        double q001 = v[mgard_common::get_index3(ncol, nfib, ir2, jr1, kr1)];
        double q101 = v[mgard_common::get_index3(ncol, nfib, ir2, jr2, kr1)];
        double q111 = v[mgard_common::get_index3(ncol, nfib, ir2, jr2, kr2)];

        double q011 = v[mgard_common::get_index3(ncol, nfib, ir2, jr1, kr2)];

        double temp =
            mgard_common::interp_3d(q000, q100, q110, q010, q001, q101, q111,
                                    q011, x1, x2, y1, y2, z1, z2, x, y, z);

        v[mgard_common::get_index3(ncol, nfib, ir, jr, kr)] -= temp;
      }
    }
  }
}

}