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
_pi_Ql_cuda_cpt_sm(int nr,           int nc,           int nf, 
                   int row_stride,   int col_stride,   int fib_stride, 
                   T * dv,      int lddv1,        int lddv2, 
                   T * ddist_r, T * ddist_c, T * ddist_f) {
  

  register int r0 = blockIdx.z * blockDim.z;
  register int c0 = blockIdx.y * blockDim.y;
  register int f0 = blockIdx.x * blockDim.x;
    
  register int total_row = ceil((double)nr/(row_stride));
  register int total_col = ceil((double)nc/(col_stride));
  register int total_fib = ceil((double)nf/(fib_stride));

  register int r_sm = threadIdx.z;
  register int c_sm = threadIdx.y;
  register int f_sm = threadIdx.x;

  register int r_sm_ex = blockDim.z;
  register int c_sm_ex = blockDim.y;
  register int f_sm_ex = blockDim.x;

  register int r_gl;
  register int c_gl;
  register int f_gl;

  register int r_gl_ex;
  register int c_gl_ex;
  register int f_gl_ex;

  extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  T * sm = reinterpret_cast<T *>(smem);

  //extern __shared__ double sm[]; // size: (blockDim.x + 1) * (blockDim.y + 1) * (blockDim.z + 1)
  int ldsm1 = blockDim.x + 1;
  int ldsm2 = blockDim.y + 1;
  T * v_sm = sm;
  T * dist_f_sm = sm + (blockDim.x + 1) * (blockDim.y + 1) * (blockDim.z + 1);
  T * dist_c_sm = dist_f_sm + blockDim.x;
  T * dist_r_sm = dist_c_sm + blockDim.y;

  for (int r = r0; r < total_row - 1; r += blockDim.z * gridDim.z) {
    r_gl = (r + r_sm) * row_stride;
    r_gl_ex = (r + blockDim.z) * row_stride;
    for (int c = c0; c < total_col - 1; c += blockDim.y * gridDim.y) {
      c_gl = (c + c_sm) * col_stride;
      c_gl_ex = (c + blockDim.y) * col_stride;
      for (int f = f0; f < total_fib - 1; f += blockDim.x * gridDim.x) {
        f_gl = (f + f_sm) * fib_stride;
        f_gl_ex = (f + blockDim.x) * fib_stride;
        /* Load v */
        if (r + r_sm < total_row && c + c_sm < total_col && f + f_sm < total_fib) {
          // load cubic
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl)];
          // load extra surfaces
          if (r + blockDim.z < total_row && r_sm == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)] = dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl)];
          }
          if (c + blockDim.y < total_col && c_sm == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)] = dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl)];
          }
          if (f + blockDim.x < total_fib && f_sm == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)] = dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl_ex)];
          }
          // load extra edges
          if (c + blockDim.y < total_col && f + blockDim.x < total_fib && c_sm == 0 && f_sm == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)] = dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)];
          }
          if (r + blockDim.z < total_row && f + blockDim.x < total_fib && r_sm == 0 && f_sm == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)] = dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)];
          }
          if (r + blockDim.z < total_row && c + blockDim.y < total_col && r_sm == 0 && c_sm == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)] = dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)];
          }
          // load extra vertex
          if (r + blockDim.z < total_row && c + blockDim.y < total_col && f + blockDim.x < total_fib &&
              r_sm == 0 && c_sm == 0 && f_sm == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm_ex)] = dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)];
          }

          // load dist
          if (c_sm == 0 && f_sm == 0 && r + r_sm < total_row) {
            dist_r_sm[r_sm] = ddist_r[r + r_sm];
          }
          if (r_sm == 0 && f_sm == 0 && c + c_sm < total_col) {
            dist_c_sm[c_sm] = ddist_c[c + c_sm];
          }
          if (c_sm == 0 && r_sm == 0 && f + f_sm < total_fib) {
            dist_f_sm[f_sm] = ddist_f[f + f_sm];
          }
          __syncthreads();

          T h1_row = dist_r_sm[r_sm - 1];
          T h2_row = dist_r_sm[r_sm];
          T h1_col = dist_c_sm[c_sm - 1];
          T h2_col = dist_c_sm[c_sm];
          T h1_fib = dist_f_sm[f_sm - 1];
          T h2_fib = dist_f_sm[f_sm];

          /* Compute */
          // edges
          if (r_sm % 2 != 0 && c_sm % 2 == 0 && f_sm % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] -= 
              (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm, f_sm)] * h2_row + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm, f_sm)] * h1_row) / 
              (h1_row + h2_row);
          }
          if (r_sm % 2 == 0 && c_sm % 2 != 0 && f_sm % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] -= 
              (v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm-1, f_sm)] * h2_col + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm+1, f_sm)] * h1_col) / 
              (h1_col + h2_col);
          }
          if (r_sm % 2 == 0 && c_sm % 2 == 0 && f_sm % 2 != 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] -= 
              (v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm-1)] * h2_fib + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm+1)] * h1_fib) / 
              (h1_fib + h2_fib);
          }
          // surfaces
          if (r_sm % 2 == 0 && c_sm % 2 != 0 && f_sm % 2 != 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] -= 
              (v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm-1, f_sm-1)] * h2_col * h2_fib + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm+1, f_sm-1)] * h1_col * h2_fib + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm-1, f_sm+1)] * h2_col * h1_fib +
               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm+1, f_sm+1)] * h1_col * h1_fib) / 
              ((h1_col + h2_col) * (h1_fib + h2_fib));
          }
          if (r_sm % 2 != 0 && c_sm % 2 == 0 && f_sm % 2 != 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] -= 
              (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm, f_sm-1)] * h2_row * h2_fib + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm, f_sm-1)] * h1_row * h2_fib + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm, f_sm+1)] * h2_row * h1_fib +
               v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm, f_sm+1)] * h1_row * h1_fib) / 
              ((h1_row + h2_row) * (h1_fib + h2_fib));
          }
          if (r_sm % 2 != 0 && c_sm % 2 != 0 && f_sm % 2 == 0) {
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] -= 
              (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm-1, f_sm)] * h2_row * h2_col + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm-1, f_sm)] * h1_row * h2_col + 
               v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm+1, f_sm)] * h2_row * h1_col +
               v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm)] * h1_row * h1_col) / 
              ((h1_row + h2_row) * (h1_col + h2_col));
          }

          // core
          if (r_sm % 2 != 0 && c_sm % 2 != 0 && f_sm % 2 != 0) {

            T x00 = (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm-1, f_sm-1)] * h2_fib + 
                          v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm-1, f_sm+1)] * h1_fib) /
                         (h2_fib + h1_fib);
            T x01 = (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm+1, f_sm-1)] * h2_fib + 
                          v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm+1, f_sm+1)] * h1_fib) /
                         (h2_fib + h1_fib);
            T x10 = (v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm-1, f_sm-1)] * h2_fib + 
                          v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm-1, f_sm+1)] * h1_fib) /
                         (h2_fib + h1_fib);
            T x11 = (v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm-1)] * h2_fib + 
                          v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm+1)] * h1_fib) /
                         (h2_fib + h1_fib);
            T y0  = (h2_col * x00 + h1_col * x01) / (h2_col + h1_col);
            T y1  = (h2_col * x10 + h1_col * x11) / (h2_col + h1_col);
            T z   = (h2_row * y0 + h1_row * y1) / (h2_row + h1_row);
            // if (v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] == 272) {
            //   printf("-1 -1 -1 %f \n", v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm-1, f_sm-1)]);
            //   printf("-1 -1 +1 %f \n", v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm-1, f_sm+1)]);

            //   printf("-1 +1 -1 %f \n", v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm+1, f_sm-1)]);
            //   printf("-1 +1 +1 %f \n", v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm+1, f_sm+1)]);

            //   printf("+1 -1 -1 %f \n", v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm-1, f_sm-1)]);
            //   printf("+1 -1 +1 %f \n", v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm-1, f_sm+1)]);

            //   printf("+1 +1 -1 %f \n", v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm-1)]);
            //   printf("+1 +1 +1 %f \n", v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm+1)]);
            //   printf("core: %f z: %f\n", v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)], z);
            // }
            

            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] -= z;
          }

          // store
          dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl)] = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];

          /* extra computaion for global boarder */
          // extra surface
          if (r + blockDim.z == total_row - 1) {
            if (r_sm == 0) {
              //edge
              if (c_sm % 2 != 0 && f_sm % 2 == 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm-1, f_sm)] * h2_col + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm+1, f_sm)] * h1_col) / 
                  (h1_col + h2_col);
              }
              if (c_sm % 2 == 0 && f_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm-1)] * h2_fib + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm+1)] * h1_fib) / 
                  (h1_fib + h2_fib);
              }
              //surface
              if (c_sm % 2 != 0 && f_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm-1, f_sm-1)] * h2_col * h2_fib + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm+1, f_sm-1)] * h1_col * h2_fib + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm-1, f_sm+1)] * h2_col * h1_fib +
                   v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm+1, f_sm+1)] * h1_col * h1_fib) / 
                  ((h1_col + h2_col) * (h1_fib + h2_fib));
              }
              dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl)] = v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)];
            }
          }

          if (c + blockDim.y == total_col - 1) {
            if (c_sm == 0) {
              //edge
              if (r_sm % 2 != 0 && f_sm % 2 == 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm_ex, f_sm)] * h2_row + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm_ex, f_sm)] * h1_row) / 
                  (h1_row + h2_row);
              }
              if (r_sm % 2 == 0 && f_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm-1)] * h2_fib + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm+1)] * h1_fib) / 
                  (h1_fib + h2_fib);
              }
              //surface
              if (r_sm % 2 != 0 && f_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm_ex, f_sm-1)] * h2_row * h2_fib + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm_ex, f_sm-1)] * h1_row * h2_fib + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm_ex, f_sm+1)] * h2_row * h1_fib +
                   v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm_ex, f_sm+1)] * h1_row * h1_fib) / 
                  ((h1_row + h2_row) * (h1_fib + h2_fib));
              }
              dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl)] = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)];
            }
          }

          if (f + blockDim.x == total_fib - 1) {
            if (f_sm == 0) {
              //edge
              if (r_sm % 2 != 0 && c_sm % 2 == 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm, f_sm_ex)] * h2_row + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm, f_sm_ex)] * h1_row) / 
                  (h1_row + h2_row);
              }
              if (r_sm % 2 == 0 && c_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm-1, f_sm_ex)] * h2_col + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm+1, f_sm_ex)] * h1_col) / 
                  (h1_col + h2_col);
              }
              //surface
              if (r_sm % 2 != 0 && c_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm-1, f_sm_ex)] * h2_row * h2_col + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm-1, f_sm_ex)] * h1_row * h2_col + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm+1, f_sm_ex)] * h2_row * h1_col +
                   v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm_ex)] * h1_row * h1_col) / 
                  ((h1_row + h2_row) * (h1_col + h2_col));
              }
              dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl_ex)] = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)];
            }
          }

          //edge
          if (c + blockDim.y == total_col - 1 && f + blockDim.x == total_fib - 1) {
            if (c_sm == 0 && f_sm == 0) {
              if (r_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm-1, c_sm_ex, f_sm_ex)] * h2_row + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm_ex, f_sm_ex)] * h1_row) / 
                  (h1_row + h2_row);
              }
              dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)] = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)];
            }
          }

          if (r + blockDim.z == total_row - 1 && f + blockDim.x == total_fib - 1) {
            if (r_sm == 0 && f_sm == 0) {
              if (c_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm-1, f_sm_ex)] * h2_col + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm+1, f_sm_ex)] * h1_col) / 
                  (h1_col + h2_col);
              }
              dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)] = v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)];
            }
          }

          if (r + blockDim.z == total_row - 1 && c + blockDim.y == total_col - 1) {
            if (r_sm == 0 && c_sm == 0) {
              if (f_sm % 2 != 0) {
                v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)] -= 
                  (v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm-1)] * h2_fib + 
                   v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm+1)] * h1_fib) / 
                  (h1_fib + h2_fib);  
              }
              dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)] = v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)];
            }
          }
        }// restrict boundary
      } // end f
    } // end c
  } // end r
}

template <typename T>
mgard_cuda_ret 
pi_Ql_cuda_cpt_sm(int nr,           int nc,           int nf, 
                  int row_stride,   int col_stride,   int fib_stride, 
                  T * dv,      int lddv1,        int lddv2, 
                  T * ddist_r, T * ddist_c, T * ddist_f,
                  int B) {
  
  int total_row = ceil((double)nr/(row_stride));
  int total_col = ceil((double)nc/(col_stride));
  int total_fib = ceil((double)nf/(fib_stride));

  int total_thread_z = total_row - 1;
  int total_thread_y = total_col - 1;
  int total_thread_x = total_fib - 1;

  int tbz = min(B, total_thread_z);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);


  size_t sm_size = ((B+1) * (B+1) * (B+1) + 3 * B) * sizeof(T);

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

  _pi_Ql_cuda_cpt_sm<<<blockPerGrid, threadsPerBlock, sm_size>>>(nr,         nc,         nf,
                                                                 row_stride, col_stride, fib_stride, 
                                                                 dv,         lddv1,      lddv2, 
                                                                 ddist_r,    ddist_c,     ddist_f);


  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}

template mgard_cuda_ret 
pi_Ql_cuda_cpt_sm<double>(int nr,           int nc,           int nf, 
                          int row_stride,   int col_stride,   int fib_stride, 
                          double * dv,      int lddv1,        int lddv2, 
                          double * ddist_r, double * ddist_c, double * ddist_f,
                          int B);
template mgard_cuda_ret 
pi_Ql_cuda_cpt_sm<float>(int nr,           int nc,           int nf, 
                          int row_stride,   int col_stride,   int fib_stride, 
                          float * dv,      int lddv1,        int lddv2, 
                          float * ddist_r, float * ddist_c, float * ddist_f,
                          int B);

}