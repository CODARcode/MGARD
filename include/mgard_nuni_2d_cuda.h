#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_cuda_helper.h"
//#include "mgard_cuda_helper_internal.h"

namespace mgard_2d {
namespace mgard_common {
__host__ __device__ int 
get_index_cuda(const int ncol, const int i, const int j);

__host__ __device__ double
interp_2d_cuda(double q11, double q12, double q21, double q22,
                        double x1, double x2, double y1, double y2, double x,
                        double y);

__host__ __device__ double
get_h_cuda(const double * coords, int i, int stride);

__host__ __device__ double 
get_dist_cuda(const double * coords, int i, int j);
}


namespace mgard_cannon {

mgard_cuda_ret 
copy_level_cuda(int nrow,       int ncol, 
                int row_stride, int col_stride,
                double * dv,    int lddv,
                double * dwork, int lddwork);

void mass_matrix_multiply_cuda(const int l, std::vector<double> &v,
                          const std::vector<double> &coords);

mgard_cuda_ret 
mass_matrix_multiply_row_cuda(int nrow,       int ncol, 
                              int row_stride, int col_stride,
                              double * dv,    int lddv,
                              double * dcoords_x);
mgard_cuda_ret 
mass_matrix_multiply_col_cuda(int nrow,       int ncol, 
                              int row_stride, int col_stride,
                              double * dv,    int lddv,
                              double * dcoords_y);

mgard_cuda_ret 
subtract_level_cuda(int nrow,       int ncol, 
                    int row_stride, int col_stride,
                    double * dv,    int lddv, 
                    double * dwork, int lddwork);

}

namespace mgard_gen {
__host__ __device__ int
get_lindex_cuda(const int n, const int no, const int i);

__host__ __device__ double 
get_h_l_cuda(const double * coords, const int n,
             const int no, int i, int stride);

__host__ __device__ double *
get_ref_cuda(double * v, const int n, const int no, const int i);

__host__ __device__ double *
get_ref_row_cuda(double * v, int ldv, const int n, const int no, const int i);

__host__ __device__ double *
get_ref_col_cuda(double * v, int ldv, const int n, const int no, const int i);


mgard_cuda_ret 
pi_Ql_first_cuda(const int nrow,     const int ncol,
                 const int nr,       const int nc,
                 int * dirow,        int * dicol,
                 int * dirowP,       int * dicolP,
                 double * dcoords_x, double * dcoords_y,
                 double * dv,        const int lddv);

mgard_cuda_ret 
pi_Ql_cuda(int nrow,           int ncol,
           int nr,             int nc,
           int row_stride,     int col_stride,
           int * dirow,        int * dicol,
           double * dv,        int lddv, 
           double * dcoords_x, double * dcoords_y);

mgard_cuda_ret 
copy_level_l_cuda(int nrow,       int ncol,
                  int nr,         int nc,
                  int row_stride, int col_stride,
                  int * dirow,    int * dicol,
                  double * dv,    int lddv,
                  double * dwork, int lddwork);

mgard_cuda_ret 
assign_num_level_l_cuda(int nrow,           int ncol,
                        int nr,             int nc,
                        int row_stride,     int col_stride,
                        int * dirow,        int * dicol,
                        double * dv,        int lddv,
                        double num);

mgard_cuda_ret 
mass_mult_l_row_cuda(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     double * dv,    int lddv,
                     double * dcoords_x);

mgard_cuda_ret 
mass_mult_l_col_cuda(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     double * dv,    int lddv,
                     double * dcoords_y);

mgard_cuda_ret 
restriction_l_row_cuda(int nrow,       int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * dirow,     int * dicol,
                       double * dv,     int lddv,
                       double * dcoords_x);

mgard_cuda_ret 
restriction_l_col_cuda(int nrow,       int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * dirow,    int * dicol,
                       double * dv,    int lddv,
                       double * dcoords_y);


void restriction_first_cuda(std::vector<double> &v, std::vector<double> &coords,
                            int n, int no);

mgard_cuda_ret 
restriction_first_row_cuda(int nrow,       int ncol, 
                           int row_stride, int * dicolP, int nc,
                           double * dv,    int lddv,
                           double * dcoords_x);
mgard_cuda_ret 
restriction_first_col_cuda(int nrow,       int ncol, 
                           int * dirowP, int nr, int col_stride,
                           double * dv,    int lddv,
                           double * dcoords_y);

void solve_tridiag_M_l_cuda(const int l, std::vector<double> &v,
                            std::vector<double> &coords, int n, int no);

mgard_cuda_ret
solve_tridiag_M_l_row_cuda(int nrow,       int ncol,
                           int nr,         int nc,
                           int row_stride, int col_stride,
                           int * dirow,    int * dicol,
                           double * dv,     int lddv, 
                           double * dcoords_x);

mgard_cuda_ret
solve_tridiag_M_l_col_cuda(int nrow,       int ncol,
                           int nr,         int nc,
                           int row_stride, int col_stride,
                           int * dirow,    int * dicol,
                           double * dv,    int lddv, 
                           double * dcoords_y);

mgard_cuda_ret
add_level_l_cuda(int nrow,       int ncol, 
                 int nr,         int nc,
                 int row_stride, int col_stride,
                 int * dirow,    int * dicol,
                 double * dv,    int lddv, 
                 double * dwork, int lddwork);

mgard_cuda_ret 
subtract_level_l_cuda(int nrow,       int ncol, 
                      int nr,         int nc,
                      int row_stride, int col_stride,
                      int * dirow,    int * dicol,
                      double * dv,    int lddv, 
                      double * dwork, int lddwork);

mgard_cuda_ret 
prolongate_l_row_cuda(int nrow,       int ncol,
                      int nr,         int nc,
                      int row_stride, int col_stride,
                      int * dirow,    int * dicol,
                      double * dv,    int lddv,
                      double * dcoords_x);

mgard_cuda_ret 
prolongate_l_col_cuda(int nrow,        int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * dirow,    int * dicol,
                       double * dv,    int lddv,
                       double * dcoords_y);

mgard_cuda_ret 
prolongate_last_row_cuda(int nrow,       int ncol, 
                         int nr,         int nc,
                         int row_stride, int col_stride,
                         int * dirow,    int * dicolP,
                         double * dv,    int lddv,
                         double * dcoords_x);

mgard_cuda_ret 
prolongate_last_col_cuda(int nrow,       int ncol,
                         int nr,         int nc,
                         int row_stride, int col_stride,
                         int * dirowP,   int * dicol, 
                         double * dv,    int lddv,
                         double * dcoords_y);


void 
prep_2D_cuda(const int nrow,     const int ncol,
             const int nr,       const int nc, 
             int * dirow,        int * dicol,
             int * dirowP,       int * dicolP,
             double * dv,        int lddv, 
             double * dwork,     int lddwork,
             double * dcoords_x, double * dcoords_y);

void 
refactor_2D_cuda(const int l_target,
                 const int nrow,     const int ncol,
                 const int nr,       const int nc, 
                 int * dirow,        int * dicol,
                 int * dirowP,       int * dicolP,
                 double * dv,        int lddv, 
                 double * dwork,     int lddwork,
                 double * dcoords_x, double * dcoords_y);

void 
recompose_2D_cuda(const int l_target,
                  const int nrow,     const int ncol,
                  const int nr,       const int nc, 
                  int * dirow,        int * dicol,
                  int * dirowP,       int * dicolP,
                  double * dv,        int lddv, 
                  double * dwork,     int lddwork,
                  double * dcoords_x, double * dcoords_y);


void 
postp_2D_cuda(const int nrow,     const int ncol,
              const int nr,       const int nc, 
              int * dirow,        int * dicol,
              int * dirowP,       int * dicolP,
              double * dv,        int lddv, 
              double * dwork,     int lddwork,
              double * dcoords_x, double * dcoords_y);
}

}

