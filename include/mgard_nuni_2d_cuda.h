#include "mgard_nuni.h"
#include "mgard.h"

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
void copy_level_cuda(const int nrow, const int ncol, const int l, double *v,
                std::vector<double> &work);
void mass_matrix_multiply_cuda(const int l, std::vector<double> &v,
                          const std::vector<double> &coords);

void mass_matrix_multiply_row_cuda(int nrow,       int ncol, 
							   	   int row_stride, int col_stride,
                                   double * v,    int ldv,
                                   double * coords_x);

void mass_matrix_multiply_col_cuda(int nrow,       int ncol, 
			                       int row_stride, int col_stride,
                                   double * v,    int ldv,
                                   double * coords_y);


}

namespace mgard_gen {
__host__ __device__ int
get_lindex_cuda(const int n, const int no, const int i);

__host__ __device__ double 
get_h_l_cuda(const double * coords, const int n,
             const int no, int i, int stride);

__host__ __device__ double *
get_ref_cuda(double * v, const int n, const int no,
                       const int i);

__host__ __device__ double *
get_ref_row_cuda(double * v, int ldv, const int n, const int no,
                       const int i);

__host__ __device__ double *
get_ref_col_cuda(double * v, int ldv, const int n, const int no,
                       const int i);

void pi_lminus1_first_cuda(std::vector<double> &v, const std::vector<double> &coords,
                      int n, int no);

void pi_Ql_first_cuda(const int nr, const int nc, const int nrow, const int ncol,
                 const int l, double *v, const std::vector<double> &coords_x,
                 const std::vector<double> &coords_y,
                 std::vector<double> &row_vec, std::vector<double> &col_vec);

void assign_num_level_l_cuda(const int l, double *v, double num, int nr, int nc,
                        const int nrow, const int ncol);


void restriction_first_cuda(std::vector<double> &v, std::vector<double> &coords,
                       int n, int no);

void restriction_first_row_cuda(int nrow,       int ncol, 
								int row_stride, int * icolP, int nc,
                                double * v,     int ldv,
                                double * coords_x);

void restriction_first_col_cuda(int nrow,       int ncol, 
								int * irowP, int nr, int col_stride,
                                double * v,    int ldv,
                                double * coords_y);

void solve_tridiag_M_l_cuda(const int l, std::vector<double> &v,
                       std::vector<double> &coords, int n, int no);

void
solve_tridiag_M_l_row_cuda(int nrow,   int ncol,
                           int nr,    int nc,
                           int * irow, int col_stride,
                           double * v, int ldv, double * coords_x);

void
solve_tridiag_M_l_col_cuda(int nrow,   int ncol,
                           int nr,    int nc,
                           int row_stride, int * icol,
                           double * v, int ldv, double * coords_y);

// void add_level_l_cuda(const int l, double *v, double *work, int nr, int nc, int nrow,
//                  int ncol);
void add_level_l_cuda(int nrow,       int ncol, 
               int nr,          int nc,
               int row_stride, int col_stride,
               int * irow,     int * icol,
               double * v,    int ldv, 
               double * work, int ldwork);

void prep_2D_cuda     (const int nr, const int nc, const int nrow, const int ncol,
             const int l_target, double *v, std::vector<double> &work,
             std::vector<double> &coords_x, std::vector<double> &coords_y,
             std::vector<double> &row_vec, std::vector<double> &col_vec);

void refactor_2D_cuda(const int nr, const int nc, const int nrow, const int ncol,
                 const int l_target, double *v, std::vector<double> &work,
                 std::vector<double> &coords_x, std::vector<double> &coords_y,
                 std::vector<double> &row_vec, std::vector<double> &col_vec);


}

}

