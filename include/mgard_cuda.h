#include "mgard_cuda_helper.h"
namespace mgard
{

template <typename T>
unsigned char *
refactor_qz_cuda(int nrow, int ncol, int nfib, const T *u,
                           int &outsize, T tol,
                           int B, mgard_cuda_handle & handle, bool profile);

template <typename T>
T *recompose_udq_cuda(int nrow, int ncol, int nfib, unsigned char *data,
                           int data_len, int B,
                           mgard_cuda_handle & handle,
                           bool profile, T dummy);

template <typename T>
unsigned char *
refactor_qz_2D_cuda (int nrow, int ncol, const T *u, int &outsize, T tol, int opt,
					 int B, mgard_cuda_handle & handle, bool profile);

template <typename T>
T* 
recompose_udq_2D_cuda(int nrow, int ncol, unsigned char *data, int data_len, int opt,
					  int B, mgard_cuda_handle & handle, bool profile, T dummy);

void
refactor_cuda (const int nrow, const int ncol, 
              const int l_target, 
              double * dv, int lddv, 
              double * dwork, int lddwork);

void
recompose_cuda (const int nrow, const int ncol, 
               const int l_target, 
               double * dv,    int lddv,
               double * dwork, int lddwork);


mgard_ret 
pi_Ql_cuda(int nrow,       int ncol, 
	      int row_stride, int col_stride,
	      double * v,     int ldv);
mgard_ret  
copy_level_cuda(int nrow,       int ncol,
	       	   int row_stride, int col_stride,
               double * v,     int ldv, 
	       	   double * work,  int ldwork);
mgard_ret 
assign_num_level_cuda(int nrow,   int ncol, 
	             int row_stride, int col_stride,
	             double * v,     int ldv,
	             double num);
mgard_ret  
mass_matrix_multiply_row_cuda(int nrow,       int ncol,
                             int row_stride, int col_stride,
                             double * work,  int ldwork);
mgard_ret  
restriction_row_cuda(int nrow,       int ncol, 
                    int row_stride, int col_stride,
                    double * work,  int ldwork);
mgard_ret  
solve_tridiag_M_row_cuda(int nrow,       int ncol,
                        int row_stride, int col_stride,
                        double * work,  int ldwork);
mgard_ret 
mass_matrix_multiply_col_cuda(int nrow,       int ncol, 
					         int row_stride, int col_stride, 
					         double * work,  int ldwork);
mgard_ret 
restriction_col_cuda(int nrow, int ncol, 
					int row_stride, int col_stride, 
					double * work, int ldwork);
mgard_ret  
solve_tridiag_M_col_cuda(int nrow, int ncol, 
						int row_stride, int col_stride, 
						double * work, int ldwork);
mgard_ret  
add_level_cuda(int nrow, int ncol, 
			  int row_stride, int col_stride, 
			  double * v, int ldv, 
			  double * work, int ldwork);
mgard_ret   
subtract_level_cuda(int nrow, int ncol,
                   int row_stride, int col_stride, 
                   double * v, int ldv, 
                   double * work, int ldwork);
mgard_ret  
interpolate_from_level_nMl_row_cuda(int nrow, int ncol, 
	                               int row_stride, int col_stride,
	                               double * work, int ldwork);
mgard_ret   
interpolate_from_level_nMl_col_cuda(int nrow, int ncol, 
	                               int row_stride, int col_stride,
	                               double * work, int ldwork);
}
