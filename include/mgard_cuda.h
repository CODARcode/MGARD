
#ifndef MGRAD_RET
#define MGRAD_RET
struct mgard_ret {
	int info;
	double time;
	mgard_ret (): info(0), time (0.0) {}
	mgard_ret (int info, double time) {
		this->info = info;
		this->time = time;
	}
};
#endif

namespace mgard
{

unsigned char *
refactor_qz_2D_cuda (int nrow, int ncol, const double *u, int &outsize, double tol);
double* 
recompose_udq_2D_cuda(int nrow, int ncol, unsigned char *data, int data_len);

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
quantize_2D_iterleave_cuda (int nrow, int ncol, 
						   double * v, int lddv, 
						   int * work, int ldwork, 
						   double norm, double tol);

mgard_ret  
dequantize_2D_iterleave_cuda (int nrow, int ncol, 
	                         double * v, int ldv, 
	                         int * work, int ldwork);
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
