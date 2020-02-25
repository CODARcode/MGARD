__device__ double
_dist(double * dcoord, double x, double y);

mgard_cuda_ret 
org_to_pow2p1(int nrow,     int ncol,
              int nr,       int nc,
              int * dirow,  int * dicol,
              double * dv,  int lddv,
              double * dcv, int lddcv);

mgard_cuda_ret 
org_to_pow2p1(int nrow,    int nr,
              int * dirow, 
              double * dv, double * dcv);

mgard_cuda_ret 
pow2p1_to_org(int nrow,     int ncol,
                       int nr,       int nc,
                       int * dirow,  int * dicol,
                       double * dcv,  int lddcv,
                       double * dv, int lddv);
mgard_cuda_ret 
pow2p1_to_org(int nrow, int nr,      
              int * dirow,  
              double * dcv, double * dv);

mgard_cuda_ret 
pow2p1_to_cpt(int nrow,      int ncol, 
              int row_stride, int col_stride,
              double * dv,    int lddv, 
              double * dcv,   int lddcv);

mgard_cuda_ret 
pow2p1_to_cpt(int nrow,  int row_stride, 
              double * dv, double * dcv);

mgard_cuda_ret
cpt_to_pow2p1(int nrow, int ncol, 
              int row_stride, int col_stride, 
              double * dcv, int lddcv,
              double * dv, int lddv);

mgard_cuda_ret
cpt_to_pow2p1(int nrow, int row_stride, 
              double * dcv, double * dv);

mgard_cuda_ret
calc_cpt_dist(int nrow, int row_stride, 
              double * dcoord, double * ddist);