__device__ double
_dist(double * dcoord, double x, double y);

mgard_cuda_ret 
org_to_pow2p1(int nrow, int ncol, int nfib,
              int nr,   int nc,   int nf,
              int * dirow,  int * dicol, int * difib, 
              double * dv,  int lddv1, int lddv2,
              double * dcv, int lddcv1, int lddcv2);

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
pow2p1_to_org(int nrow,     int ncol,    int nfib,   
              int nr,       int nc,      int nf,  
              int * dirow,  int * dicol, int * difib, 
              double * dcv, int lddcv1,  int lddcv2,
              double * dv,  int lddv1,   int lddv2);

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
pow2p1_to_cpt(int nrow,      int ncol,        int nfib, 
              int row_stride, int col_stride, int fib_stride, 
              double * dv,    int lddv1, int lddv2,
              double * dcv,   int lddcv1, int lddcv2);
mgard_cuda_ret 
pow2p1_to_cpt(int nrow,      int ncol, 
              int row_stride, int col_stride,
              double * dv,    int lddv, 
              double * dcv,   int lddcv);

mgard_cuda_ret 
pow2p1_to_cpt(int nrow,  int row_stride, 
              double * dv, double * dcv);

mgard_cuda_ret
cpt_to_pow2p1(int nrow,      int ncol,        int nfib, 
              int row_stride, int col_stride, int fib_stride, 
              double * dcv, int lddcv1, int lddcv2,
              double * dv, int lddv1, int lddv2);

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