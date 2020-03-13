template <typename T>
__device__ T
_dist(T * dcoord, double x, double y);

template <typename T>
mgard_cuda_ret 
org_to_pow2p1(int nrow,     int ncol,    int nfib,
              int nr,      int nc,       int nf,
              int * dirow,  int * dicol, int * difib, 
              T * dv,  int lddv1,   int lddv2,
              T * dcv, int lddcv1,  int lddcv2,
              int B, mgard_cuda_handle & handle, 
              int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret 
org_to_pow2p1(int nrow,     int ncol,
             int nr,        int nc,
             int * dirow,  int * dicol,
             T * dv,  int lddv,
             T * dcv, int lddcv,
             int B, mgard_cuda_handle & handle, 
             int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret 
org_to_pow2p1(int nrow,    int nr,
              int * dirow, 
              T * dv, T * dcv,
              int B, mgard_cuda_handle & handle, 
              int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret 
pow2p1_to_org(int nrow,     int ncol,    int nfib,   
              int nr,       int nc,      int nf,  
              int * dirow,  int * dicol, int * difib, 
              T * dcv, int lddcv1,  int lddcv2,
              T * dv,  int lddv1,   int lddv2,
              int B, mgard_cuda_handle & handle, 
              int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret 
pow2p1_to_org(int nrow,     int ncol,
              int nr,       int nc,
              int * dirow,  int * dicol,
              T * dcv,  int lddcv,
              T * dv, int lddv,
              int B, mgard_cuda_handle & handle, 
              int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret 
pow2p1_to_org(int nrow, int nr,      
              int * dirow,  
              T * dcv, T * dv,
              int B, mgard_cuda_handle & handle, 
              int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret 
pow2p1_to_cpt(int nrow,       int ncol,       int nfib, 
              int row_stride, int col_stride, int fib_stride, 
              T * dv,    int lddv1,      int lddv2,
              T * dcv,   int lddcv1,     int lddcv2,
              int B, mgard_cuda_handle & handle, 
              int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret 
pow2p1_to_cpt(int nrow,      int ncol, 
              int row_stride, int col_stride,
              T * dv,    int lddv, 
              T * dcv,   int lddcv,
              int B, mgard_cuda_handle & handle, 
              int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret 
pow2p1_to_cpt(int nrow,  int row_stride, 
              T * dv, T * dcv,
              int B, mgard_cuda_handle & handle, 
              int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret
cpt_to_pow2p1(int nrow,      int ncol,       int nfib, 
              int row_stride, int col_stride, int fib_stride, 
              T * dcv,   int lddcv1,     int lddcv2,
              T * dv,    int lddv1,      int lddv2,
              int B, mgard_cuda_handle & handle, 
              int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret
cpt_to_pow2p1(int nrow, int ncol, 
              int row_stride, int col_stride, 
              T * dcv, int lddcv,
              T * dv, int lddv,
              int B, mgard_cuda_handle & handle, 
              int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret
cpt_to_pow2p1(int nrow, int row_stride, 
              T * dcv, T * dv,
              int B, mgard_cuda_handle & handle, 
              int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret
calc_cpt_dist(int nrow, int row_stride, 
              T * dcoord, T * ddist,
              int B, mgard_cuda_handle & handle, 
              int queue_idx, bool profile);
