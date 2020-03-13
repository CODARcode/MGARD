namespace mgard
{
template <typename T>
mgard_cuda_ret  
dequantize_2D_iterleave_cuda (int nrow,   int ncol, 
                             T * dv, int lddv, 
                             int * dwork, int lddwork,
                             int B, mgard_cuda_handle & handle, 
                             int queue_idx, bool profile);
}