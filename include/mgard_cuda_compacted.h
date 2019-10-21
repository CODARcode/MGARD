void 
original_to_compacted_cuda(int nrow,       int ncol, 
                          int row_stride, int col_stride,
                          double * v,     int ldv, 
                          double * work,  int ldwork);
void 
compacted_to_original_cuda(int nrow,       int ncol, 
                          int row_stride, int col_stride, 
                          double * v,     int ldv, 
                          double * work,  int ldwork);

void fused_row_cuda(int nrow,       int ncol, 
                   int row_stride, int col_stride,
                   int k1_row_stride, int k1_col_stride,
                   int k2_row_stride, int k2_col_stride,
                   double * v,    int ldv);