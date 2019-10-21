#include "mgard_cuda.h"

#ifndef MGARD_API_CUDA_H
#define MGARD_API_CUDA_H

unsigned char *mgard_compress_cuda(int itype_flag,  double  *v, int &out_size, int nrow, int ncol, int nfib, double tol_in);
double *mgard_decompress_cuda(int itype_flag,  double& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib);

#endif