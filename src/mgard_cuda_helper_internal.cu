#include <iomanip> 
#include <iostream>
#include "mgard_cuda_helper_internal.h"



__device__ int 
get_idx(const int ld, const int i, const int j) {
    return ld * i + j;
}

//ld2 = nrow
//ld1 = pitch
__device__ int 
get_idx(const int ld1, const int ld2, const int i, const int j, const int k) {
    return ld2 * ld1 * i + ld1 * j + k;
}