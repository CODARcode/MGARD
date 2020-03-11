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
get_idx(const int ld1, const int ld2, const int z, const int y, const int x) {
    return ld2 * ld1 * z + ld1 * y + x;
}