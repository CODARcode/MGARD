#include <iomanip> 
#include <iostream>
#include "mgard_cuda_helper_internal.h"



__device__ int 
get_idx(const int ld, const int i, const int j) {
    return ld * i + j;
}