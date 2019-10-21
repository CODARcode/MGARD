#include <iomanip> 
#include <iostream>
#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"


void print_matrix(int nrow, int ncol, double * v, int ldv) {
    //std::cout << std::setw(10);
    //std::cout << std::setprecision(2) << std::fixed;
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            std::cout <<std::setw(8) << std::setprecision(4) << std::fixed <<  v[ldv*i + j]<<", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void compare_matrix(int nrow, int ncol, 
                    double * v1, int ldv1, 
                    double * v2, int ldv2) {
    //std::cout << std::setw(10);
    //std::cout << std::setprecision(2) << std::fixed;
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            if (abs(v1[ldv1*i + j] - v2[ldv2*i + j]) > 0.001){
                std::cout << "Diff at (" << i << ", " << j << ")" << std::endl; 
            }
        }
    }   
}






void cudaMallocHelper(void **  devPtr, size_t  size) {
    gpuErrchk(cudaMalloc(devPtr, size));
}

void cudaMemcpyHelper(void * dst, const void * src, size_t count, enum copy_type kind){
    enum cudaMemcpyKind cuda_copy_type;
    switch (kind)
    {
        case H2D :  cuda_copy_type = cudaMemcpyHostToDevice; break;
        case D2H :  cuda_copy_type = cudaMemcpyDeviceToHost; break;
    }
    gpuErrchk(cudaMemcpy(dst, src, count, cuda_copy_type));
}

void cudaMallocPitchHelper(void ** devPtr, size_t * pitch, size_t width, size_t height) {
    gpuErrchk(cudaMallocPitch(devPtr, pitch, width, height));
}

void cudaMemcpy2DHelper(void * dst, size_t dpitch, 
                        const void * src, size_t spitch, 
                        size_t width, size_t height,
                        enum copy_type kind) {
    enum cudaMemcpyKind cuda_copy_type;
    switch (kind)
    {
        case H2D :  cuda_copy_type = cudaMemcpyHostToDevice; break;
        case D2H :  cuda_copy_type = cudaMemcpyDeviceToHost; break;
    }
    gpuErrchk(cudaMemcpy2D(dst, dpitch, 
                 src, spitch,
                 width, height, 
                 cuda_copy_type));
}

void cudaFreeHelper(void * devPtr) {
    gpuErrchk(cudaFree(devPtr));
}

void cudaMemsetHelper(void * devPtr, int value, size_t count) {
    gpuErrchk(cudaMemset(devPtr, value, count));
}

void cudaMemset2DHelper(void * devPtr,  size_t  pitch, int value, size_t width, size_t height) {
    gpuErrchk(cudaMemset2D(devPtr, pitch, value, width, height));
}