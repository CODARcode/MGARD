#include <iomanip> 
#include <iostream>
#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"


mgard_cuda_handle::mgard_cuda_handle (int num_of_queues){
  this->num_of_queues = num_of_queues;
  cudaStream_t * ptr = (cudaStream_t *)this->queues;
  ptr = new cudaStream_t[num_of_queues];
  for (int i = 0; i < this->num_of_queues; i++) {
    gpuErrchk(cudaStreamCreate(&(ptr[i])));
  }
}

void * mgard_cuda_handle::get(int i) {
  cudaStream_t * ptr = (cudaStream_t *)this->queues;
  return (void *)ptr[i];
}

void mgard_cuda_handle::sync(int i) {
  cudaStream_t * ptr = (cudaStream_t *)this->queues;
  gpuErrchk(cudaStreamSynchronize(ptr[i]));
}

void mgard_cuda_handle::sync_all() {
  cudaStream_t * ptr = (cudaStream_t *)this->queues;
  for (int i = 0; i < this->num_of_queues; i++) {
    gpuErrchk(cudaStreamSynchronize(ptr[i]));
  }
}

void mgard_cuda_handle::destory_all() {
  cudaStream_t * ptr = (cudaStream_t *)this->queues;
  for (int i = 0; i < this->num_of_queues; i++) {
    gpuErrchk(cudaStreamDestroy(ptr[i]));
  }
}




// print 2D CPU
void print_matrix(int nrow, int ncol, double * v, int ldv) {
  //std::cout << std::setw(10);
  //std::cout << std::setprecision(2) << std::fixed;
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
        std::cout <<std::setw(9) << std::setprecision(6) << std::fixed <<  v[ldv*i + j]<<", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// print 2D GPU
void print_matrix_cuda(int nrow, int ncol, double * dv, int lddv) {
  //std::cout << std::setw(10);
  //std::cout << std::setprecision(2) << std::fixed;
  double * v = new double[nrow * ncol];
  cudaMemcpy2DHelper(v, ncol  * sizeof(double), 
                    dv, lddv * sizeof(double),  
                    ncol * sizeof(double), nrow, 
                    D2H);
  print_matrix(nrow, ncol, v, ncol); 
}

// print 3D GPU
void print_matrix_cuda(int nrow, int ncol, int nfib, double * dv, int lddv1, int lddv2, int sizex) {
  //std::cout << std::setw(10);
  //std::cout << std::setprecision(2) << std::fixed;
  double * v = new double[nrow * ncol * nfib];
  cudaMemcpy3DHelper(v, nfib  * sizeof(double), nfib * sizeof(double), ncol,
                     dv, lddv1 * sizeof(double), sizex * sizeof(double), lddv2,
                     nfib * sizeof(double), ncol, nrow, 
                     D2H);
  print_matrix(nrow, ncol, nfib, v, nfib, ncol); 
}

// print 3D CPU
void print_matrix(int nrow, int ncol, int nfib, double * v, int ldv1, int ldv2) {
  //std::cout << std::setw(10);
  //std::cout << std::setprecision(2) << std::fixed;
  for (int i = 0; i < nrow; i++) {
    std::cout << "[ nrow = " << i << " ]\n";
    print_matrix(ncol, nfib, v + i * ldv1 * ldv2, ldv1);
    std::cout << std::endl;
  }
}


// print 2D GPU-int
void print_matrix_cuda(int nrow, int ncol, int * dv, int lddv) {
  //std::cout << std::setw(10);
  //std::cout << std::setprecision(2) << std::fixed;
  int * v = new int[nrow * ncol];
  cudaMemcpy2DHelper(v, ncol  * sizeof(int), 
                    dv, lddv * sizeof(int),  
                    ncol * sizeof(int), nrow, 
                    D2H);
  print_matrix(nrow, ncol, v, ncol);
    
}

// print 2D CPU-int
void print_matrix(int nrow, int ncol, int * v, int ldv) {
  //std::cout << std::setw(10);
  //std::cout << std::setprecision(2) << std::fixed;
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
        std::cout <<std::setw(5) << std::setprecision(0) << std::fixed <<  v[ldv*i + j]<<", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// compare 2D CPU
bool compare_matrix(int nrow, int ncol, 
                  double * v1, int ldv1, 
                  double * v2, int ldv2) {
  //std::cout << std::setw(10);
  //std::cout << std::setprecision(2) << std::fixed;
  bool correct = true;
  bool nan = false;
  double E = 1e-6;
  for (int i = 0; i < nrow; i++) {
      for (int j = 0; j < ncol; j++) {
          double a = v1[ldv1*i + j];
          double b = v2[ldv2*i + j];
          if (abs(a - b) > E){
              correct = false;
              // std::cout << "Diff at (" << i << ", " << j << ") ";
              // std::cout << a << " - " << b << " = " << abs(a-b) << std::endl; 
          }
          if (isnan(a) || isnan(b)) {
            correct = true;
            // std::cout << "NAN at (" << i << ", " << j << ") ";
            // std::cout << a << " - " << b << " = " << abs(a-b) << std::endl;
          }
      }
  }
  if (correct) printf("Compare: correct.\n");
  else printf("Compare: wrong.\n");
  if (nan) printf("Nan: include.\n");
  //else printf("Nan: not include.\n");
  return correct;
}

// compare 2D GPU
bool compare_matrix_cuda(int nrow, int ncol, 
                      double * dv1, int lddv1, 
                      double * dv2, int lddv2) {
  double * v1 = new double[nrow * ncol];
  int ldv1 = ncol;
  cudaMemcpy2DHelper(v1, ldv1  * sizeof(double), 
                    dv1, lddv1 * sizeof(double),  
                    ncol * sizeof(double), nrow, 
                    D2H);
  double * v2 = new double[nrow * ncol];
  int ldv2 = ncol;
  cudaMemcpy2DHelper(v2, ldv2  * sizeof(double), 
                    dv2, lddv2 * sizeof(double),  
                    ncol * sizeof(double), nrow, 
                    D2H);
  bool ret = compare_matrix(nrow, ncol, 
                        v1,   ldv1, 
                        v2,   ldv2);
  delete v1;
  delete v2;
  return ret;
}


// compare 3D CPU
bool compare_matrix(int nrow, int ncol, int nfib, 
                    double * v1, int ldv11, int ldv12, 
                    double * v2, int ldv21, int ldv22) {
  //std::cout << std::setw(10);
  //std::cout << std::setprecision(2) << std::fixed;
  bool correct = true;
  bool nan = false;
  double E = 1e-6;
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      for (int k = 0; k < nfib; k++) {
        double a = v1[ldv11*ldv12*i + ldv11*j + k];
        double b = v2[ldv21*ldv22*i + ldv21*j + k];
        if (abs(a - b) > E){
          correct = false;
          // std::cout << "Diff at (" << i << ", " << j << ", " << k <<") ";
          // std::cout << a << " - " << b << " = " << abs(a-b) << std::endl; 
        }
        if (isnan(a) || isnan(b)) {
          correct = true;
          // std::cout << "NAN at (" << i << ", " << j << ") ";
          // std::cout << a << " - " << b << " = " << abs(a-b) << std::endl;
        }
      }
    }
  }
  if (correct) printf("Compare: correct.\n");
  else printf("Compare: wrong.\n");
  if (nan) printf("Nan: include.\n");
  //else printf("Nan: not include.\n");
  return correct;
}

// compare 3D GPU
bool compare_matrix_cuda(int nrow, int ncol, int nfib, 
                      double * dv1, int lddv11, int lddv12, int sizex1,
                      double * dv2, int lddv21, int lddv22, int sizex2) {
  double * v1 = new double[nrow * ncol * nfib];
  int ldv11 = nfib;
  int ldv12 = ncol;
  cudaMemcpy3DHelper(v1, ldv11  * sizeof(double), nfib * sizeof(double), ldv12, 
                    dv1, lddv11 * sizeof(double), sizex1 * sizeof(double), lddv12,
                    nfib * sizeof(double), ncol, nrow,
                    D2H);

  double * v2 = new double[nrow * ncol * nfib];
  int ldv21 = nfib;
  int ldv22 = ncol;
  cudaMemcpy3DHelper(v2, ldv21  * sizeof(double), nfib * sizeof(double), ldv22, 
                    dv2, lddv21 * sizeof(double), sizex2 * sizeof(double), lddv22, 
                    nfib * sizeof(double), ncol, nrow,
                    D2H);
  bool ret = compare_matrix(nrow, ncol, nfib, 
                            v1,   ldv11, ldv12,
                            v2,   ldv21, ldv22);
  delete [] v1;
  delete [] v2;
  return ret;
}


// Allocate 1D
void cudaMallocHelper(void **  devPtr, size_t  size) {
  gpuErrchk(cudaMalloc(devPtr, size));
}

// Allocate 2D
void cudaMallocPitchHelper(void ** devPtr, size_t * pitch, size_t width, size_t height) {
  gpuErrchk(cudaMallocPitch(devPtr, pitch, width, height));
}

//Allocate 3D
void cudaMalloc3DHelper(void ** devPtr, size_t * pitch, size_t width, size_t height, size_t depth) {
  cudaPitchedPtr devPitchedPtr;
  cudaExtent extent = make_cudaExtent(width, height, depth);
  gpuErrchk(cudaMalloc3D(&devPitchedPtr, extent));
  *devPtr = devPitchedPtr.ptr;
  *pitch = devPitchedPtr.pitch;
}

// Copy 1D
void cudaMemcpyHelper(void * dst, const void * src, size_t count, enum copy_type kind){
  enum cudaMemcpyKind cuda_copy_type;
  switch (kind)
  {
    case H2D :  cuda_copy_type = cudaMemcpyHostToDevice; break;
    case D2H :  cuda_copy_type = cudaMemcpyDeviceToHost; break;
    case D2D :  cuda_copy_type = cudaMemcpyDeviceToDevice; break;
  }
  gpuErrchk(cudaMemcpy(dst, src, count, cuda_copy_type));
}

// Copy 2D
void cudaMemcpy2DHelper(void * dst, size_t dpitch, 
                        const void * src, size_t spitch, 
                        size_t width, size_t height,
                        enum copy_type kind) {
  enum cudaMemcpyKind cuda_copy_type;
  switch (kind)
  {
    case H2D :  cuda_copy_type = cudaMemcpyHostToDevice; break;
    case D2H :  cuda_copy_type = cudaMemcpyDeviceToHost; break;
    case D2D :  cuda_copy_type = cudaMemcpyDeviceToDevice; break;
  }
  gpuErrchk(cudaMemcpy2D(dst, dpitch, 
                         src, spitch,
                         width, height, 
                         cuda_copy_type));
}


// Copy 3D
void cudaMemcpy3DHelper(void * dst, size_t dpitch, size_t dwidth, size_t dheight,
                        void * src, size_t spitch, size_t swidth, size_t sheight,
                        size_t width, size_t height, size_t depth,
                        enum copy_type kind) {
  cudaExtent extent = make_cudaExtent(width, height, depth);
  cudaMemcpy3DParms p = {0};
  p.dstPtr.ptr = dst;
  p.dstPtr.pitch = dpitch;
  p.dstPtr.xsize = dwidth;
  p.dstPtr.ysize = dheight;

  p.srcPtr.ptr = src;
  p.srcPtr.pitch = spitch;
  p.srcPtr.xsize = swidth;
  p.srcPtr.ysize = sheight;
  
  p.extent = extent;
  enum cudaMemcpyKind cuda_copy_type;
  switch (kind)
  {
    case H2D :  cuda_copy_type = cudaMemcpyHostToDevice; break;
    case D2H :  cuda_copy_type = cudaMemcpyDeviceToHost; break;
    case D2D :  cuda_copy_type = cudaMemcpyDeviceToDevice; break;
  }
  p.kind = cuda_copy_type;
  gpuErrchk(cudaMemcpy3D(&p));
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

void cudaMemset3DHelper(void * devPtr,  size_t  pitch, size_t dwidth, size_t dheight,
                        int value, size_t width, size_t height, size_t depth) {
  cudaExtent extent = make_cudaExtent(width, height, depth);
  cudaPitchedPtr devPitchedPtr;
  devPitchedPtr.ptr = devPtr;
  devPitchedPtr.pitch = pitch;
  devPitchedPtr.xsize = dwidth;
  devPitchedPtr.ysize = dheight;
  gpuErrchk(cudaMemset3D(devPitchedPtr, value, extent));
}