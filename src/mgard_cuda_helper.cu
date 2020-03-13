#include <iomanip> 
#include <iostream>
#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"


mgard_cuda_handle::mgard_cuda_handle (int num_of_queues){
  this->num_of_queues = num_of_queues;
  cudaStream_t * ptr = new cudaStream_t[num_of_queues];

  for (int i = 0; i < this->num_of_queues; i++) {
    gpuErrchk(cudaStreamCreate(ptr+i));
    // std::cout << "created a stream\n";
  }

  this->queues = (void *)ptr;

}

void * mgard_cuda_handle::get(int i) {
  cudaStream_t * ptr = (cudaStream_t *)(this->queues);
  return (void *)(ptr+i);
}

void mgard_cuda_handle::sync(int i) {
  cudaStream_t * ptr = (cudaStream_t *)(this->queues);
  gpuErrchk(cudaStreamSynchronize(ptr[i]));
}

void mgard_cuda_handle::sync_all() {
  cudaStream_t * ptr = (cudaStream_t *)(this->queues);
  for (int i = 0; i < this->num_of_queues; i++) {
    gpuErrchk(cudaStreamSynchronize(ptr[i]));
  }
}

void mgard_cuda_handle::destory_all() {
  cudaStream_t * ptr = (cudaStream_t *)(this->queues);
  for (int i = 0; i < this->num_of_queues; i++) {
    gpuErrchk(cudaStreamDestroy(ptr[i]));
  }
}




// print 2D CPU
template <typename T>
void print_matrix(int nrow, int ncol, T * v, int ldv) {
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

template void print_matrix<double>(int nrow, int ncol, double * v, int ldv);
template void print_matrix<float>(int nrow, int ncol, float * v, int ldv);
template void print_matrix<int>(int nrow, int ncol, int * v, int ldv);


// print 2D GPU
template <typename T>
void print_matrix_cuda(int nrow, int ncol, T * dv, int lddv) {
  //std::cout << std::setw(10);
  //std::cout << std::setprecision(2) << std::fixed;
  mgard_cuda_handle tmp_handle(1);
  int queue_idx = 0;
  bool profile = false;
  T * v = new T[nrow * ncol];
  cudaMemcpy2DAsyncHelper(v, ncol  * sizeof(T), 
                    dv, lddv * sizeof(T),  
                    ncol * sizeof(T), nrow, 
                    D2H, tmp_handle, queue_idx, profile);
  tmp_handle.sync(queue_idx);
  print_matrix(nrow, ncol, v, ncol); 
  delete [] v;
  tmp_handle.destory_all();
}
template void print_matrix_cuda<double>(int nrow, int ncol, double * dv, int lddv);
template void print_matrix_cuda<float>(int nrow, int ncol, float * dv, int lddv);
template void print_matrix_cuda<int>(int nrow, int ncol, int * dv, int lddv);

// print 3D GPU
template <typename T>
void print_matrix_cuda(int nrow, int ncol, int nfib, T * dv, int lddv1, int lddv2, int sizex) {
  //std::cout << std::setw(10);
  //std::cout << std::setprecision(2) << std::fixed;
  mgard_cuda_handle tmp_handle(1);
  int queue_idx = 0;
  bool profile = false;

  T * v = new T[nrow * ncol * nfib];
  cudaMemcpy3DAsyncHelper(v, nfib  * sizeof(T), nfib * sizeof(T), ncol,
                     dv, lddv1 * sizeof(T), sizex * sizeof(T), lddv2,
                     nfib * sizeof(T), ncol, nrow, 
                     D2H, tmp_handle, queue_idx, profile);
  tmp_handle.sync(queue_idx);
  print_matrix(nrow, ncol, nfib, v, nfib, ncol); 
  delete [] v;
  tmp_handle.destory_all();
}

template void print_matrix_cuda<double>(int nrow, int ncol, int nfib, double * dv, int lddv1, int lddv2, int sizex);
template void print_matrix_cuda<float>(int nrow, int ncol, int nfib, float * dv, int lddv1, int lddv2, int sizex);

// print 3D CPU
template <typename T>
void print_matrix(int nrow, int ncol, int nfib, T * v, int ldv1, int ldv2) {
  //std::cout << std::setw(10);
  //std::cout << std::setprecision(2) << std::fixed;
  for (int i = 0; i < nrow; i++) {
    std::cout << "[ nrow = " << i << " ]\n";
    print_matrix(ncol, nfib, v + i * ldv1 * ldv2, ldv1);
    std::cout << std::endl;
  }
}

template void print_matrix<double>(int nrow, int ncol, int nfib, double * v, int ldv1, int ldv2);
template void print_matrix<float>(int nrow, int ncol, int nfib, float * v, int ldv1, int ldv2);


// compare 2D CPU
template <typename T>
bool compare_matrix(int nrow, int ncol, 
                  T * v1, int ldv1, 
                  T * v2, int ldv2) {
  //std::cout << std::setw(10);
  //std::cout << std::setprecision(2) << std::fixed;
  bool correct = true;
  bool nan = false;
  double E = 1e-6;
  for (int i = 0; i < nrow; i++) {
      for (int j = 0; j < ncol; j++) {
          T a = v1[ldv1*i + j];
          T b = v2[ldv2*i + j];
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

template bool compare_matrix<double>(int nrow, int ncol, 
                                      double * v1, int ldv1, 
                                      double * v2, int ldv2);
template bool compare_matrix<float>(int nrow, int ncol, 
                                      float * v1, int ldv1, 
                                      float * v2, int ldv2);

// compare 2D GPU
template <typename T>
bool compare_matrix_cuda(int nrow, int ncol, 
                      T * dv1, int lddv1, 
                      T * dv2, int lddv2) {
  mgard_cuda_handle tmp_handle(1);
  int queue_idx = 0;
  bool profile = false;

  T * v1 = new T[nrow * ncol];
  int ldv1 = ncol;
  cudaMemcpy2DAsyncHelper(v1, ldv1  * sizeof(T), 
                    dv1, lddv1 * sizeof(T),  
                    ncol * sizeof(T), nrow, 
                    D2H, tmp_handle, queue_idx, profile);
  T * v2 = new T[nrow * ncol];
  int ldv2 = ncol;
  cudaMemcpy2DAsyncHelper(v2, ldv2  * sizeof(T), 
                    dv2, lddv2 * sizeof(T),  
                    ncol * sizeof(T), nrow, 
                    D2H, tmp_handle, queue_idx, profile);
  tmp_handle.sync(queue_idx);
  bool ret = compare_matrix(nrow, ncol, 
                        v1,   ldv1, 
                        v2,   ldv2);
  delete [] v1;
  delete [] v2;
  tmp_handle.destory_all();
  return ret;
}

template bool compare_matrix_cuda<double>(int nrow, int ncol, 
                                          double * dv1, int lddv1, 
                                          double * dv2, int lddv2);
template bool compare_matrix_cuda<float>(int nrow, int ncol, 
                                          float * dv1, int lddv1, 
                                          float * dv2, int lddv2);


// compare 3D CPU
template <typename T>
bool compare_matrix(int nrow, int ncol, int nfib, 
                    T * v1, int ldv11, int ldv12, 
                    T * v2, int ldv21, int ldv22) {
  //std::cout << std::setw(10);
  //std::cout << std::setprecision(2) << std::fixed;
  bool correct = true;
  bool nan = false;
  double E = 1e-6;
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      for (int k = 0; k < nfib; k++) {
        T a = v1[ldv11*ldv12*i + ldv11*j + k];
        T b = v2[ldv21*ldv22*i + ldv21*j + k];
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

template bool compare_matrix<double>(int nrow, int ncol, int nfib, 
                                    double * v1, int ldv11, int ldv12, 
                                    double * v2, int ldv21, int ldv22);
template bool compare_matrix<float>(int nrow, int ncol, int nfib, 
                                    float * v1, int ldv11, int ldv12, 
                                    float * v2, int ldv21, int ldv22);

// compare 3D GPU
template <typename T>
bool compare_matrix_cuda(int nrow, int ncol, int nfib, 
                      T * dv1, int lddv11, int lddv12, int sizex1,
                      T * dv2, int lddv21, int lddv22, int sizex2) {
  mgard_cuda_handle tmp_handle(1);
  int queue_idx = 0;
  bool profile = false;

  T * v1 = new T[nrow * ncol * nfib];
  int ldv11 = nfib;
  int ldv12 = ncol;
  cudaMemcpy3DAsyncHelper(v1, ldv11  * sizeof(T), nfib * sizeof(T), ldv12, 
                    dv1, lddv11 * sizeof(T), sizex1 * sizeof(T), lddv12,
                    nfib * sizeof(T), ncol, nrow,
                    D2H, tmp_handle, queue_idx, profile);

  T * v2 = new T[nrow * ncol * nfib];
  int ldv21 = nfib;
  int ldv22 = ncol;
  cudaMemcpy3DAsyncHelper(v2, ldv21  * sizeof(T), nfib * sizeof(T), ldv22, 
                    dv2, lddv21 * sizeof(T), sizex2 * sizeof(T), lddv22, 
                    nfib * sizeof(T), ncol, nrow,
                    D2H, tmp_handle, queue_idx, profile);
  tmp_handle.sync(queue_idx);
  bool ret = compare_matrix(nrow, ncol, nfib, 
                            v1,   ldv11, ldv12,
                            v2,   ldv21, ldv22);
  delete [] v1;
  delete [] v2;
  tmp_handle.destory_all();
  return ret;
}

template bool compare_matrix_cuda<double>(int nrow, int ncol, int nfib, 
                      double * dv1, int lddv11, int lddv12, int sizex1,
                      double * dv2, int lddv21, int lddv22, int sizex2);
template bool compare_matrix_cuda<float>(int nrow, int ncol, int nfib, 
                      float * dv1, int lddv11, int lddv12, int sizex1,
                      float * dv2, int lddv21, int lddv22, int sizex2);

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
mgard_cuda_ret
cudaMemcpyAsyncHelper(void * dst, const void * src, size_t count, enum copy_type kind,
                      mgard_cuda_handle & handle, int queue_idx, bool profile){

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);
  enum cudaMemcpyKind cuda_copy_type;
  switch (kind)
  {
    case H2D :  cuda_copy_type = cudaMemcpyHostToDevice; break;
    case D2H :  cuda_copy_type = cudaMemcpyDeviceToHost; break;
    case D2D :  cuda_copy_type = cudaMemcpyDeviceToDevice; break;
  }
  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }
  gpuErrchk(cudaMemcpyAsync(dst, src, count, cuda_copy_type, stream));
  if (profile) {
    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
  }
  return mgard_cuda_ret(0, milliseconds/1000.00);

}

// Copy 2D
mgard_cuda_ret 
cudaMemcpy2DAsyncHelper(void * dst, size_t dpitch, 
                        const void * src, size_t spitch, 
                        size_t width, size_t height,
                        enum copy_type kind,
                        mgard_cuda_handle & handle, int queue_idx, bool profile) {

  cudaEvent_t start, stop;
  float milliseconds = 0;
  //cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);
  mgard_cuda_handle *handle2 = new mgard_cuda_handle(1);
  cudaStream_t stream = *(cudaStream_t *)(*handle2).get(queue_idx);
  // cudaStream_t stream;
  // cudaStreamCreate(&stream);
  enum cudaMemcpyKind cuda_copy_type;
  switch (kind)
  {
    case H2D :  cuda_copy_type = cudaMemcpyHostToDevice; break;
    case D2H :  cuda_copy_type = cudaMemcpyDeviceToHost; break;
    case D2D :  cuda_copy_type = cudaMemcpyDeviceToDevice; break;
  }
  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }
  gpuErrchk(cudaMemcpy2DAsync(dst, dpitch, 
                         src, spitch,
                         width, height, 
                         cuda_copy_type,
                         stream));
  if (profile) {
    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
  }
  return mgard_cuda_ret(0, milliseconds/1000.00);
}


// Copy 3D
mgard_cuda_ret 
cudaMemcpy3DAsyncHelper(void * dst, size_t dpitch, size_t dwidth, size_t dheight,
                        void * src, size_t spitch, size_t swidth, size_t sheight,
                        size_t width, size_t height, size_t depth,
                        enum copy_type kind,
                        mgard_cuda_handle & handle, int queue_idx, bool profile) {
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

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
  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }
  gpuErrchk(cudaMemcpy3DAsync(&p, stream));
  if (profile) {
    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
  }
  return mgard_cuda_ret(0, milliseconds/1000.00);
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