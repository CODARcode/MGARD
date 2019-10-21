
#ifndef COPY_TYPE
#define COPY_TYPE
enum copy_type { H2D, D2H };
#endif

void print_matrix(int nrow, int ncol, double * v, int ldv);
void compare_matrix(int nrow, int ncol, double * v1, int ldv1, double * v2, int ldv2);
void cudaMallocHelper(void **  devPtr, size_t  size);
void cudaMemcpyHelper(void * dst, const void * src, size_t count, enum copy_type);
void cudaMallocPitchHelper(void ** devPtr, size_t * pitch, size_t width, size_t height);
void cudaMemcpy2DHelper(void * dst, size_t dpitch, 
                        const void * src, size_t spitch, 
                        size_t width, size_t height,
                        enum copy_type kind);
void cudaFreeHelper(void * devPtr);
void cudaMemsetHelper(void * devPtr, int value, size_t count);
void cudaMemset2DHelper(void * devPtr,  size_t  pitch, int value, size_t width, size_t height);
