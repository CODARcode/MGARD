
#ifndef COPY_TYPE
#define COPY_TYPE
enum copy_type { H2D, D2H, D2D};
#endif

#ifndef MGRAD_CUDA_RET
#define MGRAD_CUDA_RET
struct mgard_cuda_ret {
	int info;
	double time;
	mgard_cuda_ret (): info(0), time (0.0) {}
	mgard_cuda_ret (int info, double time) {
		this->info = info;
		this->time = time;
	}
};
#endif

#ifndef MGRAD_RET
#define MGRAD_RET
struct mgard_ret {
	int info;
	double time;
	mgard_ret (): info(0), time (0.0) {}
	mgard_ret (int info, double time) {
		this->info = info;
		this->time = time;
	}
};
#endif

void print_matrix(int nrow, int ncol, double * v, int ldv);
void print_matrix_cuda(int nrow, int ncol, double * dv, int lddv);
void print_matrix(int nrow, int ncol, int * v, int ldv);
void print_matrix_cuda(int nrow, int ncol, int * dv, int lddv);
bool compare_matrix(int nrow, int ncol, double * v1, int ldv1, double * v2, int ldv2);
bool compare_matrix_cuda(int nrow, int ncol, 
                        double * dv1, int lddv1, 
                        double * dv2, int lddv2);
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
