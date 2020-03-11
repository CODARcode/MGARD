
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

#ifndef MGRAD_CUDA_HANDLE
#define MGRAD_CUDA_HANDLE
struct mgard_cuda_handle {
  void * queues;
  int num_of_queues;
  mgard_cuda_handle (): queues(NULL), num_of_queues(0) {}
  mgard_cuda_handle (int num_of_queues);
  void * get(int i);
  void sync(int i);
  void sync_all();
  void destory_all();
};
#endif



void print_matrix(int nrow, int ncol, double * v, int ldv);
void print_matrix_cuda(int nrow, int ncol, double * dv, int lddv);

void print_matrix(int nrow, int ncol, int nfib, double * v, int ldv1, int ldv2);
void print_matrix_cuda(int nrow, int ncol, int nfib, double * dv, int lddv1, int lddv2, int sizex);

void print_matrix(int nrow, int ncol, int * v, int ldv);
void print_matrix_cuda(int nrow, int ncol, int * dv, int lddv);

bool compare_matrix(int nrow, int ncol, double * v1, int ldv1, double * v2, int ldv2);
bool compare_matrix_cuda(int nrow, int ncol, 
                        double * dv1, int lddv1, 
                        double * dv2, int lddv2);

bool compare_matrix(int nrow, int ncol, int nfib,  
                    double * v1, int ldv11, int ldv12, 
                    double * v2, int ldv21, int ldv22);
bool compare_matrix_cuda(int nrow, int ncol, int nfib, 
                      double * dv1, int lddv11, int lddv12, int sizex1,
                      double * dv2, int lddv21, int lddv22, int sizex2);

void cudaMallocHelper(void **  devPtr, size_t  size);
void cudaMallocPitchHelper(void ** devPtr, size_t * pitch, size_t width, size_t height);
void cudaMalloc3DHelper(void ** devPtr, size_t * pitch, size_t width, size_t height, size_t depth);

void cudaMemcpyHelper(void * dst, const void * src, size_t count, enum copy_type);
void cudaMemcpy2DHelper(void * dst, size_t dpitch, 
                        const void * src, size_t spitch, 
                        size_t width, size_t height,
                        enum copy_type kind);
void cudaMemcpy3DHelper(void * dst, size_t dpitch, size_t dwidth, size_t dheight,
                        void * src, size_t spitch, size_t swidth, size_t sheight,
                        size_t width, size_t height, size_t depth,
                        enum copy_type kind);

void cudaFreeHelper(void * devPtr);
void cudaMemsetHelper(void * devPtr, int value, size_t count);
void cudaMemset2DHelper(void * devPtr,  size_t  pitch, int value, size_t width, size_t height);
void cudaMemset3DHelper(void * devPtr,  size_t  pitch, size_t dwidth, size_t dheight,
                        int value, size_t width, size_t height, size_t depth);