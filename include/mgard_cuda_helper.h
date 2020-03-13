
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


template <typename T>
void print_matrix(int nrow, int ncol, T * v, int ldv);
template <typename T>
void print_matrix_cuda(int nrow, int ncol, T * dv, int lddv);

template <typename T>
void print_matrix(int nrow, int ncol, int nfib, T * v, int ldv1, int ldv2);
template <typename T>
void print_matrix_cuda(int nrow, int ncol, int nfib, T * dv, int lddv1, int lddv2, int sizex);

template <typename T>
bool compare_matrix(int nrow, int ncol, T * v1, int ldv1, T * v2, int ldv2);
template <typename T>
bool compare_matrix_cuda(int nrow, int ncol, 
                        T * dv1, int lddv1, 
                        T * dv2, int lddv2);
template <typename T>
bool compare_matrix(int nrow, int ncol, int nfib,  
                    T * v1, int ldv11, int ldv12, 
                    T * v2, int ldv21, int ldv22);
template <typename T>
bool compare_matrix_cuda(int nrow, int ncol, int nfib, 
                      T * dv1, int lddv11, int lddv12, int sizex1,
                      T * dv2, int lddv21, int lddv22, int sizex2);

void cudaMallocHelper(void **  devPtr, size_t  size);
void cudaMallocPitchHelper(void ** devPtr, size_t * pitch, size_t width, size_t height);
void cudaMalloc3DHelper(void ** devPtr, size_t * pitch, size_t width, size_t height, size_t depth);

mgard_cuda_ret cudaMemcpyAsyncHelper(void * dst, const void * src, size_t count, enum copy_type,
                          mgard_cuda_handle & handle, int queue_idx, bool profile);
mgard_cuda_ret cudaMemcpy2DAsyncHelper(void * dst, size_t dpitch, 
                        const void * src, size_t spitch, 
                        size_t width, size_t height,
                        enum copy_type kind,
                        mgard_cuda_handle & handle, int queue_idx, bool profile);
mgard_cuda_ret cudaMemcpy3DAsyncHelper(void * dst, size_t dpitch, size_t dwidth, size_t dheight,
                        void * src, size_t spitch, size_t swidth, size_t sheight,
                        size_t width, size_t height, size_t depth,
                        enum copy_type kind,
                        mgard_cuda_handle & handle, int queue_idx, bool profile);

void cudaFreeHelper(void * devPtr);
void cudaMemsetHelper(void * devPtr, int value, size_t count);
void cudaMemset2DHelper(void * devPtr,  size_t  pitch, int value, size_t width, size_t height);
void cudaMemset3DHelper(void * devPtr,  size_t  pitch, size_t dwidth, size_t dheight,
                        int value, size_t width, size_t height, size_t depth);