#include <cstdio>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void 
gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ int 
get_idx(const int ld, const int i, const int j);

__device__ int 
get_idx(const int ld1, const int ld2, const int i, const int j, const int k);