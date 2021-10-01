#ifndef REORDERTOOLSGPU_HPP
#define REORDERTOOLSGPU_HPP

#include "TensorMeshHierarchy.hpp"
#include "shuffle.hpp"

namespace mgard {

template <std::size_t D, typename T>
void ReorderCPU(TensorMeshHierarchy<D, T> &hierarchy, T * input, T * output) {
	shuffle(hierarchy, input, output);
}

template <std::size_t D, typename T>
void ReverseReorderCPU(TensorMeshHierarchy<D, T> &hierarchy, T * input, T * output) {
	unshuffle(hierarchy, input, output);
}
}


namespace mgard {
    #define KERNELS(D, T)                                                          \
      template void ReorderCPU<D, T>(TensorMeshHierarchy<D, T> &hierarchy, T * input, T * output); \
      template void ReverseReorderCPU<D, T>(TensorMeshHierarchy<D, T> &hierarchy, T * input, T * output);

  KERNELS(1, double)
  KERNELS(1, float)
  KERNELS(2, double)
  KERNELS(2, float)
  KERNELS(3, double)
  KERNELS(3, float)
  KERNELS(4, double)
  KERNELS(4, float)
  KERNELS(5, double)
  KERNELS(5, float)

  #undef KERNELS
}

#endif