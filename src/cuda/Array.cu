/* 
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/mgard_cuda_common.h"
#include "cuda/Array.h"

namespace mgard_cuda {

	template <typename T, int D>
	Array<T, D>::Array(std::vector<size_t> shape) {
    this->host_allocated = false;
    this->device_allocated = false;

    if (shape.size() != D) { 
      std::cerr << log_err << "Number of dimensions mismatch. mgard_cuda::Array not initialized!\n"; 
      return; 
    }
    D_padded = D;
    if (D < 3) { D_padded = 3; }
    if (D % 2 == 0) { D_padded = D + 1; }
    //padding dimensions
    for (int d = shape.size(); d < D_padded; d ++) {
      shape.push_back(1);
    }


    this->shape = shape;
    this->linearized_depth = 1;
    for (int i = 2; i < D_padded; i++) { linearized_depth *= shape[i]; }
    size_t dv_pitch;
    cudaMalloc3DHelper((void **)&dv, &dv_pitch, shape[0] * sizeof(T),
                       shape[1], linearized_depth);
    ldvs_h.push_back(dv_pitch / sizeof(T));
    for (int i = 1; i < D_padded; i++) { ldvs_h.push_back(shape[i]); }
		mgard_cuda_handle<T, D> handle;
		cudaMallocHelper((void **)&ldvs_d, ldvs_h.size() * sizeof(int));
  	cudaMemcpyAsyncHelper(handle, ldvs_d, ldvs_h.data(), 
    											ldvs_h.size() * sizeof(int), AUTO, 0);

    this->device_allocated = true;
	}

	template <typename T, int D>
	Array<T, D>::~Array() {
    if (device_allocated) {
  		cudaFreeHelper(ldvs_d);
  		cudaFreeHelper(dv);
    }
    if (host_allocated) {
      delete[] hv;
    }
	}

  template <typename T, int D>
  void Array<T, D>::loadData(T * data, int ld = -1) {
    if (ld == -1) { ld = shape[0]; }
    mgard_cuda::mgard_cuda_handle<T, D> handle;
    cudaMemcpy3DAsyncHelper(
        handle, dv, ldvs_h[0] * sizeof(T), shape[0] * sizeof(T), shape[1], 
        data, ld * sizeof(T), shape[0] * sizeof(T), shape[1],
        shape[0] * sizeof(T), shape[1], linearized_depth, AUTO, 0);
    handle.sync(0);
  }

	template <typename T, int D>
	T * Array<T, D>::getDataHost() {
    mgard_cuda::mgard_cuda_handle<T, D> handle;
    if (!host_allocated) {
      cudaMallocHostHelper((void **)&hv,
                          sizeof(T) * shape[0] * shape[1] * linearized_depth);
    }
    cudaMemcpy3DAsyncHelper(
        handle, hv, shape[0] * sizeof(T), shape[0] * sizeof(T), shape[1], 
        dv, ldvs_h[0] * sizeof(T), shape[0] * sizeof(T), shape[1], 
        shape[0] * sizeof(T), shape[1], linearized_depth, AUTO, 0);
    handle.sync(0);
    return hv;
	}

	template <typename T, int D>
	T * Array<T, D>::getDataDevice(int &ld) {
    ld = ldvs_h[0];
    return dv;
	}

template class Array<double, 1>;
template class Array<float, 1>;
template class Array<double, 2>;
template class Array<float, 2>;
template class Array<double, 3>;
template class Array<float, 3>;
template class Array<double, 4>;
template class Array<float, 4>;
template class Array<double, 5>;
template class Array<float, 5>;

}