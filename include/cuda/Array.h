/* 
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGARD_CUDA_ARRAY
#define MGARD_CUDA_ARRAY

namespace mgard_cuda {

template <typename T, int D>
class Array {
public:
	Array(std::vector<size_t> shape);
	~Array();
	void loadData(T * data, int ld);
	T * getDataHost();
	T * getDataDevice(int& ld);
private:
	int D_padded;
	T * dv;
	T * hv;
	bool device_allocated;
	bool host_allocated;
	std::vector<int> ldvs_h;
	int * ldvs_d;
	std::vector<size_t> shape;
	size_t linearized_depth;
};

}
#endif