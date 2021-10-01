#ifndef REORDERTOOLSGPU_H
#define REORDERTOOLSGPU_H

#include "../Common.h"

namespace mgard_cuda {

template <DIM D, typename T>
void ReorderGPU(Handle<D, T> &handle, SubArray<D, T> dinput, 
                             SubArray<D, T> &doutput, int l_target, int queue_idx);
template <DIM D, typename T>
void ReverseReorderGPU(Handle<D, T> &handle, SubArray<D, T> dinput, 
                             SubArray<D, T> &doutput, int l_target, int queue_idx);

}

#endif