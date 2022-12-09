/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_UNCERTAINTY_COLLECTOR
#define MGARD_X_UNCERTAINTY_COLLECTOR

namespace mgard_x {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class UncertaintyDistributionFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT UncertaintyDistributionFunctor() {}
  MGARDX_CONT
  UncertaintyDistributionFunctor(
      SubArray<D, T, DeviceType> original_data,
      SubArray<D, T, DeviceType> reconstructed_data, double max_error,
      int num_bins, int reduced_levels,
      SubArray<D + 1, int, DeviceType> error_distribution)
      : original_data(original_data), reconstructed_data(reconstructed_data),
        max_error(max_error), num_bins(num_bins),
        reduced_levels(reduced_levels), error_distribution(error_distribution) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() *
                (FunctorBase<DeviceType>::GetBlockDimX() *
                 FunctorBase<DeviceType>::GetBlockDimY())) +
               (FunctorBase<DeviceType>::GetThreadIdY() *
                FunctorBase<DeviceType>::GetBlockDimX()) +
               FunctorBase<DeviceType>::GetThreadIdX();

    SIZE idx[D];
    SIZE firstD = div_roundup(original_data.shape(D - 1), F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    // idx[0] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();
    idx[D - 1] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();

    // printf("firstD %d idx[0] %d\n", firstD, idx[0]);

    bidx /= firstD;
    if (D >= 2)
      idx[D - 2] = FunctorBase<DeviceType>::GetBlockIdY() *
                       FunctorBase<DeviceType>::GetBlockDimY() +
                   FunctorBase<DeviceType>::GetThreadIdY();
    if (D >= 3)
      idx[D - 3] = FunctorBase<DeviceType>::GetBlockIdZ() *
                       FunctorBase<DeviceType>::GetBlockDimZ() +
                   FunctorBase<DeviceType>::GetThreadIdZ();

    for (int d = D - 4; d >= 0; d--) {
      idx[d] = bidx % original_data.shape(d);
      bidx /= original_data.shape(d);
    }

    bool in_range = true;
    for (DIM d = 0; d < D; d++) {
      if (idx[d] >= original_data.shape(d))
        in_range = false;
    }
    if (in_range) {
      // printf("%d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
      double rel_error =
          (original_data[idx] - reconstructed_data[idx]) / max_error;
      double bin_size = 1 / num_bins;
      int bin_id = rel_error / bin_size;
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  SubArray<D, T, DeviceType> original_data;
  SubArray<D, T, DeviceType> reconstructed_data;
  double max_error;
  int num_bins;
  int reduced_levels;
  SubArray<D + 1, int, DeviceType> error_distribution;
  IDX threadId;
};

template <DIM D, typename T, typename DeviceType>
class UncertaintyDistributionKernel : public Kernel {
public:
  constexpr static DIM NumDim = D;
  using DataType = T;
  constexpr static std::string_view Name = "lwpk";
  MGARDX_CONT
  UncertaintyDistributionKernel(
      SubArray<D, T, DeviceType> original_data,
      SubArray<D, T, DeviceType> reconstructed_data, double max_error,
      int num_bins, int reduced_levels,
      SubArray<D + 1, int, DeviceType> error_distribution)
      : original_data(original_data), reconstructed_data(reconstructed_data),
        max_error(max_error), num_bins(num_bins),
        reduced_levels(reduced_levels), error_distribution(error_distribution) {
  }
  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<UncertaintyDistributionFunctor<D, T, R, C, F, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType =
        UncertaintyDistributionFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(original_data, reconstructed_data, max_error, num_bins,
                        reduced_levels, error_distribution);

    SIZE total_thread_z = 1;
    SIZE total_thread_y = 1;
    SIZE total_thread_x = 1;
    if (D >= 3)
      total_thread_z = original_data.shape(D - 3);
    if (D >= 2)
      total_thread_y = original_data.shape(D - 2);
    total_thread_x = original_data.shape(D - 1);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    for (int d = D - 4; d >= 0; d--) {
      gridx *= original_data.shape(d);
    }
    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1],
    // shape.dataHost()[0]); PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<D, T, DeviceType> original_data;
  SubArray<D, T, DeviceType> reconstructed_data;
  double max_error;
  int num_bins;
  int reduced_levels;
  SubArray<D + 1, int, DeviceType> error_distribution;
  IDX threadId;
};

template <DIM D, typename T, typename DeviceType> class UncertaintyCollector {

public:
  UncertaintyCollector(Hierarchy<D, T, DeviceType> hierarchy, double max_error,
                       int num_bins)
      : hierarchy(hierarchy), max_error(max_error), num_bins(num_bins) {}

  void Collect(Array<D, T, DeviceType> &original_data,
               Array<D, T, DeviceType> &reconstructed_data,
               int reconstructed_final_level,
               Array<D + 1, int, DeviceType> &error_dist, int queue_idx) {
    // PrintSubarray("original_data", SubArray(original_data));
    // PrintSubarray("reconstructed_data", SubArray(reconstructed_data));
    DeviceLauncher<DeviceType>::Execute(
        UncertaintyDistributionKernel<D, T, DeviceType>(
            SubArray(original_data), SubArray(reconstructed_data), max_error,
            num_bins, hierarchy.l_target() - reconstructed_final_level,
            SubArray(error_dist)),
        queue_idx);
  }

  Hierarchy<D, T, DeviceType> hierarchy;
  double max_error;
  int num_bins;
};

} // namespace mgard_x
#endif