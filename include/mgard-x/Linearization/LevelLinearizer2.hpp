#ifndef MGARD_X_LEVEL_LINEARIZER_2_TEMPLATE_HPP
#define MGARD_X_LEVEL_LINEARIZER_2_TEMPLATE_HPP

namespace mgard_x {

#define Interleave 0
#define Reposition 1

template <DIM D, typename T, int R, int C, int F, OPTION Direction,
          typename DeviceType>
class LevelLinearizer2Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  LevelLinearizer2Functor() {}
  MGARDX_CONT
  LevelLinearizer2Functor(SubArray<1, SIZE, DeviceType> ranges, SIZE l_target,
                          SubArray<D, T, DeviceType> v,
                          SubArray<1, T, DeviceType> *level_v)
      : ranges(ranges), l_target(l_target), v(v), level_v(level_v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {

    debug = false;
    if (FunctorBase<DeviceType>::GetBlockIdZ() == 0 &&
        FunctorBase<DeviceType>::GetBlockIdY() == 0 &&
        FunctorBase<DeviceType>::GetBlockIdX() == 0 &&
        FunctorBase<DeviceType>::GetThreadIdX() == 0 &&
        FunctorBase<DeviceType>::GetThreadIdY() == 0 &&
        FunctorBase<DeviceType>::GetThreadIdZ() == 0)
      debug = true;

    SIZE threadId = (FunctorBase<DeviceType>::GetThreadIdZ() *
                     FunctorBase<DeviceType>::GetBlockDimX() *
                     FunctorBase<DeviceType>::GetBlockDimY()) +
                    (FunctorBase<DeviceType>::GetThreadIdY() *
                     FunctorBase<DeviceType>::GetBlockDimX()) +
                    FunctorBase<DeviceType>::GetThreadIdX();

    int8_t *sm_p = (int8_t *)FunctorBase<DeviceType>::GetSharedMemory();
    ranges_sm = (SIZE *)sm_p;
    sm_p += D * (l_target + 2) * sizeof(SIZE);

    for (SIZE i = threadId; i < D * (l_target + 2);
         i += FunctorBase<DeviceType>::GetBlockDimX() *
              FunctorBase<DeviceType>::GetBlockDimY() *
              FunctorBase<DeviceType>::GetBlockDimZ()) {
      ranges_sm[i] = *ranges(i);
    }

    // __syncthreads();
    // if (debug) {
    //   printf("l_target = %u\n", l_target);
    //   for (int d = 0; d < D; d++) {
    //     printf("ranges_sm[d = %d]: ", d);
    //     for (int l = 0; l < l_target + 2; l++) {
    //       printf("%u ", ranges_sm[d * (l_target + 2) + l]);
    //     }
    //     printf("\n");
    //   }
    // }

    // __syncthreads();
  }

  MGARDX_EXEC void Operation2() {

    SIZE idx[D];

    SIZE firstD = div_roundup(ranges_sm[l_target + 1], F);
    if (debug) {
      // printf("ranges_sm[l_target + 1]: %u\n", ranges_sm[l_target + 1]);
    }
    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    idx[0] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();
    bidx /= firstD;
    if (D >= 2) {
      idx[1] = FunctorBase<DeviceType>::GetBlockIdY() *
                   FunctorBase<DeviceType>::GetBlockDimY() +
               FunctorBase<DeviceType>::GetThreadIdY();
    }
    if (D >= 3) {
      idx[2] = FunctorBase<DeviceType>::GetBlockIdZ() *
                   FunctorBase<DeviceType>::GetBlockDimZ() +
               FunctorBase<DeviceType>::GetThreadIdZ();
    }

    for (DIM d = 3; d < D; d++) {
      idx[d] = bidx % ranges_sm[(l_target + 2) * d + l_target + 1];
      bidx /= ranges_sm[(l_target + 2) * d + l_target + 1];
    }

    bool in_range = true;
    for (DIM d = 0; d < D; d++) {
      if (idx[d] >= ranges_sm[(l_target + 2) * d + l_target + 1]) {
        in_range = false;
      }
    }

    if (in_range) {
      SIZE level = 0;
      long long unsigned int l_bit[D];
      for (DIM d = 0; d < D; d++) {
        l_bit[d] = 0l;
        for (SIZE l = 0; l < l_target + 1; l++) {
          long long unsigned int bit =
              (idx[d] >= ranges_sm[(l_target + 2) * d + l]) &&
              (idx[d] < ranges_sm[(l_target + 2) * d + l + 1]);
          l_bit[d] += bit << l;
        }
        level = Math<DeviceType>::Max((int)level,
                                      Math<DeviceType>::ffsll(l_bit[d]));
      }

      // distinguish different regions
      SIZE curr_region = 0;
      for (DIM d = 0; d < D; d++) {
        // SIZE bit = !(level == Math<DeviceType>::ffsll(l_bit[d]));
        // curr_region += bit << (D - 1 - d);
        SIZE bit = level == Math<DeviceType>::ffsll(l_bit[d]);
        curr_region += bit << d;
      }

      // convert to 0 based level
      level = level - 1;

      // region size
      SIZE coarse_level_size[D];
      SIZE diff_level_size[D];
      SIZE curr_region_dims[D];
      for (DIM d = 0; d < D; d++) {
        coarse_level_size[d] = ranges_sm[(l_target + 2) * d + level];
        diff_level_size[d] = ranges_sm[(l_target + 2) * d + level + 1] -
                             ranges_sm[(l_target + 2) * d + level];
      }

      for (DIM d = 0; d < D; d++) {
        // SIZE bit = (curr_region >> (D - 1 - d)) & 1u;
        // curr_region_dims[d] = bit ? coarse_level_size[d] :
        // diff_level_size[d];

        SIZE bit = (curr_region >> d) & 1u;
        curr_region_dims[d] = bit ? diff_level_size[d] : coarse_level_size[d];
      }

      SIZE curr_region_size = 1;
      for (DIM d = 0; d < D; d++) {
        curr_region_size *= curr_region_dims[d];
      }

      // printf("(%u %u %u): level: %u, region: %u, dims: %u %u %u
      // coarse_level_size: %u %u %u, diff_level_size: %u %u %u\n", idx[0],
      // idx[1], idx[2],
      //                                           level, curr_region,
      //                                           curr_region_dims[0],
      //                                           curr_region_dims[1],
      //                                           curr_region_dims[2],
      //                                           coarse_level_size[0],
      //                                           coarse_level_size[1],
      //                                           coarse_level_size[2],
      //                                           diff_level_size[0],
      //                                           diff_level_size[1],
      //                                           diff_level_size[2]);

      // region offset
      SIZE curr_region_offset = 0;
      // for (SIZE prev_region = 0; prev_region < curr_region; prev_region++) {
      for (SIZE prev_region = 1; prev_region < curr_region; prev_region++) {
        SIZE prev_region_size = 1;
        for (DIM d = 0; d < D; d++) {
          // SIZE bit = (prev_region >> (D - 1 - d)) & 1u;
          // prev_region_size *= bit ? coarse_level_size[d] :
          // diff_level_size[d];
          SIZE bit = (prev_region >> d) & 1u;
          prev_region_size *= bit ? diff_level_size[d] : coarse_level_size[d];
        }
        curr_region_offset += prev_region_size;
      }

      // printf("(%u %u): level: %u, curr_region: %u, curr_region_offset: %u\n",
      // idx[0], idx[1], level, curr_region, curr_region_offset);

      // thread offset
      SIZE curr_region_thread_idx[D];
      SIZE curr_thread_offset = 0;
      SIZE coarse_level_offset = 0;
      for (SIZE d = 0; d < D; d++) {
        // SIZE bit = (curr_region >> D - 1 - d) & 1u;
        // curr_region_thread_idx[d] = bit ? idx[d] : idx[d] -
        // coarse_level_size[d];
        SIZE bit = (curr_region >> d) & 1u;
        curr_region_thread_idx[d] =
            bit ? idx[d] - coarse_level_size[d] : idx[d];
      }

      SIZE global_data_idx[D];
      for (SIZE d = 0; d < D; d++) {
        SIZE bit = (curr_region >> d) & 1u;
        if (level == 0) {
          global_data_idx[d] = curr_region_thread_idx[d];
        } else if (ranges_sm[(l_target + 2) * d + level + 1] % 2 == 0 &&
                   curr_region_thread_idx[d] ==
                       ranges_sm[(l_target + 2) * d + level + 1] / 2) {
          global_data_idx[d] = ranges_sm[(l_target + 2) * d + level + 1] - 1;
        } else {
          global_data_idx[d] = curr_region_thread_idx[d] * 2 + bit;
        }
      }

      SIZE stride = 1;
      for (SIZE d = 0; d < D; d++) {
        curr_thread_offset += global_data_idx[d] * stride;
        stride *= ranges_sm[(l_target + 2) * d + level + 1];
      }

      stride = 1;
      for (SIZE d = 0; d < D; d++) {
        if (global_data_idx[d] % 2 != 0 &&
            global_data_idx[d] !=
                ranges_sm[(l_target + 2) * d + level + 1] - 1) {
          coarse_level_offset = 0;
        }
        if (global_data_idx[d]) {
          coarse_level_offset += ((global_data_idx[d] - 1) / 2 + 1) * stride;
        }
        stride *= (ranges_sm[(l_target + 2) * d + level + 1]) / 2 + 1;
      }

      if (level == 0)
        coarse_level_offset = 0;

      SIZE level_offset = curr_thread_offset - coarse_level_offset;

      // if (level == 3)
      // printf("(%u %u): level: %u, curr_region: %u, (%u %u)(%u %u), (%u %u)\
      // curr_thread_offset: %u, coarse_level_offset: %u, level_offset: %u\n",
      //                                           idx[0], idx[1],
      //                                           level, curr_region,
      //                                           curr_region_thread_idx[0],
      //                                           curr_region_thread_idx[1],
      //                                           global_data_idx[0],
      //                                           global_data_idx[1],
      //                                           ranges_sm[(l_target + 2) * 0
      //                                           + level + 1],
      //                                           ranges_sm[(l_target + 2) * 1
      //                                           + level + 1],
      //                                           curr_thread_offset,
      //                                           coarse_level_offset,
      //                                           level_offset);

      if (Direction == Interleave) {
        // printf("%u %u %u (%f) --> %u\n", idx[0], idx[1], idx[2], *v(idx),
        // level_offset);
        *(level_v[level]((IDX)level_offset)) = *v(idx);
      } else if (Direction == Reposition) {
        *v(idx) = *(level_v[level]((IDX)level_offset));
      }
    }
  }

  MGARDX_EXEC void Operation3() {}

  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size += D * (l_target + 2) * sizeof(SIZE);
    // printf("sm_size: %llu\n", size);
    return size;
  }

private:
  SubArray<1, SIZE, DeviceType> ranges;
  SIZE l_target;
  SubArray<D, T, DeviceType> v;
  SubArray<1, T, DeviceType> *level_v;

  // thread private variables
  bool debug;
  SIZE *ranges_sm;
};

template <DIM D, typename T, OPTION Direction, typename DeviceType>
class LevelLinearizer2 : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  LevelLinearizer2() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT
      Task<LevelLinearizer2Functor<D, T, R, C, F, Direction, DeviceType>>
      GenTask(SubArray<1, SIZE, DeviceType> shape, SIZE l_target,
              SubArray<1, SIZE, DeviceType> ranges,
              SubArray<D, T, DeviceType> v, SubArray<1, T, DeviceType> *level_v,
              int queue_idx) {
    using FunctorType =
        LevelLinearizer2Functor<D, T, R, C, F, Direction, DeviceType>;
    FunctorType functor(ranges, l_target, v, level_v);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    int total_thread_z = shape.dataHost()[2];
    int total_thread_y = shape.dataHost()[1];
    int total_thread_x = shape.dataHost()[0];
    // linearize other dimensions
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    // printf("LevelLinearizer2 config: %u %u %u %u %u %u\n", tbx, tby, tbz,
    // gridx, gridy, gridz);
    for (int d = 3; d < D; d++) {
      gridx *= shape.dataHost()[d];
    }
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx);
  }

  MGARDX_CONT
  void Execute(SubArray<1, SIZE, DeviceType> shape, SIZE l_target,
               SubArray<1, SIZE, DeviceType> ranges,
               SubArray<D, T, DeviceType> v,
               SubArray<1, T, DeviceType> linearized_v, int queue_idx) {
   
    SubArray<1, T, DeviceType> *level_v =
        new SubArray<1, T, DeviceType>[l_target + 1];
    SIZE *ranges_h = ranges.dataHost();
    SIZE last_level_size = 0;
    for (SIZE l = 0; l < l_target + 1; l++) {
      SIZE level_size = 1;
      for (DIM d = 0; d < D; d++) {
        level_size *= ranges_h[d * (l_target + 2) + l + 1];
      }
      level_v[l] = SubArray<1, T, DeviceType>({level_size - last_level_size},
                                              linearized_v(last_level_size));
      last_level_size = level_size;
    }

    SubArray<1, T, DeviceType> *d_level_v;
    MemoryManager<DeviceType>::Malloc1D(d_level_v, l_target + 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncDevice();
    MemoryManager<DeviceType>::Copy1D(d_level_v, level_v, l_target + 1,
                                      queue_idx);
    DeviceRuntime<DeviceType>::SyncDevice();


    int range_l = std::min(6, (int)std::log2(v.getShape(0)) - 1);
    int prec = TypeToIdx<T>();

    int config = AutoTuner<DeviceType>::autoTuningTable.llk[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;

    #define LLK(CONFIG)                                                          \
    if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
      const int R = LWPK_CONFIG[D - 1][CONFIG][0];                               \
      const int C = LWPK_CONFIG[D - 1][CONFIG][1];                               \
      const int F = LWPK_CONFIG[D - 1][CONFIG][2];                               \
      using FunctorType =                                                       \
        LevelLinearizer2Functor<D, T, R, C, F, Direction, DeviceType>;           \
      using TaskType = Task<FunctorType>;                                          \
      TaskType task =                                                              \
        GenTask<R, C, F>(shape, l_target, ranges, v, d_level_v, queue_idx);      \
      DeviceAdapter<TaskType, DeviceType> adapter;                                 \
      ExecutionReturn ret = adapter.Execute(task);                               \
      if (AutoTuner<DeviceType>::ProfileKernels) {                               \
        if (min_time > ret.execution_time) {                                     \
          min_time = ret.execution_time;                                         \
          min_config = CONFIG;                                                   \
        }                                                                        \
      }                                                                          \
    }
    LLK(0)
    LLK(1)
    LLK(2)
    LLK(3)
    LLK(4)
    LLK(5)
    LLK(6)
#undef LLK
    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("llk", prec, range_l, min_config);
    }
  }
};

} // namespace mgard_x

#endif