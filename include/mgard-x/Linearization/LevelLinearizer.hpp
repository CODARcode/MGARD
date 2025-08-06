#ifndef MGARD_X_LEVEL_LINEARIZER_TEMPLATE_HPP
#define MGARD_X_LEVEL_LINEARIZER_TEMPLATE_HPP

namespace mgard_x {

#define Interleave 0
#define Reposition 1

template <DIM D, typename T, int R, int C, int F, OPTION Direction,
          typename DeviceType>
class LevelLinearizerFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  LevelLinearizerFunctor() {}
  MGARDX_CONT
  LevelLinearizerFunctor(SubArray<2, SIZE, DeviceType> level_ranges,
                         SIZE l_target, SubArray<D, T, DeviceType> v,
                         SubArray<1, T, DeviceType> *level_v)
      : level_ranges(level_ranges), l_target(l_target), v(v), level_v(level_v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE idx[D];

    SIZE firstD = div_roundup(v.shape(D - 1), F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    idx[D - 1] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();
    bidx /= firstD;
    if (D >= 2) {
      idx[D - 2] = FunctorBase<DeviceType>::GetBlockIdY() *
                       FunctorBase<DeviceType>::GetBlockDimY() +
                   FunctorBase<DeviceType>::GetThreadIdY();
    }
    if (D >= 3) {
      idx[D - 3] = FunctorBase<DeviceType>::GetBlockIdZ() *
                       FunctorBase<DeviceType>::GetBlockDimZ() +
                   FunctorBase<DeviceType>::GetThreadIdZ();
    }

    for (int d = D - 4; d >= 0; d--) {
      idx[d] = bidx % v.shape(d);
      bidx /= v.shape(d);
    }

    bool in_range = true;
    for (int d = D - 1; d >= 0; d--) {
      if (idx[d] >= v.shape(d)) {
        in_range = false;
      }
    }

    if (in_range) {
      SIZE level = 0;
      long long unsigned int l_bit[D];
      for (int d = D - 1; d >= 0; d--) {
        l_bit[d] = 0l;
        for (SIZE l = 0; l < l_target + 1; l++) {
          long long unsigned int bit = (idx[d] >= *level_ranges(l, d)) &&
                                       (idx[d] < *level_ranges(l + 1, d));
          l_bit[d] += bit << l;
        }
        level = Math<DeviceType>::Max((int)level,
                                      Math<DeviceType>::ffsll(l_bit[d]));
      }

      // Use curr_region to encode region id to distinguish different regions
      // curr_region of current level is always >=1,
      // since curr_region=0 refers to the next coarser level
      // most significant bit --> fastest dim
      // least signigiciant bit --> slowest dim
      SIZE curr_region = 0;
      for (int d = D - 1; d >= 0; d--) {
        SIZE bit = level == Math<DeviceType>::ffsll(l_bit[d]);
        curr_region += bit << d;
      }

      // convert to 0 based level
      level = level - 1;

      // region size
      SIZE coarse_level_size[D];
      SIZE diff_level_size[D];
      for (int d = D - 1; d >= 0; d--) {
        coarse_level_size[d] = *level_ranges(level, d);
        diff_level_size[d] =
            *level_ranges(level + 1, d) - *level_ranges(level, d);
      }

      SIZE curr_region_dims[D];
      for (int d = D - 1; d >= 0; d--) {
        // Use region id to decode dimension of this region
        SIZE bit = (curr_region >> d) & 1u;
        curr_region_dims[d] = bit ? diff_level_size[d] : coarse_level_size[d];
      }

      SIZE curr_region_size = 1;
      for (int d = D - 1; d >= 0; d--) {
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
      // prev_region start with 1 since that is the region id of the first
      // region of current level
      for (SIZE prev_region = 1; prev_region < curr_region; prev_region++) {
        SIZE prev_region_size = 1;
        for (int d = D - 1; d >= 0; d--) {
          // Use region id to decode dimension of a previous region
          SIZE bit = (prev_region >> d) & 1u;
          // Calculate the num of elements of the previous region
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
      for (int d = D - 1; d >= 0; d--) {
        SIZE bit = (curr_region >> d) & 1u;
        curr_region_thread_idx[d] =
            bit ? idx[d] - coarse_level_size[d] : idx[d];
      }

      SIZE global_data_idx[D];
      for (int d = D - 1; d >= 0; d--) {
        SIZE bit = (curr_region >> d) & 1u;
        if (level == 0) {
          global_data_idx[d] = curr_region_thread_idx[d];
        } else if (*level_ranges(level + 1, d) % 2 == 0 &&
                   curr_region_thread_idx[d] ==
                       *level_ranges(level + 1, d) / 2) {
          global_data_idx[d] = *level_ranges(level + 1, d) - 1;
        } else {
          global_data_idx[d] = curr_region_thread_idx[d] * 2 + bit;
        }
      }

      SIZE stride = 1;
      for (int d = D - 1; d >= 0; d--) {
        curr_thread_offset += global_data_idx[d] * stride;
        stride *= *level_ranges(level + 1, d);
      }

      stride = 1;
      for (int d = D - 1; d >= 0; d--) {
        if (global_data_idx[d] % 2 != 0 &&
            global_data_idx[d] != *level_ranges(level + 1, d) - 1) {
          coarse_level_offset = 0;
        }
        if (global_data_idx[d]) {
          coarse_level_offset += ((global_data_idx[d] - 1) / 2 + 1) * stride;
        }
        stride *= (*level_ranges(level + 1, d)) / 2 + 1;
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
        *(level_v[level]((IDX)level_offset)) = v[idx];
      } else if (Direction == Reposition) {
        v[idx] = *(level_v[level]((IDX)level_offset));
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  SubArray<2, SIZE, DeviceType> level_ranges;
  SIZE l_target;
  SubArray<D, T, DeviceType> v;
  SubArray<1, T, DeviceType> *level_v;
};

template <DIM D, typename T, OPTION Direction, typename DeviceType>
class LevelLinearizer : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  LevelLinearizer() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<LevelLinearizerFunctor<D, T, R, C, F, Direction, DeviceType>>
  GenTask(SubArray<2, SIZE, DeviceType> level_ranges, SIZE l_target,
          SubArray<D, T, DeviceType> v, SubArray<1, T, DeviceType> *level_v,
          int queue_idx) {
    using FunctorType =
        LevelLinearizerFunctor<D, T, R, C, F, Direction, DeviceType>;
    FunctorType functor(level_ranges, l_target, v, level_v);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    int total_thread_z = v.shape(D - 3);
    int total_thread_y = v.shape(D - 2);
    int total_thread_x = v.shape(D - 1);
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    for (int d = D - 4; d >= 0; d--) {
      gridx *= v.shape(d);
    }
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "LevelLinearizer");
  }

  MGARDX_CONT
  void Execute(SubArray<2, SIZE, DeviceType> level_ranges, SIZE l_target,
               SubArray<D, T, DeviceType> v,
               SubArray<1, T, DeviceType> linearized_v, int queue_idx) {

    SubArray<1, T, DeviceType> *level_v =
        new SubArray<1, T, DeviceType>[l_target + 1];
    SIZE *ranges_h = level_ranges.dataHost();
    SIZE last_level_size = 0;
    for (SIZE l = 0; l < l_target + 1; l++) {
      SIZE level_size = 1;
      for (DIM d = 0; d < D; d++) {
        level_size *= ranges_h[(l + 1) * D + d];
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

    int range_l = std::min(6, (int)std::log2(v.shape(D - 1)) - 1);
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.llk[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define LLK(CONFIG)                                                            \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = LWPK_CONFIG[D - 1][CONFIG][0];                               \
    const int C = LWPK_CONFIG[D - 1][CONFIG][1];                               \
    const int F = LWPK_CONFIG[D - 1][CONFIG][2];                               \
    using FunctorType =                                                        \
        LevelLinearizerFunctor<D, T, R, C, F, Direction, DeviceType>;          \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task =                                                            \
        GenTask<R, C, F>(level_ranges, l_target, v, d_level_v, queue_idx);     \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ret = adapter.Execute(task);                                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (ret.success && min_time > ret.execution_time) {                      \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }
    LLK(6) if (!ret.success) config--;
    LLK(5) if (!ret.success) config--;
    LLK(4) if (!ret.success) config--;
    LLK(3) if (!ret.success) config--;
    LLK(2) if (!ret.success) config--;
    LLK(1) if (!ret.success) config--;
    LLK(0) if (!ret.success) config--;
    if (config < 0 && !ret.success) {
      std::cout << log::log_err << "no suitable config for LevelLinearizer.\n";
      exit(-1);
    }
#undef LLK
    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("llk", prec, range_l, min_config);
    }
  }
};

} // namespace mgard_x

#endif