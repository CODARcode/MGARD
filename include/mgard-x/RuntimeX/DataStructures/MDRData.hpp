/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MDR_X_MDR_DATA_HPP
#define MDR_X_MDR_DATA_HPP

#include "MDRMetadata.hpp"

namespace mgard_x {
namespace MDR {

template <typename DeviceType> class MDRData {
public:
  MDRData() {}
  // Data
  std::vector<std::vector<Array<1, Byte, DeviceType>>> compressed_bitplanes;
  std::vector<Array<1, bool, DeviceType>> level_signs;

  MDRData(SIZE num_levels, SIZE num_bitplanes) {
    compressed_bitplanes.resize(num_levels);
    level_signs.resize(num_levels);
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      compressed_bitplanes[level_idx].resize(num_bitplanes);
    }
  }

  // Resize before refactoring
  template <typename RefactorType, typename HierarchyType>
  void Resize(RefactorType &refactor, HierarchyType &hierarchy, int queue_idx) {
    std::vector<std::vector<SIZE>> estimation =
        RefactorType::EstimateMaxBitplaneSizes(hierarchy);
    SIZE num_levels = estimation.size();
    SIZE num_bitplanes = estimation[0].size();
    std::vector<SIZE> level_num_elems = hierarchy.level_num_elems();
    compressed_bitplanes.resize(num_levels);
    level_signs.resize(num_levels);
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      compressed_bitplanes[level_idx].resize(num_bitplanes);
      // The decoder's batched bitplane kernels (e.g.
      // BPDecoderRegisterBlockFunctor::DecodeBinary) write one sign per slot
      // across the full BATCH_SIZE-aligned range, exactly like
      // level_data_array in ComposedRefactor/ComposedReconstructor -- so this
      // needs the same round_up, not the raw element count, or the decoder
      // overruns it whenever a level's element count isn't already a
      // multiple of the batch size.
      level_signs[level_idx].resize(
          {round_up(hierarchy.level_num_elems(level_idx),
                    RefactorType::BATCH_SIZE)},
          queue_idx);
      for (int bitplane_idx = 0; bitplane_idx < num_bitplanes; bitplane_idx++) {
        compressed_bitplanes[level_idx][bitplane_idx].resize(
            {estimation[level_idx][bitplane_idx]}, queue_idx);
      }
    }
  }

  // Resize before reconstruction
  template <typename ReconstructorType>
  void Resize(ReconstructorType &reconstructor, MDRMetadata &mdr_metadata,
              int queue_idx) {
    compressed_bitplanes.resize(mdr_metadata.num_levels);
    level_signs.resize(mdr_metadata.num_levels);
    for (int level_idx = 0; level_idx < mdr_metadata.num_levels; level_idx++) {
      compressed_bitplanes[level_idx].resize(mdr_metadata.num_bitplanes);
      // Rounded up to the decoder's batch size, as in the refactoring Resize
      // above: the decoder writes a sign for every slot of its last batch.
      level_signs[level_idx].resize(
          {round_up((SIZE)mdr_metadata.level_num_elems[level_idx],
                    ReconstructorType::BATCH_SIZE)},
          queue_idx);
      for (int bitplane_idx = 0; bitplane_idx < mdr_metadata.num_bitplanes;
           bitplane_idx++) {
        compressed_bitplanes[level_idx][bitplane_idx].resize(
            {mdr_metadata.level_sizes[level_idx][bitplane_idx]}, queue_idx);
      }
    }
  }

  void ResetSigns(int queue_idx) {
    for (int level_idx = 0; level_idx < level_signs.size(); level_idx++) {
      level_signs[level_idx].memset(0, queue_idx);
    }
  }

  void CopyFromRefactoredData(MDRMetadata &mdr_metadata,
                              std::vector<std::vector<Byte *>> &refactored_data,
                              int queue_idx) {
    for (int level_idx = 0; level_idx < mdr_metadata.num_levels; level_idx++) {
      for (int bitplane_idx =
               mdr_metadata.loaded_level_num_bitplanes[level_idx];
           bitplane_idx < mdr_metadata.requested_level_num_bitplanes[level_idx];
           bitplane_idx++) {
        MemoryManager<DeviceType>::Copy1D(
            compressed_bitplanes[level_idx][bitplane_idx].data(),
            refactored_data[level_idx][bitplane_idx],
            mdr_metadata.level_sizes[level_idx][bitplane_idx], queue_idx);
      }
    }
    mdr_metadata.DoneLoadingBitplans();
  }

  void CopyFromRefactoredSigns(MDRMetadata &mdr_metadata,
                               std::vector<bool *> &refactored_level_signs,
                               int queue_idx) {
    for (int level_idx = 0; level_idx < mdr_metadata.num_levels; level_idx++) {
      MemoryManager<DeviceType>::Copy1D(
          level_signs[level_idx].data(), refactored_level_signs[level_idx],
          mdr_metadata.level_num_elems[level_idx], queue_idx);
    }
  }

  void CopyToRefactoredData(MDRMetadata &mdr_metadata,
                            std::vector<std::vector<Byte *>> &refactored_data,
                            std::vector<std::vector<SIZE>> &allocation_size,
                            int queue_idx) {
    refactored_data.resize(mdr_metadata.num_levels);
    for (int level_idx = 0; level_idx < mdr_metadata.num_levels; level_idx++) {
      refactored_data[level_idx].resize(mdr_metadata.num_bitplanes);
      for (int bitplane_idx = 0; bitplane_idx < mdr_metadata.num_bitplanes;
           bitplane_idx++) {
        if (allocation_size[level_idx][bitplane_idx] >=
            mdr_metadata.level_sizes[level_idx][bitplane_idx]) {
          MemoryManager<DeviceType>::Copy1D(
              refactored_data[level_idx][bitplane_idx],
              compressed_bitplanes[level_idx][bitplane_idx].data(),
              mdr_metadata.level_sizes[level_idx][bitplane_idx], queue_idx);
        } else {
          throw std::runtime_error(
              "Bitplane copy failed. level_idx(" + std::to_string(level_idx) +
              ") bitplane_idx(" + std::to_string(bitplane_idx) +
              ") Insufficient buffer space " +
              std::to_string(allocation_size[level_idx][bitplane_idx]) +
              " vs. " +
              std::to_string(
                  mdr_metadata.level_sizes[level_idx][bitplane_idx]));
        }
      }
    }
  }

  void CopyToRefactoredSigns(MDRMetadata &mdr_metadata,
                             std::vector<bool *> &refactored_level_signs,
                             int queue_idx) {
    for (int level_idx = 0; level_idx < mdr_metadata.num_levels; level_idx++) {
      MemoryManager<DeviceType>::Copy1D(
          refactored_level_signs[level_idx], level_signs[level_idx].data(),
          mdr_metadata.level_num_elems[level_idx], queue_idx);
    }
  }

  void VerifyLoadedBitplans(MDRMetadata &mdr_metadata) {
    // TODO: load
    for (int level_idx = 0; level_idx < mdr_metadata.num_levels; level_idx++) {
      if (!level_signs[level_idx].hasDeviceAllocation()) {
        throw std::runtime_error("Bitplane verification failed. level_signs(" +
                                 std::to_string(level_idx) +
                                 ") No allocation found.\n");
      }
      // level_signs is allocated rounded up to the decoder's batch size (see
      // Resize above), so it is expected to be >= the raw element count, not
      // exactly equal to it.
      if (level_signs[level_idx].shape(0) <
          mdr_metadata.level_num_elems[level_idx]) {
        throw std::runtime_error(
            "Bitplane verification failed. level_signs(" +
            std::to_string(level_idx) + ") Insufficient buffer space " +
            std::to_string(level_signs[level_idx].shape(0)) + " vs. " +
            std::to_string(mdr_metadata.level_num_elems[level_idx]));
      }
      for (int bitplane_idx =
               mdr_metadata.prev_used_level_num_bitplanes[level_idx];
           bitplane_idx < mdr_metadata.loaded_level_num_bitplanes[level_idx];
           bitplane_idx++) {
        if (!compressed_bitplanes[level_idx][bitplane_idx]
                 .hasDeviceAllocation()) {
          throw std::runtime_error(
              "Bitplane verification failed. level_idx(" +
              std::to_string(level_idx) + ") bitplane_idx(" +
              std::to_string(bitplane_idx) + ") No allocation found.\n");
        }
        if (compressed_bitplanes[level_idx][bitplane_idx].shape(0) !=
            mdr_metadata.level_sizes[level_idx][bitplane_idx]) {
          throw std::runtime_error(
              "Bitplane verification failed. level_idx(" +
              std::to_string(level_idx) + ") bitplane_idx(" +
              std::to_string(bitplane_idx) + ") Size mismatch " +
              std::to_string(
                  compressed_bitplanes[level_idx][bitplane_idx].shape(0)) +
              " vs. " +
              std::to_string(
                  mdr_metadata.level_sizes[level_idx][bitplane_idx]));
        }
      }
    }
  }
};

} // namespace MDR
} // namespace mgard_x

#endif