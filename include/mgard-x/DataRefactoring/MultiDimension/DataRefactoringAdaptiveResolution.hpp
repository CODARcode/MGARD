/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../Hierarchy/Hierarchy.hpp"
#include "../../RuntimeX/RuntimeX.h"

#include "DataRefactoring.h"

#include "AdaptiveResolution/AdaptiveResolutionTree.hpp"

#include <iostream>

#ifndef MGARD_X_DATA_REFACTORING_ADAPTIVE_RESOLUTION_HPP
#define MGARD_X_DATA_REFACTORING_ADAPTIVE_RESOLUTION_HPP

namespace mgard_x {

template <DIM D, typename T, typename FeatureDetectorType, typename DeviceType>
Array<D, T, DeviceType> recompose_adaptive_resolution(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, T tol, bool interpolate_full_resolution, FeatureDetectorType feature_detector, int queue_idx) {
  AdaptiveResolutionTree tree(hierarchy);
  tree.buildTree(v);
  return tree.constructData(tol, interpolate_full_resolution, feature_detector, queue_idx);
}

}

#endif