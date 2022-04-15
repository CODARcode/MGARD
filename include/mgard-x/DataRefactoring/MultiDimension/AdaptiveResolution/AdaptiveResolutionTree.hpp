/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include <queue>

#include "../../../Hierarchy/Hierarchy.h"
#include "../../../RuntimeX/RuntimeXPublic.h"
// #include "../Correction/LevelwiseProcessingKernel.hpp"
#include "../DataRefactoring.h"

#include "../../../MDR-X/BitplaneEncoder/GroupedBPEncoderGPU.hpp"

#include <fstream> //for dumping files

#ifndef MGARD_X_ADAPTIVE_RESOLUTION_TREE
#define MGARD_X_ADAPTIVE_RESOLUTION_TREE

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
class AdaptiveResolutionTreeNode {

public:

  using T_bitplane = uint32_t;
  using T_error = double;

  using BoundingIntervalHierarchyType =
      std::vector<std::vector<std::vector<std::pair<SIZE, SIZE>>>>;
  using HierarchicalIndexType = std::vector<std::vector<std::vector<SIZE>>>;

  using ErrorImpactType = std::vector<T_error>;

  using IndexType = std::vector<SIZE>;

  

  AdaptiveResolutionTreeNode() {}
  AdaptiveResolutionTreeNode(Hierarchy<D, T, DeviceType> *hierarchy, std::vector<SIZE> index_start,
                             std::vector<SIZE> index_end, SIZE level, SubArray<D, T, SERIAL> coeff,
                             BoundingIntervalHierarchyType bounding_interval_hierarchy,
                             HierarchicalIndexType hierarchical_index, T max_coefficient)
                            : hierarchy(hierarchy), index_start(index_start), index_end(index_end),
                              level(level), coeff(coeff),
                              bounding_interval_hierarchy(bounding_interval_hierarchy), hierarchical_index(hierarchical_index){
    
    if constexpr (std::is_same<T, float>::value) {  
      num_bitplanes = 32;      
    } else if constexpr (std::is_same<T, double>::value) {
      num_bitplanes = 64;
    }

    max_error = std::vector<T_error>(num_bitplanes + 1);
    int level_exp = 0;
    frexp(max_coefficient, &level_exp);
    max_error[0] = max_coefficient;
    T_error err = ldexp(1.0, level_exp - 1);
    for (int i = 1; i < max_error.size(); i++) {
      max_error[i] = err;
      err /= 2;
    }

    // std::cout << "Node at level " << level << " error: ";
    // for (int i = 0; i < max_error.size(); i++) {
    //   std::cout << max_error[i] << " ";
    // }
    // std::cout << "\n";
    index_start_reordered.resize(D);
    index_end_reordered.resize(D);
    for (DIM d = 0; d < D; d++) {
      for (int i = 0; i < hierarchical_index[d][level].size(); i++) {
        if(hierarchical_index[d][level][i] == index_start[d]) {
          index_start_reordered[d] = i;
        }
        if (hierarchical_index[d][level][i] == index_end[d]) {
          index_end_reordered[d] = i;
        }
      }
    }
  }
  // metadata
  Hierarchy<D, T, DeviceType> *hierarchy = NULL;
  bool is_leaf = true;
  std::vector<SIZE> index_start; // inclusive
  std::vector<SIZE> index_end;   // inclusive
  std::vector<SIZE> index_start_reordered; // inclusive
  std::vector<SIZE> index_end_reordered;   // inclusive

  std::vector<T> coord_start;  // inclusive
  std::vector<T> coord_end;    // inclusive
  SIZE level;
  SIZE max_level;
  SubArray<D, T, SERIAL> coeff;
  bool contain_ghost_cell[D];
  AdaptiveResolutionTreeNode<D, T, DeviceType> *child = NULL;
  SIZE num_children = 0;
  BoundingIntervalHierarchyType bounding_interval_hierarchy;
  HierarchicalIndexType hierarchical_index;
  SIZE num_bitplanes = 0;
  bool potential_contain_feature = true;

  // // Max coefficient of mine (keeping all bitplanes)
  // T max_coefficient = 0.0;

  // Max error when keep N bitplane of my coefficient
  std::vector<T_error> max_error;

  // Error impact if discarding all children 
  ErrorImpactType all_children_error_impact;

  // Error impact if discarding some of my children 
  std::vector<ErrorImpactType> child_error_impact;

  // Coefficent index of all my children
  std::vector<std::vector<IndexType>> child_coefficient_index;

  // Coefficent index of all my children (reordered)
  std::vector<std::vector<IndexType>> child_coefficient_index_reordered;



  // Bitplanes
  std::vector<int> child_exp;
  std::vector<Array<2, T_bitplane, DeviceType>> child_encoded_bitplanes;

  ErrorImpactType combine_error_impact(ErrorImpactType error_impact1, ErrorImpactType error_impact2) {
    if (error_impact1.size() != hierarchy->l_target + 1 || 
        error_impact2.size() != hierarchy->l_target + 1) {
      std::cout << log::log_err << "Wrong error_impact input length.\n";
      exit(-1);
    }
    ErrorImpactType combined_error_impact(hierarchy->l_target + 1);
    for (int i = 0; i < hierarchy->l_target + 1; i++) {
      combined_error_impact[i] = std::max(error_impact1[i], error_impact2[i]);
    }
    return combined_error_impact;
  }

  ErrorImpactType initialize() {

    // std::cout << "level " << level << " curr ";
    // std::cout << " start: ";
    // for (int d = D - 1; d >= 0; d--) std::cout << index_start[d] << " ";
    // std::cout << "end: ";
    // for (int d = D - 1; d >= 0; d--) std::cout << index_end[d] << " ";
    // std::cout << "\n";
    // std::cout << " start-reordered: ";
    // for (int d = D - 1; d >= 0; d--) std::cout << index_start_reordered[d] << " ";
    // std::cout << "end-reordered: ";
    // for (int d = D - 1; d >= 0; d--) std::cout << index_end_reordered[d] << " ";
    // std::cout << "\n";

    SIZE next_level = level + 1;

    if (next_level >= hierarchy->l_target + 1) {
      all_children_error_impact = ErrorImpactType(hierarchy->l_target + 1, 0);
      return all_children_error_impact;
    }
    
    // Calculate the number of children
    SIZE grid_index[D];
    SIZE child_grid_dim[D];
    num_children = 1;
    for (DIM d = 0; d < D; d++) {
      child_grid_dim[d] = 0;
      for (int i = 0; i < hierarchical_index[d][next_level].size(); i++) {
        if(hierarchical_index[d][next_level][i] >= index_start[d] &&
           hierarchical_index[d][next_level][i] < index_end[d]) {
          child_grid_dim[d]++;
        }
        if (hierarchical_index[d][next_level][i] == index_start[d]) {
          grid_index[d] = i;
        }
      }
      num_children *= child_grid_dim[d];
    }

    // std::cout << "num_children: " << num_children << "\n";

    // Allocate memory for child
    child = new AdaptiveResolutionTreeNode<D, T, DeviceType>[num_children];

    // Coefficient index
    child_coefficient_index.resize(num_children);
    child_coefficient_index_reordered.resize(num_children);

    // Error impact
    all_children_error_impact = ErrorImpactType(hierarchy->l_target + 1, 0);
    child_error_impact.resize(num_children);

    // Bitplanes
    child_exp.resize(num_children);
    child_encoded_bitplanes.resize(num_children);

    // Generate coefficient markers
    std::vector<std::vector<int>> coefficient_marker(D);
    for (DIM d = 0; d < D; d++) {
      for (int i = 0; i < hierarchical_index[d][next_level].size(); i++) {
        bool exist_in_prev_level = false; 
        for (int j = 0; j < hierarchical_index[d][level].size(); j++) {
          if (hierarchical_index[d][next_level][i] == hierarchical_index[d][level][j]) {
            exist_in_prev_level = true;
            break;
          }
        }
        if (!exist_in_prev_level) {
          coefficient_marker[d].push_back(hierarchical_index[d][next_level][i]);
        }
      }
    }

    // std::cout << "coefficient_marker: \n";
    // for (DIM d = 0; d < D; d++) {
    //   std::cout << "D: " << d << ": ";
    //   for (int i = 0; i < coefficient_marker[d].size(); i++) {
    //     std::cout << coefficient_marker[d][i] << " ";
    //   }
    //   std::cout << "\n";
    // }

    // Max coefficient amoung all my children
    T all_children_max_coefficient = 0.0;

    // Initialize each child
    for (SIZE r = 0; r < num_children; r++) {

      // Determine child index in the next level
      SIZE child_grid_index[D];
      SIZE linearized_child_index = r;
      for (DIM d = 0; d < D; d++) {
        child_grid_index[d] = linearized_child_index % child_grid_dim[d];
        linearized_child_index /= child_grid_dim[d];
      }

      // Determine child index start/end
      std::vector<SIZE> child_index_start(D);
      std::vector<SIZE> child_index_end(D);
      for (DIM d = 0; d < D; d++) {
        child_index_start[d] = hierarchical_index[d][next_level][child_grid_index[d] + grid_index[d]];
        child_index_end[d] = hierarchical_index[d][next_level][child_grid_index[d] + grid_index[d] + 1];
      }

      // std::cout << "child " << r << " ";
      // std::cout << " start: ";
      // for (int d = D - 1; d >= 0; d--) std::cout << child_index_start[d] << " ";
      // std::cout << "end: ";
      // for (int d = D - 1; d >= 0; d--) std::cout << child_index_end[d] << " ";
      // std::cout << "\n";

      // Determine coefficients correspond to the partition
      SIZE ndof = std::pow(2, D);
      for (SIZE i = 0; i < ndof; i++) {
        std::vector<SIZE> coefficient_index(D);
        SIZE linearized_dof_index = i;
        for (DIM d = 0; d < D; d++) {
          SIZE dof_dim = 2;
          if (linearized_dof_index % dof_dim == 0) {
            coefficient_index[d] = child_index_start[d];
          } else {
            coefficient_index[d] = child_index_end[d];
          }
          linearized_dof_index /= dof_dim;
        }
        bool is_coefficient = false;
        for (DIM d = 0; d < D; d++) {
          for (int i = 0; i < coefficient_marker[d].size(); i++) {
            if (coefficient_index[d] == coefficient_marker[d][i]) {
              is_coefficient = true;
              break;
            }
          }
        }
        if (is_coefficient) {
          child_coefficient_index[r].push_back(coefficient_index);
        }
      }

      // std::cout << "child " << r << " coefficient_index: ";
      // for (SIZE i = 0; i < child_coefficient_index[r].size(); i++) {
      //   std::cout << "(" << child_coefficient_index[r][i][1] << ", " << child_coefficient_index[r][i][0] << ") ";
      // }
      // std::cout << "\n";
      
      // Fetch all coefficients of the child
      std::vector<T> child_coefficients;
      for (SIZE i = 0; i < child_coefficient_index[r].size(); i++) {
        SIZE coefficient_index_reordered[D];
        std::vector<SIZE> coefficient_index_reordered_vec(D);
        // level index
        SIZE stride = std::pow(2, hierarchy->l_target - level);
        for (DIM d = 0; d < D; d++) {
          bool along_coefficient_dim = false;
          for (int m = 0; m < coefficient_marker[d].size(); m++) {
            if (child_coefficient_index[r][i][d] == coefficient_marker[d][m]) {
              along_coefficient_dim = true;
              break;
            }
          }
          if (along_coefficient_dim) {
            coefficient_index_reordered[d] = hierarchy->dofs[d][hierarchy->l_target - level];
            coefficient_index_reordered[d] += (child_coefficient_index[r][i][d] - stride / 2) / stride;
          } else {
            coefficient_index_reordered[d] = std::ceil((float)child_coefficient_index[r][i][d] / stride);
          }
          coefficient_index_reordered_vec[d] = coefficient_index_reordered[d];
        }


        T coefficient = *coeff(coefficient_index_reordered);
        
        child_coefficients.push_back(coefficient);
        child_coefficient_index_reordered[r].push_back(coefficient_index_reordered_vec);
      }

      T child_max_coefficient = 0;
      for (SIZE i = 0; i < child_coefficients.size(); i++) {
        child_max_coefficient = std::max(child_max_coefficient, abs(child_coefficients[i]));
      }

      if (child_coefficients.size() > 0) {

        frexp(child_max_coefficient, &(child_exp[r]));
        Array<1, T, DeviceType> child_coefficients_array({(SIZE)child_coefficients.size()});
        child_coefficients_array.load(child_coefficients.data());
        SubArray child_coefficients_subarray(child_coefficients_array);
        Array<1, T_error, DeviceType> child_square_errors_array({num_bitplanes + 1});
        SubArray<1, T_error, DeviceType> child_square_errors_subarray(child_square_errors_array);
        std::vector<SIZE> bitplane_sizes(num_bitplanes);
        
        auto encoder = mgard_x::MDR::GroupedBPEncoder<T, T_bitplane, T_error, DeviceType>();
        child_encoded_bitplanes[r] = encoder.encode(child_coefficients.size(), num_bitplanes, child_exp[r], 
                       child_coefficients_subarray, child_square_errors_subarray, bitplane_sizes, 0);
      }

      // std::cout << "child_coefficient: ";
      // for (SIZE i = 0; i < child_coefficients.size(); i++) {
      //   std::cout << child_coefficients[i] << ", ";
      // }
      // std::cout << "\n";

      // std::cout << "level " << level << " partition " << r << " child_max_coefficient " << child_max_coefficient << "\n";


      child[r] = AdaptiveResolutionTreeNode<D, T, DeviceType>(
          hierarchy, child_index_start, child_index_end, next_level,
          coeff, bounding_interval_hierarchy, hierarchical_index, child_max_coefficient);

      all_children_max_coefficient = std::max(all_children_max_coefficient, child_max_coefficient);

    }

    // std::cout << "child_coefficient_index: " << 

    for (SIZE r = 0; r < num_children; r++) {
      child_error_impact[r] = child[r].initialize();
    }

    // Gather error impact from children
    for (SIZE r = 0; r < num_children; r++) {
      all_children_error_impact = combine_error_impact(all_children_error_impact, child_error_impact[r]);
      child_error_impact[r][next_level] = child[r].max_error[0];
    }
    all_children_error_impact[next_level] = all_children_max_coefficient;

    return all_children_error_impact;
  }


  Array<1, T, DeviceType> retrieve_child_coefficient(int child_index, SIZE num_bitplanes, int queue_idx) {
    auto encoder = mgard_x::MDR::GroupedBPEncoder<T, T_bitplane, T_error, DeviceType>();
    SIZE starting_bitplane = 0;
    int level = 0; // level need always be 0 so that sign bit are retrieved
    if (child_coefficient_index[child_index].size() > 0 && num_bitplanes > 0) {
      return encoder.progressive_decode(child_coefficient_index[child_index].size(), starting_bitplane, num_bitplanes,
                       child_exp[child_index], child_encoded_bitplanes[child_index],
                       0, queue_idx);
    } else if (child_coefficient_index[child_index].size() > 0){
      Array<1, T, DeviceType> zero_coefficient({(SIZE)child_coefficient_index[child_index].size()});
      zero_coefficient.memset(0);
      return zero_coefficient;
    } else {
      std::cout << "Cannot retrivel zero coefficient.\n";
      exit(-1);
    }
  }

  ~AdaptiveResolutionTreeNode() {
    if (child) {
      for (SIZE i = 0; i < num_children; i++) {
        child[i].~AdaptiveResolutionTreeNode();
      }
    }
  }
};

template <DIM D, typename T, typename DeviceType> class AdaptiveResolutionTree {

  using BoundingIntervalHierarchyType =
      std::vector<std::vector<std::vector<std::pair<SIZE, SIZE>>>>;
  using HierarchicalIndexType = std::vector<std::vector<std::vector<SIZE>>>;
  using NodeType = AdaptiveResolutionTreeNode<D, T, DeviceType>;

public:
  AdaptiveResolutionTree(Hierarchy<D, T, DeviceType> &hierarchy)
      : hierarchy(hierarchy) {}

  std::pair<SIZE, SIZE> merge_intervals(std::pair<SIZE, SIZE> t1,
                                        std::pair<SIZE, SIZE> t2) {
    if (t1.first > t1.second || t2.first > t2.second) {
      std::cout << log::log_err << "merge_interval wrong input.\n";
      exit(-1);
    }
    std::pair<SIZE, SIZE> res;
    if (t1.second == t2.first && t1.first <= t2.second) {
      res.first = t1.first;
      res.second = t2.second;
    } else if (t2.second == t1.first && t2.first <= t1.second) {
      res.first = t2.first;
      res.second = t1.second;
    } else {
      std::cout << log::log_err << "merge_interval wrong input.\n";
      exit(-1);
    }
    return res;
  }

  BoundingIntervalHierarchyType buildBoundingIntervalHierarchy() {
    BoundingIntervalHierarchyType bounding_interval_hierarchy(D);
    for (DIM d = 0; d < D; d++) {
      std::vector<std::vector<std::pair<SIZE, SIZE>>> curr_dim(hierarchy.l_target + 1);
      SIZE dof = hierarchy.dofs[d][0];
      std::vector<std::pair<SIZE, SIZE>> l_bottom(dof - 1);
      // l == hierarchy.l_target
      std::vector<std::pair<SIZE, SIZE>> curr_interval;
      // std::cout << "l_bottom " << d << ": ";
      for (SIZE i = 0; i < dof - 1; i++) {
        l_bottom[i] = std::pair<SIZE, SIZE>(i, i + 1);
        // std::cout << "(" << l_bottom[i].first << ", " << l_bottom[i].second << ") ";
      }
      // std::cout << "\n";

      curr_dim[hierarchy.l_target] = l_bottom;
      for (int l = hierarchy.l_target - 1; l >= 0; l--) {
        // std::cout << "d: " << d << "l: " << l << ":";
        SIZE n_intervals = curr_dim[l + 1].size() / 2;
        std::vector<std::pair<SIZE, SIZE>> curr_level;
        for (SIZE i = 0; i < n_intervals; i++) {
          curr_level.push_back(merge_intervals(curr_dim[l + 1][i * 2], curr_dim[l + 1][i * 2 + 1]));
          // std::cout << "(" << curr_level[i].first << ", " << curr_level[i].second << ") ";
        }
        if (curr_dim[l + 1].size() % 2 != 0) {
          curr_level.push_back(curr_dim[l + 1][curr_dim[l + 1].size() - 1]);
          // std::cout << "(" << curr_level[n_intervals].first << ", " << curr_level[n_intervals].second << ") ";
        }

        // std::cout << "\n";
        curr_dim[l] = curr_level;
      }
      bounding_interval_hierarchy[d] = curr_dim;
    }
    return bounding_interval_hierarchy;
  }


  HierarchicalIndexType buildHierarchicalIndex() {
    HierarchicalIndexType hierarchical_index(D);
    for (DIM d = 0; d < D; d++) {
      std::vector<std::vector<SIZE>> curr_dim(hierarchy.l_target + 1);
      SIZE dof = hierarchy.dofs[d][0];
      std::vector<SIZE> l_bottom(dof);
      // l == hierarchy.l_target
      for (SIZE i = 0; i < dof; i++) {
        l_bottom[i] = i;
      }

      // std::cout << "D " << d << " L " << hierarchy.l_target << ": ";
      // for (SIZE i = 0; i < dof; i++) {
      //   std::cout << l_bottom[i] << " ";
      // }
      // std::cout << "\n";

      curr_dim[hierarchy.l_target] = l_bottom;
      for (int l = hierarchy.l_target - 1; l >= 0; l--) {
        SIZE n = curr_dim[l + 1].size() / 2 + 1;
        std::vector<SIZE> curr_level;
        for (SIZE i = 0; i < curr_dim[l + 1].size() / 2; i++) {
          curr_level.push_back(curr_dim[l + 1][i * 2]);
        }
        curr_level.push_back(curr_dim[l + 1][curr_dim[l + 1].size()-1]);
        curr_dim[l] = curr_level;

        // std::cout << "D " << d << " L " << l << ": ";
        // for (SIZE i = 0; i < n; i++) {
        //   std::cout << curr_level[i] << " ";
        // }
        // std::cout << "\n";
      }
      hierarchical_index[d] = curr_dim;
    }
    return hierarchical_index;
  }

  void buildTree(SubArray<D, T, DeviceType> &coeff) {
    this->coeff = coeff;
    bounding_interval_hierarchy = buildBoundingIntervalHierarchy();
    HierarchicalIndexType hierarchical_index = buildHierarchicalIndex();

    std::vector<SIZE> root_start(D, 0);
    std::vector<SIZE> root_end(D, 0);
    std::vector<SIZE> root_shape(D);
    for (DIM d = 0; d < D; d++) {
      root_end[d] = hierarchy.shapes_vec[0][d] - 1;
      root_shape[d] = hierarchy.shapes_vec[hierarchy.l_target][d];
    }

    coeff_host_array = Array<D, T, SERIAL>(hierarchy.shape_org);
    MemoryManager<DeviceType>().CopyND(
        coeff_host_array.data(), coeff_host_array.ld()[0], coeff.data(),
        coeff.getLd()[0], hierarchy.dofs[0][0],
        (SIZE)(hierarchy.dofs[1][0] * hierarchy.linearized_depth), 0);
    SubArray coeff_host_subarray(coeff_host_array);

    // PrintSubarray("coeff_host_subarray", coeff_host_subarray);

    T root_max_coeff = 0.0;
    SIZE root_coeff_index[D];
    SIZE root_ndof = 1;
    for (DIM d = 0; d < D; d++) root_ndof = root_shape[d];
    for (SIZE i = 0; i < root_ndof; i++) {
      SIZE linearized_dof_index = i;
      for (DIM d = 0; d < D; d++) {
        root_coeff_index[d] = linearized_dof_index % root_shape[d];
        linearized_dof_index /= root_shape[d];
      }
      T coeff = *coeff_host_subarray(root_coeff_index);
      root_max_coeff = std::max(root_max_coeff, coeff);
    }

    root = NodeType(
        &hierarchy, root_start, root_end, 0, coeff_host_subarray,
        bounding_interval_hierarchy, hierarchical_index, root_max_coeff);

    typename NodeType::ErrorImpactType all_children_error_impact = root.initialize();
    all_children_error_impact[0] = root_max_coeff;

    // std::cout << "all_children_error_impact: "; 
    // for (int i = 0; i < all_children_error_impact.size(); i++) {
    //   std::cout << all_children_error_impact[i] << " ";
    // }
    // std::cout << "\n";
  } 

typename NodeType::T_error error_estimate(typename NodeType::ErrorImpactType error_impact) {
  T sum = 0;
  for (int i = 0; i < error_impact.size(); i++) {
    sum += error_impact[i];
  }
  return sum * (1 + std::pow(std::sqrt(3) / 2, D)); 
  // std::cout << "sum:" << sum << " "
  // return sum * (1 + std::pow(3, D)); 
}

typename NodeType::ErrorImpactType combine_error_impact(typename NodeType::ErrorImpactType error_impact1, typename NodeType::ErrorImpactType error_impact2) {
  if (error_impact1.size() != hierarchy.l_target + 1 || 
      error_impact2.size() != hierarchy.l_target + 1) {
    std::cout << log::log_err << "Wrong error_impact input length.\n";
    exit(-1);
  }
  typename NodeType::ErrorImpactType combined_error_impact(hierarchy.l_target + 1);
  for (int i = 0; i < hierarchy.l_target + 1; i++) {
    combined_error_impact[i] = std::max(error_impact1[i], error_impact2[i]);
  }
  return combined_error_impact;
}


void refine_level(SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> w, SIZE curr_level, int queue_idx) {
  SubArray<D, T, DeviceType> v_fine = v;
  SubArray<D, T, DeviceType> w_fine = w;
  SubArray<D, T, DeviceType> v_coeff = v;
  SubArray<D, T, DeviceType> w_correction = w;
  SubArray<D, T, DeviceType> v_coarse = v;
  SIZE l = hierarchy.l_target - 1 - curr_level;

  v_coeff.resize(hierarchy.shapes_vec[l]);
  w_correction.resize(hierarchy.shapes_vec[l]);
  CalcCorrection3D(hierarchy, v_coeff, w_correction, l, queue_idx);

  w_correction.resize(hierarchy.shapes_vec[l + 1]);
  v_coarse.resize(hierarchy.shapes_vec[l + 1]);

  // TO DO: corrections should only be applied to refined children
  SubtractND(w_correction, v_coarse, queue_idx);

  v_coeff.resize(hierarchy.shapes_vec[l]);
  w_fine.resize(hierarchy.shapes_vec[l]);
  CoefficientsRestore3D(hierarchy, v_coeff, w_fine, l, queue_idx);

  v_fine.resize(hierarchy.shapes_vec[l]);
  CopyND(w_fine, v_fine, queue_idx);
}

template <typename FeatureDetectorType>
Array<D, T, DeviceType> constructData(T target_tol, bool interpolate_full_resolution, FeatureDetectorType feature_detector, int queue_idx) {
  using Mem = MemoryManager<DeviceType>;
  std::queue<NodeType*> to_be_refined_node;
  Array<D, T, DeviceType> reconstructed_data(hierarchy.shape_org);
  reconstructed_data.memset(0);
  SubArray v(reconstructed_data);

  Array<D, T, DeviceType> workspace(hierarchy.shape_org);
  SubArray w(workspace);


  // Node in 'to_be_refined_node' are nodes already created and to be future fined.
  // So root node is first inserted and the level 0 is reconstructed first
  to_be_refined_node.push(&root);
  

  

  SubArray root = v;
  SubArray root_org = coeff;
  root.resize(hierarchy.shapes_vec[hierarchy.l_target]);
  root_org.resize(hierarchy.shapes_vec[hierarchy.l_target]);
  CopyND(root_org, root, queue_idx);
  DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

  // PrintSubarray("v", v);
  
  std::ofstream myfile;
  myfile.open("adaptive_resolution_config.txt", std::ios::out);


  std::vector<NodeType *> final_output_cells;
  int reconstructed_cells = 0;
  int curr_level = 0;
  typename NodeType::ErrorImpactType curr_error_impact(hierarchy.l_target+1, 0);
  while (to_be_refined_node.size() > 0) {
    NodeType* node = to_be_refined_node.front();
    to_be_refined_node.pop();    
    std::cout << "Level: " << node->level << " ";

    T combined_error_discard_all = error_estimate(combine_error_impact(curr_error_impact, node->all_children_error_impact));
    bool discard_all_children = combined_error_discard_all < target_tol;
    bool contain_feature = false;
    if (node->potential_contain_feature) {
      contain_feature = feature_detector(node, combined_error_discard_all, v);
    }
    
    if (!contain_feature) {
      for (int c = 0; c < node->num_children; c++) {
        node->child[c].potential_contain_feature = false;
      }
    }
    bool final_cells = contain_feature & discard_all_children;

    myfile << node->index_start[0] << "," << node->index_start[1] << "," 
         << node->index_end[0] << "," << node->index_end[1] << ","
         << combined_error_discard_all << "," << final_cells <<"," << node->potential_contain_feature <<"\n";

    if (final_cells) final_output_cells.push_back(node);
    if (discard_all_children) reconstructed_cells ++;


    std::cout << node->index_start[0] << "," << node->index_start[1] << "," 
         << node->index_end[0] << "," << node->index_end[1] << ","
         << combined_error_discard_all << "," << contain_feature <<"," << node->potential_contain_feature <<"\n";


    //for debug
    // if (feature_detector(node, combined_error_discard_all, v) == true && node->potential_contain_feature == false) {
    //   std::cout << "Error: feature detection mismatch\n";
    //   exit(-1);
    // }    
    
    if (discard_all_children) {
      // Discard all children
      std::cout << "Discard all chilren\n";
      curr_error_impact = combine_error_impact(curr_error_impact, node->all_children_error_impact);
    } else {
      // Create all children
      std::cout << "Create all "<< node->num_children <<" children ";
      int children_num_bitplanes = 0;
      int min_bitplanes = 0;
      if (curr_level < hierarchy.l_target - 1) {
        min_bitplanes = node->num_bitplanes;
      } 
      for (int b = min_bitplanes; b <= node->num_bitplanes; b++) {
        typename NodeType::T_error all_children_max_error = 0;
        for (int c = 0; c < node->num_children; c++) {
          all_children_max_error = std::max(all_children_max_error, node->child[c].max_error[b]);
        }
        typename NodeType::ErrorImpactType temp_error_impact = curr_error_impact;
        // std::cout << all_children_max_error << " ";
        // Get max error for all children of all node of this level
        temp_error_impact[curr_level + 1] = std::max(all_children_max_error, temp_error_impact[curr_level + 1]);
        if (error_estimate(temp_error_impact) < target_tol) {
          children_num_bitplanes = b;
          curr_error_impact = temp_error_impact;
          break;
        }
      }
      std::cout << "by retrieving " << children_num_bitplanes << " bitplanes\n";
    
      for (int c = 0; c < node->num_children; c++) {
        if (node->child_coefficient_index_reordered[c].size() > 0) {
          Array<1, T, DeviceType> child_coefficients_array = 
              node->retrieve_child_coefficient(c, children_num_bitplanes, queue_idx);
          SubArray child_coefficients_subarray(child_coefficients_array);

          for (int j = 0; j < node->child_coefficient_index_reordered[c].size(); j++) {
            SIZE coefficient_index[D];
            for (DIM d = 0; d < D; d++) {
              coefficient_index[d] = node->child_coefficient_index_reordered[c][j][d];
            }
            // Mem::Copy1D(v(coefficient_index), coeff(coefficient_index), 1, queue_idx);
            Mem::Copy1D(v(coefficient_index), child_coefficients_subarray(j), 1, queue_idx);
          }
        }

        NodeType * child = &(node->child[c]);
        // if (child->num_children > 0) {
          // std::cout << "push a child with " << child->num_children << " children at level "<< child->level <<"\n";
          to_be_refined_node.push(&(node->child[c]));
        // }
      }
    }

    // If we are about to proceed to the next level,
    // We need to actaully complete refining this level before proceed.
    if (to_be_refined_node.size() > 0 && to_be_refined_node.front()->level > curr_level ||
        to_be_refined_node.size() == 0 && curr_level < hierarchy.l_target) {

      std::cout << "current error: " << error_estimate(curr_error_impact) << "\n";
      std::cout << "Refining level " << curr_level << "\n";
      refine_level(v, w, curr_level, queue_idx);
      curr_level += 1;
      // PrintSubarray("v", v);
    }
    
  }

  if (interpolate_full_resolution) {
    for (; curr_level < hierarchy.l_target; curr_level++) {
      std::cout << "Interpolate level " << curr_level << "\n";
      refine_level(v, w, curr_level, queue_idx);
      // PrintSubarray("v", v);
    }
  }

  myfile.close();

  int org_cells = 1;
  for (int d = 0; d < D; d++) {
    org_cells *= (hierarchy.shape_org[d] - 1);
  }
  int reduced_cells = final_output_cells.size();
  std::cout << "Original cells: " << org_cells << "\n";
  std::cout << " Redueced cells: " << reconstructed_cells << " Ratio: " << (float)org_cells/reconstructed_cells << "\n";
  std::cout << " Feature cells: " << reduced_cells << " Ratio: " << (float)org_cells/reduced_cells << "\n";

  return reconstructed_data;
}

private:
  std::vector<T> max_coeff_level;
  std::vector<std::vector<T>> max_coeff_partition;
  NodeType root;
  SIZE total_level;
  Hierarchy<D, T, DeviceType> &hierarchy;
  BoundingIntervalHierarchyType bounding_interval_hierarchy;
  Array<D, T, SERIAL> coeff_host_array;
  SubArray<D, T, DeviceType> coeff;
};

} // namespace mgard_x

#endif