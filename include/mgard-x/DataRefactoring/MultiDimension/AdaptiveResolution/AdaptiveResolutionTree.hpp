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

#ifndef MGARD_X_ADAPTIVE_RESOLUTION_TREE
#define MGARD_X_ADAPTIVE_RESOLUTION_TREE

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
class AdaptiveResolutionTreeNode {

  using BoundingIntervalHierarchyType =
      std::vector<std::vector<std::vector<std::pair<SIZE, SIZE>>>>;
  using HierarchicalIndexType = std::vector<std::vector<std::vector<SIZE>>>;

public:
  AdaptiveResolutionTreeNode() {}
  AdaptiveResolutionTreeNode(Hierarchy<D, T, DeviceType> *hierarchy, std::vector<SIZE> index_start,
                             std::vector<SIZE> index_end, SIZE level, SubArray<D, T, SERIAL> coeff,
                             BoundingIntervalHierarchyType bounding_interval_hierarchy,
                             HierarchicalIndexType hierarchical_index, T max_coefficient)
                            : hierarchy(hierarchy), index_start(index_start), index_end(index_end),
                              level(level), coeff(coeff),
                              bounding_interval_hierarchy(bounding_interval_hierarchy), hierarchical_index(hierarchical_index), max_coefficient(max_coefficient){
    std::vector<SIZE> cell_shape(D, 2);
    data = Array<D, T, SERIAL>(cell_shape);
  }
  // metadata
  Hierarchy<D, T, DeviceType> *hierarchy = NULL;
  bool is_leaf = true;
  std::vector<SIZE> index_start; // inclusive
  std::vector<SIZE> index_end;   // inclusive
  std::vector<T> coord_start;  // inclusive
  std::vector<T> coord_end;    // inclusive
  SIZE level;
  SIZE max_level;
  SubArray<D, T, SERIAL> coeff;
  bool contain_ghost_cell[D];
  AdaptiveResolutionTreeNode<D, T, DeviceType> *child = NULL;
  SIZE num_children;
  BoundingIntervalHierarchyType bounding_interval_hierarchy;
  HierarchicalIndexType hierarchical_index;

  // Max coefficient of mine
  T max_coefficient = 0.0;

  // Max coefficient amoung all my children
  T all_children_max_coefficient = 0.0;

  // Error impact if discarding all children
  std::vector<T> all_children_error_impact;

  // Error impact if discard some children
  std::vector<std::vector<T>> child_error_impact;

  // Coefficent index of all my children
  std::vector<std::vector<SIZE>> child_coefficient_index;

  // Coefficent index of all my children (reordered)
  std::vector<std::vector<SIZE>> child_coefficient_index_reordered;

  Array<D, T, SERIAL> data;

  std::vector<T> combine_error_impact(std::vector<T> error_impact1, std::vector<T> error_impact2) {
    if (error_impact1.size() != hierarchy->l_target + 1 || 
        error_impact2.size() != hierarchy->l_target + 1) {
      std::cout << log::log_err << "Wrong error_impact input length.\n";
      exit(-1);
    }
    std::vector<T> combined_error_impact(hierarchy->l_target + 1);
    for (int i = 0; i < hierarchy->l_target + 1; i++) {
      combined_error_impact[i] = std::max(error_impact1[i], error_impact2[i]);
    }
    return combined_error_impact;
  }

  std::vector<T> initialize() {

    std::cout << "level " << level << " curr ";
    std::cout << " start: ";
    for (int d = D - 1; d >= 0; d--) std::cout << index_start[d] << " ";
    std::cout << "end: ";
    for (int d = D - 1; d >= 0; d--) std::cout << index_end[d] << " ";
    std::cout << "\n";

    SIZE next_level = level + 1;

    if (next_level >= hierarchy->l_target + 1) {
      all_children_error_impact = std::vector<T>(hierarchy->l_target + 1, 0);
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

    std::cout << "num_children: " << num_children << "\n";

    // Allocate memory for child
    child = new AdaptiveResolutionTreeNode<D, T, DeviceType>[num_children];

    // Error impact
    all_children_error_impact = std::vector<T>(hierarchy->l_target + 1, 0);
    child_error_impact = std::vector<std::vector<T>>(num_children);

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

    std::cout << "coefficient_marker: \n";
    for (DIM d = 0; d < D; d++) {
      std::cout << "D: " << d << ": ";
      for (int i = 0; i < coefficient_marker[d].size(); i++) {
        std::cout << coefficient_marker[d][i] << " ";
      }
      std::cout << "\n";
    }

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

      std::cout << "child " << r << " ";
      std::cout << " start: ";
      for (int d = D - 1; d >= 0; d--) std::cout << child_index_start[d] << " ";
      std::cout << "end: ";
      for (int d = D - 1; d >= 0; d--) std::cout << child_index_end[d] << " ";
      std::cout << "\n";

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
          child_coefficient_index.push_back(coefficient_index);
        }
      }

      // std::cout << "child_coefficient_index: ";
      // for (SIZE i = 0; i < child_coefficient_index.size(); i++) {
      //   std::cout << "(" << child_coefficient_index[i][1] << ", " << child_coefficient_index[i][0] << ") ";
      // }
      // std::cout << "\n";

      
      // Fetch all coefficients of the child
      std::vector<T> child_coefficients;
      for (SIZE i = 0; i < child_coefficient_index.size(); i++) {
        SIZE coefficient_index_reordered[D];
        std::vector<SIZE> coefficient_index_reordered_vec(D);
        // level index
        SIZE stride = std::pow(2, hierarchy->l_target - level);
        for (DIM d = 0; d < D; d++) {
          bool along_coefficient_dim = false;
          for (int m = 0; m < coefficient_marker[d].size(); m++) {
            if (child_coefficient_index[i][d] == coefficient_marker[d][m]) {
              along_coefficient_dim = true;
              break;
            }
          }
          if (along_coefficient_dim) {
            coefficient_index_reordered[d] = hierarchy->dofs[d][hierarchy->l_target - level];
            coefficient_index_reordered[d] += (child_coefficient_index[i][d] - stride / 2) / stride;
          } else {
            coefficient_index_reordered[d] = std::ceil((float)child_coefficient_index[i][d] / stride);
          }
          coefficient_index_reordered_vec[d] = coefficient_index_reordered[d];
        }


        T coefficient = *coeff(coefficient_index_reordered);
        
        child_coefficients.push_back(coefficient);
        child_coefficient_index_reordered.push_back(coefficient_index_reordered_vec);
      }

      T child_max_coefficient = 0;
      for (SIZE i = 0; i < child_coefficients.size(); i++) {
        child_max_coefficient = std::max(child_max_coefficient, abs(child_coefficients[i]));
      }


      std::cout << "child_coefficient: ";
      for (SIZE i = 0; i < child_coefficients.size(); i++) {
        std::cout << child_coefficients[i] << ", ";
      }
      std::cout << "\n";

      // std::cout << "level " << level << " partition " << r << " child_max_coefficient " << child_max_coefficient << "\n";


      child[r] = AdaptiveResolutionTreeNode<D, T, DeviceType>(
          hierarchy, child_index_start, child_index_end, next_level,
          coeff, bounding_interval_hierarchy, hierarchical_index, child_max_coefficient);

      all_children_max_coefficient = std::max(all_children_max_coefficient, child_max_coefficient);

    }


    for (SIZE r = 0; r < num_children; r++) {
      child_error_impact[r] = child[r].initialize();
    }

    // Gather error impact from children
    for (SIZE r = 0; r < num_children; r++) {
      all_children_error_impact = combine_error_impact(all_children_error_impact, child_error_impact[r]);
      child_error_impact[r][next_level] = child[r].max_coefficient;
    }
    all_children_error_impact[next_level] = all_children_max_coefficient;

    return all_children_error_impact;
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
      std::cout << "l_bottom " << d << ": ";
      for (SIZE i = 0; i < dof - 1; i++) {
        l_bottom[i] = std::pair<SIZE, SIZE>(i, i + 1);
        std::cout << "(" << l_bottom[i].first << ", " << l_bottom[i].second << ") ";
      }
      std::cout << "\n";

      curr_dim[hierarchy.l_target] = l_bottom;
      for (int l = hierarchy.l_target - 1; l >= 0; l--) {
        std::cout << "d: " << d << "l: " << l << ":";
        SIZE n_intervals = curr_dim[l + 1].size() / 2;
        std::vector<std::pair<SIZE, SIZE>> curr_level;
        for (SIZE i = 0; i < n_intervals; i++) {
          curr_level.push_back(merge_intervals(curr_dim[l + 1][i * 2], curr_dim[l + 1][i * 2 + 1]));
          std::cout << "(" << curr_level[i].first << ", " << curr_level[i].second << ") ";
        }
        if (curr_dim[l + 1].size() % 2 != 0) {
          curr_level.push_back(curr_dim[l + 1][curr_dim[l + 1].size() - 1]);
          std::cout << "(" << curr_level[n_intervals].first << ", " << curr_level[n_intervals].second << ") ";
        }

        std::cout << "\n";
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

      std::cout << "D " << d << " L " << hierarchy.l_target << ": ";
      for (SIZE i = 0; i < dof; i++) {
        std::cout << l_bottom[i] << " ";
      }
      std::cout << "\n";

      curr_dim[hierarchy.l_target] = l_bottom;
      for (int l = hierarchy.l_target - 1; l >= 0; l--) {
        SIZE n = curr_dim[l + 1].size() / 2 + 1;
        std::vector<SIZE> curr_level;
        for (SIZE i = 0; i < curr_dim[l + 1].size() / 2; i++) {
          curr_level.push_back(curr_dim[l + 1][i * 2]);
        }
        curr_level.push_back(curr_dim[l + 1][curr_dim[l + 1].size()-1]);
        curr_dim[l] = curr_level;

        std::cout << "D " << d << " L " << l << ": ";
        for (SIZE i = 0; i < n; i++) {
          std::cout << curr_level[i] << " ";
        }
        std::cout << "\n";
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

    PrintSubarray("coeff_host_subarray", coeff_host_subarray);

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

    std::vector<T> all_children_error_impact = root.initialize();
    all_children_error_impact[0] = root_max_coeff;

    // std::cout << "all_children_error_impact: "; 
    // for (int i = 0; i < all_children_error_impact.size(); i++) {
    //   std::cout << all_children_error_impact[i] << " ";
    // }
    // std::cout << "\n";
  } 

T error_estimate(std::vector<T> error_impact) {
  T sum = 0;
  for (int i = 0; i < error_impact.size(); i++) {
    sum += error_impact[i];
  }
  return sum / (1 + std::pow(std::sqrt(3) / 2, D)); 
}

std::vector<T> combine_error_impact(std::vector<T> error_impact1, std::vector<T> error_impact2) {
    if (error_impact1.size() != hierarchy.l_target + 1 || 
        error_impact2.size() != hierarchy.l_target + 1) {
      std::cout << log::log_err << "Wrong error_impact input length.\n";
      exit(-1);
    }
    std::vector<T> combined_error_impact(hierarchy.l_target + 1);
    for (int i = 0; i < hierarchy.l_target + 1; i++) {
      combined_error_impact[i] = std::max(error_impact1[i], error_impact2[i]);
    }
    return combined_error_impact;
  }

void constructData(T target_tol) {
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
  // LwpkReo<D, T, COPY, DeviceType>().Execute(root_org, root, 0);
  CopyND(root_org, root, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  PrintSubarray("v", v);
  
  int curr_level = 0;
  std::vector<T> curr_error_impact(hierarchy.l_target+1, 0);
  while (to_be_refined_node.size() > 0) {
    NodeType* node = to_be_refined_node.front();
    to_be_refined_node.pop();    

    // T error_discard_all = error_estimate(node->all_children_error_impact);
    // std::cout << "error_discard_all: " << error_discard_all << "\n";
    std::cout << "Level: " << node->level << " ";
    T combined_error_discard_all = error_estimate(combine_error_impact(curr_error_impact, node->all_children_error_impact));
    if (combined_error_discard_all < target_tol) {
      // Discard all children
      std::cout << "Discard all chilren\n";
      curr_error_impact = combine_error_impact(curr_error_impact, node->all_children_error_impact);
    } else {
      // Create all children
      std::cout << "Create all children\n";
      for (int i = 0; i < node->num_children; i++) {
        for (int j = 0; j < node->child_coefficient_index_reordered.size(); j++) {
          SIZE coefficient_index[D];
          for (DIM d = 0; d < D; d++) {
            coefficient_index[d] = node->child_coefficient_index_reordered[j][d];
          }
          Mem::Copy1D(v(coefficient_index), coeff(coefficient_index), 1, 0);
        }

        NodeType * child = &(node->child[i]);
        if (child->num_children > 0) {
          to_be_refined_node.push(&(node->child[i]));
        }
      }
    }

    // If we are about to proceed to the next level,
    // We need to actaully complete refining this level before proceed.
    if (to_be_refined_node.size() > 0 && to_be_refined_node.front()->level > curr_level) {
      SubArray<D, T, DeviceType> v_coeff = v;
      SubArray<D, T, DeviceType> w_correction = w;
      SubArray<D, T, DeviceType> v_coarse = v;
      SIZE l = hierarchy.l_target - 1 - curr_level;
      v_coeff.resize(hierarchy.shapes_vec[l]);
      w_correction.resize(hierarchy.shapes_vec[l]);
      // calc_correction_3d(hierarchy, v_coeff, w_correction, l, 0);
      CalcCorrection3D(hierarchy, v_coeff, w_correction, l, 0);
      w_correction.resize(hierarchy.shapes_vec[l + 1]);
      v_coarse.resize(hierarchy.shapes_vec[l + 1]);
      // LwpkReo<D, T, SUBTRACT, DeviceType>().Execute(w_correction, v_coarse, 0);
      SubtractND(w_correction, v_coarse, 0);
      curr_level += 1;
    }

    std::cout << "current error: " << error_estimate(curr_error_impact) << "\n";
    PrintSubarray("v", v);
  }
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