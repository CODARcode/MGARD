/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_COMPRESSED_SPARSE_EDGE
#define MGARD_X_COMPRESSED_SPARSE_EDGE

#include "Edge.hpp"

namespace mgard_x {

#define NO_NEIGHBOR 0
// shared-face neighbor
#define pX 1
#define nX 2
#define pY 4
#define nY 8
#define pZ 16
#define nZ 32
// shared-edge neighbor
#define pXpY 64
#define nXpY 128
#define pXnY 256
#define nXnY 512  
#define pXpZ 1024
#define nXpZ 2048
#define pXnZ 4096
#define nXnZ 8192 
#define pYpZ 16384
#define nYpZ 32768
#define pYnZ 65536
#define nYnZ 131072

template <DIM D, typename T, typename DeviceType>
class CompressedSparseEdge {
public:
  using EdgeType = Edge<D, T, DeviceType>;
  CompressedSparseEdge() {}
  CompressedSparseEdge(std::vector<SIZE> data_shape, std::vector<EdgeType> edge_list,
                       std::vector<std::vector<SIZE>> level_index_vec){
    // std::cout << "shape: ";
    this->shape[2] = 1; 
    for (DIM d = 0; d < D; d++) {
      if (d == 1) {
        this->shape[d] = data_shape[d] - 1;
      } else {
        this->shape[d] = data_shape[d];
      }
      // std::cout << this->shape[d] << ", ";
    }

    // std::cout << "\n";
    
    empty = edge_list.size() == 0;
    if (empty) return;

    std::cout << "create vec\n";
    std::vector<std::vector<T>> start_value_vec(shape[2]*shape[1]);
    std::vector<std::vector<T>> end_value_vec(shape[2]*shape[1]);
    std::vector<std::vector<SIZE>> index_vec(shape[2]*shape[1]);
    std::vector<std::vector<SIZE>> pY_index_vec(shape[2]*shape[1]);
    std::vector<std::vector<SIZE>> nY_index_vec(shape[2]*shape[1]);
    std::vector<std::vector<SIZE>> pZ_index_vec(shape[2]*shape[1]);
    std::vector<std::vector<SIZE>> nZ_index_vec(shape[2]*shape[1]);
    std::vector<std::vector<SIZE>> neighbor_vec(shape[2]*shape[1]);
    std::vector<std::vector<SIZE>> cell_ids_vec(shape[2]*shape[1]);
    std::vector<std::vector<SIZE>> role_vec(shape[2]*shape[1]);

    std::vector<std::vector<std::vector<SIZE>>> sparsity_map(shape[2]);
    for (int i = 0; i < shape[2]; i++) {
      sparsity_map[i] = std::vector<std::vector<SIZE>>(shape[1]);
      for (int j = 0; j < shape[1]; j++) {
        sparsity_map[i][j] = std::vector<SIZE>(shape[0], NONE);
      }
    }

    std::cout << "add edge :" << edge_list.size() << "\n";    
    for (int i = 0; i < edge_list.size(); i++) {
      SIZE x_index = 0, y_index = 0, z_index = 0;

      for (x_index = 0; x_index < level_index_vec[0].size(); x_index++) {
        if (level_index_vec[0][x_index] == edge_list[i].start_index[0]) {
          break;
        }
      }
      for (y_index = 0; y_index < level_index_vec[1].size(); y_index++) {
        if (level_index_vec[1][y_index] == edge_list[i].start_index[1]) {
          break;
        }
      }

      if (D == 3) {
        for (z_index = 0; z_index < level_index_vec[2].size(); z_index++) {
          if (level_index_vec[2][z_index] == edge_list[i].start_index[2]) {
            break;
          }
        }
      }

      int current_role = sparsity_map[z_index][y_index][x_index];
      if (current_role != LEAD) {
        sparsity_map[z_index][y_index][x_index] = edge_list[i].role;
      }
        



      // std::cout << "index: " << x_index << ", "<< y_index << ", " << z_index << "\n"; 

      int yz = z_index * shape[1] + y_index;

      // std::cout << "shape[2]*shape[1]: " << shape[2]*shape[1] << " yz: " << yz << "\n";
      SIZE exist = false;
      int j = 0;
      for (j = 0; j < index_vec[yz].size(); j++) {
        if (index_vec[yz][j] == x_index) {
          exist = true;
          break;
        }
      }
      // std::cout << "exist: " << exist <<"\n";
      if (!exist) {
        start_value_vec[yz].push_back(edge_list[i].start_value);
        end_value_vec[yz].push_back(edge_list[i].end_value);
        index_vec[yz].push_back(x_index);
        if (edge_list[i].role == LEAD) {
          role_vec[yz].push_back(LEAD);
        } else {
          role_vec[yz].push_back(REGULAR);
        }
      } 

      // Replacing REGULAR edge with LEAD edge
      if (exist && role_vec[yz][j] == REGULAR && edge_list[i].role == LEAD) {
        role_vec[yz][j] = LEAD;
        // std::cout << "replace with LEAD\n";
      }
    }


    std::cout << "Find neighbor\n";
    for (SIZE z_index = 0; z_index < shape[2]; z_index++) {
      for (SIZE y_index = 0; y_index < shape[1]; y_index++) {
        int yz = z_index * shape[1] + y_index;
        neighbor_vec[yz] = std::vector<SIZE>(role_vec[yz].size(), NO_NEIGHBOR);
        for (SIZE i = 0; i < role_vec[yz].size(); i++) {
          if (role_vec[yz][i] == LEAD) {
            SIZE x_index = index_vec[yz][i];
            if (sparsity_map[z_index][y_index][x_index] != LEAD) {
              std::cout << log::log_err << "error in finding neighbor\n";
              exit(-1);
            }
            if (x_index < shape[0]-1 && sparsity_map[z_index][y_index][x_index+1] == LEAD) {
              neighbor_vec[yz][i] += pX;
            }
            if (x_index >= 1 && sparsity_map[z_index][y_index][x_index-1] == LEAD) {
              neighbor_vec[yz][i] += nX;
            }
            if (y_index < shape[1]-1 && sparsity_map[z_index][y_index+1][x_index] == LEAD) {
              neighbor_vec[yz][i] += pY;
            }
            if (y_index >= 1 && sparsity_map[z_index][y_index-1][x_index] == LEAD) {
              neighbor_vec[yz][i] += nY;
            }
            if (z_index < shape[2]-1 && sparsity_map[z_index+1][y_index][x_index] == LEAD) {
              neighbor_vec[yz][i] += pZ;
            }
            if (z_index >= 1 && sparsity_map[z_index-1][y_index][x_index] == LEAD) {
              neighbor_vec[yz][i] += nZ;
            }

            if (x_index < shape[0]-1 && y_index < shape[1]-1 && sparsity_map[z_index][y_index+1][x_index+1] == LEAD) {
              neighbor_vec[yz][i] += pXpY;
            }
            if (x_index >= 1 && y_index < shape[1]-1 && sparsity_map[z_index][y_index+1][x_index-1] == LEAD) {
              neighbor_vec[yz][i] += nXpY;
            }
            if (x_index < shape[0]-1 && y_index >= 1 && sparsity_map[z_index][y_index-1][x_index+1] == LEAD) {
              neighbor_vec[yz][i] += pXnY;
            }
            if (x_index >= 1 && y_index >= 1 && sparsity_map[z_index][y_index-1][x_index-1] == LEAD) {
              neighbor_vec[yz][i] += nXnY;
            }

            if (x_index < shape[0]-1 && z_index < shape[2]-1 && sparsity_map[z_index+1][y_index][x_index+1] == LEAD) {
              neighbor_vec[yz][i] += pXpZ;
            }
            if (x_index >= 1 && z_index < shape[2]-1 && sparsity_map[z_index+1][y_index][x_index-1] == LEAD) {
              neighbor_vec[yz][i] += nXpZ;
            }
            if (x_index < shape[0]-1 && z_index >= 1 && sparsity_map[z_index-1][y_index][x_index+1] == LEAD) {
              neighbor_vec[yz][i] += pXnZ;
            }
            if (x_index >= 1 && z_index >= 1 && sparsity_map[z_index-1][y_index][x_index-1] == LEAD) {
              neighbor_vec[yz][i] += nXnZ;
            }

            if (y_index < shape[1]-1 && z_index < shape[2]-1 && sparsity_map[z_index+1][y_index+1][x_index] == LEAD) {
              neighbor_vec[yz][i] += pYpZ;
            }
            if (y_index >= 1 && z_index < shape[2]-1 && sparsity_map[z_index+1][y_index-1][x_index] == LEAD) {
              neighbor_vec[yz][i] += nYpZ;
            }
            if (y_index < shape[1]-1 && z_index >= 1 && sparsity_map[z_index-1][y_index+1][x_index] == LEAD) {
              neighbor_vec[yz][i] += pYnZ;
            }
            if (y_index >= 1 && z_index >= 1 && sparsity_map[z_index-1][y_index-1][x_index] == LEAD) {
              neighbor_vec[yz][i] += nYnZ;
            }

          }
        }
      }
    }


    std::cout << "Create index for fast neighbor search\n";
    for (SIZE z_index = 0; z_index < shape[2]; z_index++) {
      for (SIZE y_index = 0; y_index < shape[1]; y_index++) {
        int yz = z_index * shape[1] + y_index;
        pY_index_vec[yz] = std::vector<SIZE>(role_vec[yz].size());
        nY_index_vec[yz] = std::vector<SIZE>(role_vec[yz].size());
        pZ_index_vec[yz] = std::vector<SIZE>(role_vec[yz].size());
        nZ_index_vec[yz] = std::vector<SIZE>(role_vec[yz].size());
        for (SIZE i = 0; i < role_vec[yz].size(); i++) {
          if (role_vec[yz][i] == LEAD) {
            if (y_index < shape[1] - 1) { // Find pY index
              int yz_forward = (z_index) * shape[1] + y_index + 1;
              for (SIZE j = 0; j < index_vec[yz_forward].size(); j++) {
                if (index_vec[yz_forward][j] == index_vec[yz][i]) {
                  pY_index_vec[yz][i] = j;
                }
              }
            } else {
               pY_index_vec[yz][i] = i;
            }
            if (y_index >= 1) { // Find nY index
              int yz_backward = (z_index) * shape[1] + y_index - 1;
              for (SIZE j = 0; j < index_vec[yz_backward].size(); j++) {
                if (index_vec[yz_backward][j] == index_vec[yz][i]) {
                  nY_index_vec[yz][i] = j;
                }
              }
            } else {
               pY_index_vec[yz][i] = i;
            }
            if (z_index < shape[2] - 1) { // Find pZ index
              int yz_below = (z_index + 1) * shape[1] + y_index;
              for (SIZE j = 0; j < index_vec[yz_below].size(); j++) {
                if (index_vec[yz_below][j] == index_vec[yz][i]) {
                  pZ_index_vec[yz][i] = j;
                }
              }
            }
            if (z_index >= 1) { // Find pZ index
              int yz_above = (z_index - 1) * shape[1] + y_index;
              for (SIZE j = 0; j < index_vec[yz_above].size(); j++) {
                if (index_vec[yz_above][j] == index_vec[yz][i]) {
                  nZ_index_vec[yz][i] = j;
                }
              }
            }
          }
        }
      }
    }

    std::cout << "Find Cell id\n";
    for (SIZE i = 0; i < shape[2] * shape[1]; i++) {
      cell_ids_vec[i] = std::vector<SIZE>(role_vec[i].size());
      for (SIZE j = 0; j < role_vec[i].size(); j++) {
        if (role_vec[i][j] == LEAD) {
          cell_ids_vec[i][j] = cell_count;
          // std::cout << "cell id: " << cell_count << "\n";
          cell_count++;
        }
      }

    }

    start_value_array = new Array<1, T, DeviceType> [shape[2]*shape[1]];
    end_value_array = new Array<1, T, DeviceType> [shape[2]*shape[1]];
    index_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    pY_index_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    nY_index_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    pZ_index_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    nZ_index_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    neighbor_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    cell_ids_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    role_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    level_index_array = new Array<1, SIZE, DeviceType> [D];

    start_value = new SubArray<1, T, DeviceType> [shape[2]*shape[1]];
    end_value = new SubArray<1, T, DeviceType> [shape[2]*shape[1]];
    index = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    pY_index = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    nY_index = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    pZ_index = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    nZ_index = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    neighbor = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    cell_ids = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    role = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    level_index = new SubArray<1, SIZE, DeviceType> [D];

    for (SIZE i = 0; i < shape[2] * shape[1]; i++) {
      start_value_array[i] = Array<1, T, DeviceType>({(SIZE)start_value_vec[i].size()});
      end_value_array[i]   = Array<1, T, DeviceType>({(SIZE)end_value_vec[i].size()});
      index_array[i]       = Array<1, SIZE, DeviceType>({(SIZE)index_vec[i].size()});
      pY_index_array[i]    = Array<1, SIZE, DeviceType>({(SIZE)pY_index_vec[i].size()});
      nY_index_array[i]    = Array<1, SIZE, DeviceType>({(SIZE)nY_index_vec[i].size()});
      pZ_index_array[i]    = Array<1, SIZE, DeviceType>({(SIZE)pZ_index_vec[i].size()});
      nZ_index_array[i]    = Array<1, SIZE, DeviceType>({(SIZE)nZ_index_vec[i].size()});
      neighbor_array[i]    = Array<1, SIZE, DeviceType>({(SIZE)neighbor_vec[i].size()});
      cell_ids_array[i]     = Array<1, SIZE, DeviceType>({(SIZE)cell_ids_vec[i].size()});
      role_array[i]        = Array<1, SIZE, DeviceType>({(SIZE)role_vec[i].size()});

      start_value_array[i].load(start_value_vec[i].data());
      end_value_array[i].load(end_value_vec[i].data());
      index_array[i].load(index_vec[i].data());
      pY_index_array[i].load(pY_index_vec[i].data());
      nY_index_array[i].load(nY_index_vec[i].data());
      pZ_index_array[i].load(pZ_index_vec[i].data());
      nZ_index_array[i].load(nZ_index_vec[i].data());
      neighbor_array[i].load(neighbor_vec[i].data());
      cell_ids_array[i].load(cell_ids_vec[i].data());
      role_array[i].load(role_vec[i].data());

      start_value[i] = SubArray(start_value_array[i]);
      end_value[i] = SubArray(end_value_array[i]);
      index[i] = SubArray(index_array[i]);
      pY_index[i] = SubArray(pY_index_array[i]);
      nY_index[i] = SubArray(nY_index_array[i]);
      pZ_index[i] = SubArray(pZ_index_array[i]);
      nZ_index[i] = SubArray(nZ_index_array[i]);
      neighbor[i] = SubArray(neighbor_array[i]);
      cell_ids[i] = SubArray(cell_ids_array[i]);
      role[i] = SubArray(role_array[i]);

      // printf("y_neighbor_array size %u data = %llu\n", (SIZE)y_neighbor_vec[i].size(), y_neighbor_array[i].data());

    }

    for (SIZE d = 0; d < D; d++) {
      level_index_array[d] = Array<1, SIZE, DeviceType>({(SIZE)level_index_vec[d].size()});
      level_index_array[d].load(level_index_vec[d].data());
      level_index[d] = SubArray(level_index_array[d]);
      // std::cout << "cse::level_index_vec " << d << " " << level_index_vec[d].size() << "\n";
    }

    // std::cout << "Done contruction\n";
  }

  CompressedSparseEdge(const CompressedSparseEdge<D, T, DeviceType> &cse) {
    for (DIM d = 0; d < 3; d++) {
      shape[d] = cse.shape[d];
    }
    empty = cse.empty;
    if (empty) return;

    start_value_array = new Array<1, T, DeviceType> [shape[2]*shape[1]];
    end_value_array = new Array<1, T, DeviceType> [shape[2]*shape[1]];
    index_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    role_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    pY_index_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    nY_index_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    pZ_index_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    nZ_index_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    neighbor_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    cell_ids_array = new Array<1, SIZE, DeviceType> [shape[2]*shape[1]];
    level_index_array = new Array<1, SIZE, DeviceType> [D];

    start_value = new SubArray<1, T, DeviceType> [shape[2]*shape[1]];
    end_value = new SubArray<1, T, DeviceType> [shape[2]*shape[1]];
    index = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    role = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    pY_index = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    nY_index = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    pZ_index = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    nZ_index = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    neighbor = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    cell_ids = new SubArray<1, SIZE, DeviceType> [shape[2]*shape[1]];
    level_index = new SubArray<1, SIZE, DeviceType> [D];

    for (SIZE i = 0; i < shape[2] * shape[1]; i++) {
      start_value_array[i] = cse.start_value_array[i];
      end_value_array[i] = cse.end_value_array[i];
      index_array[i] = cse.index_array[i];
      pY_index_array[i] = cse.pY_index_array[i];
      nY_index_array[i] = cse.nY_index_array[i];
      pZ_index_array[i] = cse.pZ_index_array[i];
      nZ_index_array[i] = cse.nZ_index_array[i];
      neighbor_array[i] = cse.neighbor_array[i];
      cell_ids_array[i] = cse.cell_ids_array[i];
      role_array[i] = cse.role_array[i];

      start_value[i] = SubArray(start_value_array[i]);
      end_value[i] = SubArray(end_value_array[i]);
      index[i] = SubArray(index_array[i]);
      pY_index[i] = SubArray(pY_index_array[i]);
      nY_index[i] = SubArray(nY_index_array[i]);
      pZ_index[i] = SubArray(pZ_index_array[i]);
      nZ_index[i] = SubArray(nZ_index_array[i]);
      neighbor[i] = SubArray(neighbor_array[i]);
      cell_ids[i] = SubArray(cell_ids_array[i]);
      role[i] = SubArray(role_array[i]);
    }

    for (SIZE d = 0; d < D; d++) {
      level_index_array[d] = cse.level_index_array[d];
      level_index[d] = SubArray(level_index_array[d]);
      // std::cout << "cse::level_index_vec " << d << " " << level_index[d].getShape(0) << "\n";
    }

    cell_count = cse.cell_count;

  }

  ~CompressedSparseEdge() {
    if (!empty) {
      delete [] start_value_array;
      delete [] end_value_array;
      delete [] index_array;
      delete [] role_array;
      delete [] pY_index_array;
      delete [] nY_index_array;
      delete [] pZ_index_array;
      delete [] nZ_index_array;
      delete [] neighbor_array;
      delete [] cell_ids_array;
      delete [] level_index_array;

      delete [] start_value;
      delete [] end_value;
      delete [] index;
      delete [] role;
      delete [] pY_index;
      delete [] nY_index;
      delete [] pZ_index;
      delete [] nZ_index;
      delete [] neighbor;
      delete [] cell_ids;
      delete [] level_index;
    }
  }
  
  bool empty = true;
  SIZE shape[3];
  SIZE cell_count = 0;
  Array<1, T, DeviceType> * start_value_array = NULL;
  Array<1, T, DeviceType> * end_value_array = NULL;
  Array<1, SIZE, DeviceType> * index_array = NULL;
  Array<1, SIZE, DeviceType> * pY_index_array = NULL;
  Array<1, SIZE, DeviceType> * nY_index_array = NULL;
  Array<1, SIZE, DeviceType> * pZ_index_array = NULL;
  Array<1, SIZE, DeviceType> * nZ_index_array = NULL;
  Array<1, SIZE, DeviceType> * neighbor_array = NULL;
  Array<1, SIZE, DeviceType> * cell_ids_array = NULL;
  Array<1, SIZE, DeviceType> * level_index_array = NULL;
  Array<1, SIZE, DeviceType> * role_array = NULL;

  SubArray<1, T, DeviceType> * start_value = NULL;
  SubArray<1, T, DeviceType> * end_value = NULL;
  SubArray<1, SIZE, DeviceType> * index = NULL;
  SubArray<1, SIZE, DeviceType> * pY_index = NULL;
  SubArray<1, SIZE, DeviceType> * nY_index = NULL;
  SubArray<1, SIZE, DeviceType> * pZ_index = NULL;
  SubArray<1, SIZE, DeviceType> * nZ_index = NULL;
  SubArray<1, SIZE, DeviceType> * neighbor = NULL;
  SubArray<1, SIZE, DeviceType> * cell_ids = NULL;
  SubArray<1, SIZE, DeviceType> * level_index = NULL;
  SubArray<1, SIZE, DeviceType> * role = NULL;
};

}

#endif