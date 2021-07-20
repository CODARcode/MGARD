#ifndef _MDR_MGARD_DECOMPOSER_HPP
#define _MDR_MGARD_DECOMPOSER_HPP

#include "DecomposerInterface.hpp"
#include "mgard.hpp"
#include "shuffle.hpp"

namespace MDR {
    // MGARD decomposer with orthogonal basis
    template<class T>
    class MGARDOrthoganalDecomposer : public concepts::DecomposerInterface<T> {
    public:
        MGARDOrthoganalDecomposer(){}
        void decompose(T * data, const std::vector<uint32_t>& dimensions, uint32_t target_level) const {
            size_t total_size = 1;
            for (int i = 0; i < dimensions.size(); i++) total_size *= dimensions[i];
            T * unshuffled = new T[total_size];
            memcpy(unshuffled, data, total_size*sizeof(T));
            if (dimensions.size() == 3) {
                mgard::TensorMeshHierarchy<3, T> hierarchy({dimensions[0], dimensions[1], dimensions[2]});
                mgard::shuffle(hierarchy, unshuffled, data);
                mgard::decompose(hierarchy, data);
            } else if (dimensions.size() == 2) {
                mgard::TensorMeshHierarchy<2, T> hierarchy({dimensions[0], dimensions[1]});
                mgard::shuffle(hierarchy, unshuffled, data);
                mgard::decompose(hierarchy, data);
            } else if (dimensions.size() == 1) {
                mgard::TensorMeshHierarchy<1, T> hierarchy({dimensions[0]});
                mgard::shuffle(hierarchy, unshuffled, data);
                mgard::decompose(hierarchy, data);
            }
            delete [] unshuffled;
 
        }
        void recompose(T * data, const std::vector<uint32_t>& dimensions, uint32_t target_level) const {
            size_t total_size = 1;
            for (int i = 0; i < dimensions.size(); i++) total_size *= dimensions[i];
            T * shuffled = new T[total_size];
            memcpy(shuffled, data, total_size*sizeof(T));
            if (dimensions.size() == 3) {
                mgard::TensorMeshHierarchy<3, T> hierarchy({dimensions[0], dimensions[1], dimensions[2]});
                mgard::recompose(hierarchy, shuffled);
                mgard::unshuffle(hierarchy, shuffled, data);
            } else if (dimensions.size() == 2) {
                mgard::TensorMeshHierarchy<2, T> hierarchy({dimensions[0], dimensions[1]});
                mgard::recompose(hierarchy, shuffled);
                mgard::unshuffle(hierarchy, shuffled, data);
            } else if (dimensions.size() == 1) {
                mgard::TensorMeshHierarchy<1, T> hierarchy({dimensions[0]});
                mgard::recompose(hierarchy, shuffled);
                mgard::unshuffle(hierarchy, shuffled, data);   
            }
            delete [] shuffled;
        }
        void print() const {
            std::cout << "MGARD orthogonal decomposer" << std::endl;
        }
    };
}
#endif
