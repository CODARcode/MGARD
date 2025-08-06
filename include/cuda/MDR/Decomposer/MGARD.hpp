#ifndef _MDR_MGARD_DECOMPOSER_HPP
#define _MDR_MGARD_DECOMPOSER_HPP

#include "DecomposerInterface.hpp"
#include "decompose.hpp"
#include "recompose.hpp"
#include "../../DataRefactoring.h"
#include <cstring>
namespace mgard_cuda {
namespace MDR {
    // MGARD decomposer with orthogonal basis
    template<DIM D, typename T>
    class MGARDOrthoganalDecomposer : public concepts::DecomposerInterface<D, T> {
    public:
        MGARDOrthoganalDecomposer(Handle<D, T> &handle): _handle(handle){}
        void decompose(T * data, const std::vector<uint32_t>& dimensions, uint32_t target_level) const {
            MGARD::Decomposer<T> decomposer;
            std::vector<size_t> dims(dimensions.size());
            for(int i=0; i<dims.size(); i++){
                dims[i] = dimensions[i];
            }
            decomposer.decompose(data, dims, target_level);

            // size_t size = 1;
            // std::vector<mgard_cuda::SIZE> shape(D);
            // for(mgard_cuda::DIM i = 0; i<D; i++){
            //     shape[i] = dimensions[i];
            //     size *= dimensions[i];
            // }
            // mgard_cuda::Handle<D, T> handle(shape);
            // mgard_cuda::Array<D, T> array(shape);
            // array.loadData((const T*)data);
            // handle.allocate_workspace();
            // mgard_cuda::decompose<D, T>(handle, array.get_dv(), array.get_ldvs_h(), array.get_ldvs_d(),
            // target_level, 0);
            // handle.sync_all();
            // handle.free_workspace();
            // std::memcpy(data, array.getDataHost(), size*sizeof(T)); 
        }
        void recompose(T * data, const std::vector<uint32_t>& dimensions, uint32_t target_level) const {
            MGARD::Recomposer<T> recomposer;
            std::vector<size_t> dims(dimensions.size());
            for(int i=0; i<dims.size(); i++){
                dims[i] = dimensions[i];
            }
            recomposer.recompose(data, dims, target_level);

            // size_t size = 1;
            // std::vector<mgard_cuda::SIZE> shape(D);
            // for(mgard_cuda::DIM i=0; i<D; i++){
            //     shape[i] = dimensions[i];
            //     size *= dimensions[i];
            // }
            // mgard_cuda::Handle<D, T> handle(shape);
            // mgard_cuda::Array<D, T> array(shape);
            // array.loadData((const T*)data);
            // handle.allocate_workspace();
            // mgard_cuda::recompose<D, T>(handle, array.get_dv(), array.get_ldvs_h(), array.get_ldvs_d(),
            // target_level, 0);
            // handle.sync_all();
            // handle.free_workspace();
            // std::memcpy(data, array.getDataHost(), size*sizeof(T)); 
            
        }
        void print() const {
            std::cout << "MGARD orthogonal decomposer" << std::endl;
        }
    private:
        Handle<D, T> &_handle;
    };
    // MGARD decomposer with hierarchical basis
    template<DIM D, typename T>
    class MGARDHierarchicalDecomposer : public concepts::DecomposerInterface<D, T> {
    public:
        MGARDHierarchicalDecomposer(Handle<D, T> &handle): _handle(handle){}
        void decompose(T * data, const std::vector<uint32_t>& dimensions, uint32_t target_level) const {
            // MGARD::Decomposer<T> decomposer;
            // std::vector<size_t> dims(dimensions.size());
            // for(int i=0; i<dims.size(); i++){
            //     dims[i] = dimensions[i];
            // }
            // decomposer.decompose(data, dims, target_level, true);
        }
        void recompose(T * data, const std::vector<uint32_t>& dimensions, uint32_t target_level) const {
            // MGARD::Recomposer<T> recomposer;
            // std::vector<size_t> dims(dimensions.size());
            // for(int i=0; i<dims.size(); i++){
            //     dims[i] = dimensions[i];
            // }
            // recomposer.recompose(data, dims, target_level, true);
        }
        void print() const {
            std::cout << "MGARD hierarchical decomposer" << std::endl;
        }
    private:
        Handle<D, T> &_handle;
    };
}
}


namespace mgard_m {
namespace MDR {
    // MGARD decomposer with orthogonal basis
    template<typename HandleType, mgard_cuda::DIM D, typename T>
    class MGARDOrthoganalDecomposer : public concepts::DecomposerInterface<HandleType, D, T> {
    public:
        MGARDOrthoganalDecomposer(HandleType &handle): handle(handle){}
        void decompose(mgard_cuda::SubArray<D, T> v, mgard_cuda::SIZE target_level, int queue_idx) const {
            handle.allocate_workspace();
            mgard_cuda::decompose<D, T>(handle, v.dv, v.ldvs_h, v.ldvs_d, target_level, queue_idx);
            handle.sync(queue_idx);
            handle.free_workspace();
        }
        void recompose(mgard_cuda::SubArray<D, T> v, mgard_cuda::SIZE target_level, int queue_idx) const {
            handle.allocate_workspace();
            mgard_cuda::recompose<D, T>(handle, v.dv, v.ldvs_h, v.ldvs_d, target_level, queue_idx);
            handle.sync(queue_idx);
            handle.free_workspace();
        }
        void print() const {
            std::cout << "MGARD orthogonal decomposer" << std::endl;
        }
    private:
        HandleType &handle;
    };

}
}
#endif
