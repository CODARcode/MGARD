#ifndef _MDR_DECOMPOSER_INTERFACE_HPP
#define _MDR_DECOMPOSER_INTERFACE_HPP
#include "../../Common.h"
#include "../../CommonInternal.h"
namespace mgard_cuda {
namespace MDR {
    namespace concepts {

        // inplace data decomposer: de-correlates and overwrites original data
        template<DIM D, typename T>
        class DecomposerInterface {
        public:

            virtual ~DecomposerInterface() = default;

            virtual void decompose(T * data, const std::vector<SIZE>& dimensions, SIZE target_level) const = 0;

            virtual void recompose(T * data, const std::vector<SIZE>& dimensions, SIZE target_level) const = 0;

            virtual void print() const = 0;
        };
    }
}
}

namespace mgard_m {
namespace MDR {
    namespace concepts {

        // inplace data decomposer: de-correlates and overwrites original data
        template<typename HandleType, mgard_cuda::DIM D, typename T>
        class DecomposerInterface {
        public:

            virtual ~DecomposerInterface() = default;

            virtual void decompose(mgard_cuda::SubArray<D, T, mgard_cuda::CUDA> v, mgard_cuda::SIZE target_level, int queue_idx) const = 0;

            virtual void recompose(mgard_cuda::SubArray<D, T, mgard_cuda::CUDA> v, mgard_cuda::SIZE target_level, int queue_idx) const = 0;

            virtual void print() const = 0;
        };
    }
}
}

#endif
