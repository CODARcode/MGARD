#ifndef _MDR_INTERLEAVER_INTERFACE_HPP
#define _MDR_INTERLEAVER_INTERFACE_HPP

#include "../../CommonInternal.h"

namespace mgard_cuda {
namespace MDR {
    namespace concepts {

        // level data interleaver: interleave level coefficients
        template<mgard_cuda::DIM D, typename T>
        class InterleaverInterface {
        public:

            virtual ~InterleaverInterface() = default;

            virtual void interleave(T const * data, const std::vector<uint32_t>& dims, const std::vector<uint32_t>& dims_fine, const std::vector<uint32_t>& dims_coasre, T * buffer) const = 0;

            virtual void reposition(T const * buffer, const std::vector<uint32_t>& dims, const std::vector<uint32_t>& dims_fine, const std::vector<uint32_t>& dims_coasre, T * data) const = 0;

            virtual void print() const = 0;
        };
    }
}
}


namespace mgard_m {
namespace MDR {
    namespace concepts {

        // level data interleaver: interleave level coefficients
        template<typename HandleType, mgard_cuda::DIM D, typename T>
        class InterleaverInterface {
        public:

            virtual ~InterleaverInterface() = default;

            virtual void interleave(mgard_cuda::SubArray<D, T> decomposed_data, 
                                    mgard_cuda::SubArray<1, T> * levels_decomposed_data, 
                                    int queue_idx) const = 0;

            virtual void reposition(mgard_cuda::SubArray<1, T> * levels_decomposed_data, 
                                    mgard_cuda::SubArray<D, T> decomposed_data, 
                                    int queue_idx) const = 0;

            virtual void print() const = 0;
        };
    }
}
}

#endif
