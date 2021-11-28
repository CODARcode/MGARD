#ifndef _MDR_INTERLEAVER_INTERFACE_HPP
#define _MDR_INTERLEAVER_INTERFACE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {
namespace MDR {
    namespace concepts {

        // level data interleaver: interleave level coefficients
        template<mgard_x::DIM D, typename T>
        class InterleaverInterface {
        public:

            virtual ~InterleaverInterface() = default;

            virtual void interleave(T const * data, const std::vector<SIZE>& dims, const std::vector<SIZE>& dims_fine, const std::vector<SIZE>& dims_coasre, T * buffer) const = 0;

            virtual void reposition(T const * buffer, const std::vector<SIZE>& dims, const std::vector<SIZE>& dims_fine, const std::vector<SIZE>& dims_coasre, T * data) const = 0;

            virtual void print() const = 0;
        };
    }
}
}


namespace mgard_m {
namespace MDR {
    namespace concepts {

        // level data interleaver: interleave level coefficients
        template<typename HandleType, mgard_x::DIM D, typename T>
        class InterleaverInterface {
        public:

            virtual ~InterleaverInterface() = default;

            virtual void interleave(mgard_x::SubArray<D, T, mgard_x::CUDA> decomposed_data, 
                                    mgard_x::SubArray<1, T, mgard_x::CUDA> * levels_decomposed_data, 
                                    int queue_idx) const = 0;

            virtual void reposition(mgard_x::SubArray<1, T, mgard_x::CUDA> * levels_decomposed_data, 
                                    mgard_x::SubArray<D, T, mgard_x::CUDA> decomposed_data, 
                                    int queue_idx) const = 0;

            virtual void print() const = 0;
        };
    }
}
}

#endif
