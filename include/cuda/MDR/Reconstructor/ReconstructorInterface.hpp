#ifndef _MDR_RECONSTRUCTOR_INTERFACE_HPP
#define _MDR_RECONSTRUCTOR_INTERFACE_HPP

#include "../../CommonInternal.h"

namespace mgard_cuda {
namespace MDR {
    namespace concepts {

        // reconstructor: a general interface for scientific data reconstructor
        template<typename T>
        class ReconstructorInterface {
        public:

            virtual ~ReconstructorInterface() = default;

            virtual T * reconstruct(double tolerance) = 0;

            virtual T * progressive_reconstruct(double tolerance) = 0;

            virtual void load_metadata() = 0;

            virtual void print() const = 0;
        };
    }
}
}


namespace mgard_m {
namespace MDR {
    namespace concepts {

        // reconstructor: a general interface for scientific data reconstructor
        template<typename HandleType, mgard_cuda::DIM D, typename T_data, typename T_bitplane>
        class ReconstructorInterface {
        public:

            virtual ~ReconstructorInterface() = default;

            virtual T_data * reconstruct(double tolerance) = 0;

            virtual T_data * progressive_reconstruct(double tolerance) = 0;

            virtual void load_metadata() = 0;

            virtual void print() const = 0;
        };
    }
}
}

#endif
