#ifndef _MDR_REFACTOR_INTERFACE_HPP
#define _MDR_REFACTOR_INTERFACE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {
namespace MDR {
    namespace concepts {

        // refactor: a general interface for scnetific data refactor
        template<typename T>
        class RefactorInterface {
        public:

            virtual ~RefactorInterface() = default;

            virtual void refactor(T const * data_, const std::vector<SIZE>& dims, uint8_t target_level, uint8_t num_bitplanes) = 0;

            virtual void write_metadata() const = 0;

            virtual void print() const = 0;
        };
    }
}
}

namespace mgard_m {
namespace MDR {
    namespace concepts {

        // refactor: a general interface for scnetific data refactor
        template<typename HandleType, mgard_x::DIM D, typename T_data, typename T_bitplane>
        class RefactorInterface {
        public:

            virtual ~RefactorInterface() = default;

            virtual void refactor(T_data const * data_, const std::vector<mgard_x::SIZE>& dims, uint8_t target_level, uint8_t num_bitplanes) = 0;

            virtual void write_metadata() const = 0;

            virtual void print() const = 0;
        };
    }
}
}


#endif
