#ifndef _MDR_INTERLEAVER_INTERFACE_HPP
#define _MDR_INTERLEAVER_INTERFACE_HPP

namespace MDR {
    namespace concepts {

        // level data interleaver: interleave level coefficients
        template<class T>
        class InterleaverInterface {
        public:

            virtual ~InterleaverInterface() = default;

            virtual void interleave(T const * data, const std::vector<uint32_t>& dims, const std::vector<uint32_t>& dims_fine, const std::vector<uint32_t>& dims_coasre, T * buffer) const = 0;

            virtual void reposition(T const * buffer, const std::vector<uint32_t>& dims, const std::vector<uint32_t>& dims_fine, const std::vector<uint32_t>& dims_coasre, T * data) const = 0;

            virtual void print() const = 0;
        };
    }
}
#endif
