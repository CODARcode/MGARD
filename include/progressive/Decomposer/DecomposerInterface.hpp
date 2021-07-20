#ifndef _MDR_DECOMPOSER_INTERFACE_HPP
#define _MDR_DECOMPOSER_INTERFACE_HPP

namespace MDR {
    namespace concepts {

        // inplace data decomposer: de-correlates and overwrites original data
        template<class T>
        class DecomposerInterface {
        public:

            virtual ~DecomposerInterface() = default;

            virtual void decompose(T * data, const std::vector<uint32_t>& dimensions, uint32_t target_level) const = 0;

            virtual void recompose(T * data, const std::vector<uint32_t>& dimensions, uint32_t target_level) const = 0;

            virtual void print() const = 0;
        };
    }
}
#endif
