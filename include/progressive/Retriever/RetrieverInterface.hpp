#ifndef _MDR_RETRIEVER_INTERFACE_HPP
#define _MDR_RETRIEVER_INTERFACE_HPP

#include <cassert>

namespace MDR {
    namespace concepts {

        // Error-controlled data retrieval
        class RetrieverInterface {
        public:

            virtual ~RetrieverInterface() = default;

            virtual std::vector<std::vector<const uint8_t*>> retrieve_level_components(const std::vector<std::vector<uint32_t>>& level_sizes, const std::vector<uint32_t>& retrieve_sizes, const std::vector<uint8_t>& prev_level_num_bitplanes, const std::vector<uint8_t>& level_num_bitplanes) = 0;

            virtual uint8_t * load_metadata() const = 0;

            virtual void release() = 0;

            virtual void print() const = 0;
        };
    }
}
#endif
