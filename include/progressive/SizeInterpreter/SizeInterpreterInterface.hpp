#ifndef _MDR_SIZE_INTERPRETER_INTERFACE_HPP
#define _MDR_SIZE_INTERPRETER_INTERFACE_HPP

namespace MDR {
    namespace concepts {

        // level bit-plane reorganizer: EBCOT-like algorithm for multilevel bit-plane truncation
        class SizeInterpreterInterface {
        public:

            virtual ~SizeInterpreterInterface() = default;

            virtual std::vector<uint32_t> interpret_retrieve_size(const std::vector<std::vector<uint32_t>>& level_sizes, const std::vector<std::vector<double>>& level_errors, double tolerance, std::vector<uint8_t>& index) const = 0;

            virtual void print() const = 0;
        };
    }
}
#endif
