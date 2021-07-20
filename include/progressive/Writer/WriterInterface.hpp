#ifndef _MDR_WRITER_INTERFACE_HPP
#define _MDR_WRITER_INTERFACE_HPP

namespace MDR {
    namespace concepts {

        // Refactored data writer
        class WriterInterface {
        public:

            virtual ~WriterInterface() = default;

            virtual std::vector<uint32_t> write_level_components(const std::vector<std::vector<uint8_t*>>& level_components, const std::vector<std::vector<uint32_t>>& level_sizes) const = 0;

            virtual void write_metadata(uint8_t const * metadata, uint32_t size) const = 0;

            virtual void print() const = 0;
        };
    }
}
#endif
