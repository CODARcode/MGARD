#ifndef _MDR_RETRIEVER_INTERFACE_HPP
#define _MDR_RETRIEVER_INTERFACE_HPP

#include <cassert>
namespace mgard_x {
namespace MDR {
namespace concepts {

// Error-controlled data retrieval
class RetrieverInterface {
public:
  virtual ~RetrieverInterface() = default;

  virtual std::vector<std::vector<const uint8_t *>> retrieve_level_components(
      const std::vector<std::vector<SIZE>> &level_sizes,
      const std::vector<SIZE> &retrieve_sizes,
      const std::vector<uint8_t> &prev_level_num_bitplanes,
      const std::vector<uint8_t> &level_num_bitplanes) = 0;

  virtual uint8_t *load_metadata() const = 0;

  virtual void release() = 0;

  virtual void print() const = 0;
};
} // namespace concepts
} // namespace MDR
} // namespace mgard_x
#endif
