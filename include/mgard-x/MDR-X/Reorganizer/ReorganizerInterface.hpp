#ifndef _MDR_REORGANIZER_INTERFACE_HPP
#define _MDR_REORGANIZER_INTERFACE_HPP
namespace mgard_x {
namespace MDR {
namespace concepts {

// level bit-plane reorganizer: EBCOT-like algorithm for multilevel bit-plane
// truncation
class ReorganizerInterface {
public:
  virtual ~ReorganizerInterface() = default;

  virtual uint8_t *
  reorganize(const std::vector<std::vector<uint8_t *>> &level_components,
             const std::vector<std::vector<SIZE>> &level_sizes,
             std::vector<uint8_t> &order, SIZE &total_size) const = 0;

  virtual void print() const = 0;
};
} // namespace concepts
} // namespace MDR
} // namespace mgard_x
#endif
