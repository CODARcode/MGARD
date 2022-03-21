#ifndef _MDR_SIZE_INTERPRETER_INTERFACE_HPP
#define _MDR_SIZE_INTERPRETER_INTERFACE_HPP
namespace mgard_x {
namespace MDR {
namespace concepts {

// level bit-plane reorganizer: EBCOT-like algorithm for multilevel bit-plane
// truncation
class SizeInterpreterInterface {
public:
  virtual ~SizeInterpreterInterface() = default;

  virtual std::vector<SIZE>
  interpret_retrieve_size(const std::vector<std::vector<SIZE>> &level_sizes,
                          const std::vector<std::vector<double>> &level_errors,
                          double tolerance,
                          std::vector<uint8_t> &index) const = 0;

  virtual void print() const = 0;
};
} // namespace concepts
} // namespace MDR
} // namespace mgard_x
#endif
