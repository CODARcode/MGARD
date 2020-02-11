#include <cmath>

namespace mgard {

template <typename Real>
Real square_indicator_factor(const IndicatorInput<Real> input, const float s) {
  return std::pow(2, 2 * s * input.l) *
         input.mesh.containing_elements_measure(input.node);
}

} // namespace mgard
