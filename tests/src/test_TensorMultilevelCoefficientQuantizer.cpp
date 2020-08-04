#include "catch2/catch.hpp"

#include <array>
#include <numeric>

#include "TensorMeshHierarchy.hpp"
#include "TensorMultilevelCoefficientQuantizer.hpp"

namespace {

template <std::size_t N, typename Real, typename Int>
void test_mc_quantization_iteration(const std::array<std::size_t, N> shape,
                                    const Real s, const Real tolerance) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(shape);
  const std::size_t M = hierarchy.ndof();

  {
    std::vector<Real> u_(M);
    std::iota(u_.begin(), u_.end(), -20);
    Real *const u = u_.data();

    const mgard::TensorMultilevelCoefficientQuantizer<N, Real, Int> quantizer(
        hierarchy, s, tolerance);
    TrialTracker tracker;
    std::size_t count = 0;
    for (const Int n : quantizer(u)) {
      static_cast<void>(n);
      ++count;
    }
    REQUIRE(count == M);
  }
}

} // namespace

TEST_CASE("tensor multilevel coefficient (de)quantization iteration",
          "[TensorMultilevelCoefficientQuantizer]") {
  test_mc_quantization_iteration<1, float, int>(
      {65}, std::numeric_limits<float>::infinity(), 0.1);
  test_mc_quantization_iteration<2, double, long int>(
      {22, 19}, std::numeric_limits<double>::infinity(), 0.01);
  test_mc_quantization_iteration<3, float, long int>(
      {10, 9, 9}, std::numeric_limits<float>::infinity(), 0.25);
}
