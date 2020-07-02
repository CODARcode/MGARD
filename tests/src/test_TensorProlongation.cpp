#include "catch2/catch.hpp"

#include <array>

#include "testing_random.hpp"
#include "testing_utilities.hpp"

#include "TensorMeshHierarchy.hpp"
#include "TensorProlongation.hpp"
#include "utilities.hpp"

TEST_CASE("constituent prolongations", "[TensorProlongation]") {
  SECTION("1D and default spacing") {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({9});
    const std::array<float, 9> u_ = {-10, 10, -2, 3, 9, -8, 9, -8, 4};
    const std::size_t dimension = 0;
    const std::array<std::array<float, 9>, 3> expecteds = {
        {{-10, 4, -2, 6.5, 9, 1, 9, -1.5, 4},
         {-10, 10, -2.5, 3, 9, -8, 15.5, -8, 4},
         {-10, 10, -2, 3, 6, -8, 9, -8, 4}}};
    for (std::size_t l = 3; l > 0; --l) {
      const std::size_t i = 3 - l;
      const std::array<float, 9> &expected = expecteds.at(i);
      const mgard::ConstituentProlongationAddition<1, float> PA(hierarchy, l,
                                                                dimension);
      std::array<float, 9> v_ = u_;
      float *const v = v_.data();
      PA({0}, v);
      TrialTracker tracker;
      for (std::size_t i = 0; i < 9; ++i) {
        tracker += v_.at(i) == expected.at(i);
      }
      REQUIRE(tracker);
    }
  }

  SECTION("2D and custom spacing") {
    const mgard::TensorMeshHierarchy<2, double> hierarchy(
        {3, 3}, {{{0, 0.9, 1}, {1, 1.4, 2}}});
    const std::array<double, 9> u_ = {4, -1, 4, 7, 3, -3, -4, 1, -7};
    {
      const std::size_t l = 1;
      const std::size_t dimension = 0;
      const mgard::ConstituentProlongationAddition<2, double> PA(hierarchy, l,
                                                                 dimension);
      const std::array<std::array<std::size_t, 2>, 3> multiindices = {
          {{0, 0}, {0, 1}}};
      const std::array<std::array<double, 9>, 2> expecteds = {
          {{4, -1, 4, 3.8, 3, -3, -4, 1, -7},
           {4, -1, 4, 7, 3.8, -3, -4, 1, -7}}};
      for (std::size_t i = 0; i < 2; ++i) {
        const std::array<std::size_t, 2> &multiindex = multiindices.at(i);
        const std::array<double, 9> &expected = expecteds.at(i);
        std::array<double, 9> v_ = u_;
        double *const v = v_.data();
        PA(multiindex, v);
        TrialTracker tracker;
        for (std::size_t i = 0; i < 9; ++i) {
          tracker += v_.at(i) == Approx(expected.at(i));
        }
        REQUIRE(tracker);
      }
    }
    {
      const std::size_t l = 1;
      const std::size_t dimension = 1;
      const mgard::ConstituentProlongationAddition<2, double> PA(hierarchy, l,
                                                                 dimension);
      const std::array<std::array<std::size_t, 2>, 3> multiindices = {
          {{1, 0}, {2, 0}}};
      const std::array<std::array<double, 9>, 2> expecteds = {
          {{4, -1, 4, 7, 6.0, -3, -4, 1, -7},
           {4, -1, 4, 7, 3, -3, -4, -4.2, -7}}};
      for (std::size_t i = 0; i < 2; ++i) {
        const std::array<std::size_t, 2> &multiindex = multiindices.at(i);
        const std::array<double, 9> &expected = expecteds.at(i);
        std::array<double, 9> v_ = u_;
        double *const v = v_.data();
        PA(multiindex, v);
        TrialTracker tracker;
        for (std::size_t i = 0; i < 9; ++i) {
          tracker += v_.at(i) == Approx(expected.at(i));
        }
        REQUIRE(tracker);
      }
    }
    {
      bool thrown = false;
      try {
        const mgard::ConstituentProlongationAddition<2, double> PA(hierarchy, 0,
                                                                   0);
      } catch (...) {
        // The constructor will throw a `std::invalid_argument` exception, but
        // (as of this writing) the attempt to initialize `coarse_indices`
        // throws first.
        thrown = true;
      }
      REQUIRE(thrown);
    }
  }
}

namespace {

template <std::size_t N, typename Real>
std::array<Real, N>
multiindex_coordinates(const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
                       const std::array<std::size_t, N> multiindex) {
  std::array<Real, N> coordinates;
  for (std::size_t i = 0; i < N; ++i) {
    coordinates.at(i) = hierarchy.coordinates.at(i).at(multiindex.at(i));
  }
  return coordinates;
}

template <std::size_t N, typename Real>
void test_tensor_product_prolongations(std::default_random_engine &generator,
                                       const std::array<std::size_t, N> shape) {
  std::uniform_real_distribution<Real> node_spacing_distribution(0.1, 1);
  std::uniform_real_distribution<Real> polynomial_coefficient_distribution(-2,
                                                                           2);
  const mgard::TensorMeshHierarchy<N, Real> hierarchy =
      hierarchy_with_random_spacing(generator, node_spacing_distribution,
                                    shape);
  std::vector<Real> u_(hierarchy.ndof());
  Real *const u = u_.data();
  // For each level, the indices (separated by dimension) whose Cartesian
  // product will give the multiindices of the nodes on that level.
  std::vector<std::array<std::vector<std::size_t>, N>> multiindex_factors(
      hierarchy.L + 1);
  for (std::size_t l = 0; l <= hierarchy.L; ++l) {
    std::array<std::vector<std::size_t>, N> &factors = multiindex_factors.at(l);
    for (std::size_t i = 0; i < N; ++i) {
      factors.at(i) = hierarchy.indices(l, i);
    }
  }
  for (std::size_t l = hierarchy.L; l > 0; --l) {
    const MultilinearPolynomial<Real, N> p(generator,
                                           polynomial_coefficient_distribution);
    for (const std::array<std::size_t, N> multiindex :
         mgard::CartesianProduct(multiindex_factors.at(l))) {
      hierarchy.at(u, multiindex) = 0;
    }
    for (const std::array<std::size_t, N> multiindex :
         mgard::CartesianProduct(multiindex_factors.at(l - 1))) {
      hierarchy.at(u, multiindex) =
          p(multiindex_coordinates(hierarchy, multiindex));
    }
    const mgard::TensorProlongationAddition<N, Real> PA(hierarchy, l);
    PA(u);
    TrialTracker tracker;
    for (const std::array<std::size_t, N> multiindex :
         mgard::CartesianProduct(multiindex_factors.at(l))) {
      tracker += hierarchy.at(u, multiindex) ==
                 Approx(p(multiindex_coordinates(hierarchy, multiindex)))
                     .epsilon(0.001);
    }
    REQUIRE(tracker);
  }
}

} // namespace

TEST_CASE("tensor product prolongations", "[TensorProlongation]") {
  std::default_random_engine generator(176067);
  test_tensor_product_prolongations<1, float>(generator, {129});
  test_tensor_product_prolongations<2, double>(generator, {17, 17});
  // Before increasing the `Approx` tolerance we got a handful of errors (all
  // quite small) with this one.
  test_tensor_product_prolongations<3, float>(generator, {9, 9, 17});
  test_tensor_product_prolongations<4, double>(generator, {33, 17, 33, 17});
}
