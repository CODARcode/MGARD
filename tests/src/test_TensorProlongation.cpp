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
        for (std::size_t j = 0; j < 9; ++j) {
          tracker += v_.at(j) == Approx(expected.at(j));
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
        for (std::size_t j = 0; j < 9; ++j) {
          tracker += v_.at(j) == Approx(expected.at(j));
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

  SECTION("2D and nondyadic") {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({4, 4});
    const std::array<float, 16> u_ = {-2, -4, -9, -4, -5,  8,  4, -9,
                                      -4, 2,  -5, 0,  -10, -5, 9, -1};
    {
      const std::array<std::size_t, 2> ls = {2, 1};
      const std::array<std::size_t, 2> dimensions = {0, 1};

      const std::array<std::array<std::size_t, 2>, 4> multiindices = {
          {{0, 0}, {0, 1}, {2, 0}, {3, 0}}};
      const std::array<std::array<float, 16>, 4> expecteds = {
          {{-2, -4, -9, -4, -5, 8, 4, -9, -11.5, 2, -5, 0, -10, -5, 9, -1},
           {-2, -4, -9, -4, -5, 8, 4, -9, -4, 3.5, -5, 0, -10, -5, 9, -1},
           {-2, -4, -9, -4, -5, 8, 4, -9, -4, -2. / 3, -5, 0, -10, -5, 9, -1},
           {-2, -4, -9, -4, -5, 8, 4, -9, -4, 2, -5, 0, -10, -12, 9, -1}}};
      for (std::size_t i = 0; i < 2; ++i) {
        const std::size_t l = ls.at(i);
        const std::size_t dimension = dimensions.at(i);
        const mgard::ConstituentProlongationAddition<2, float> PA(hierarchy, l,
                                                                  dimension);
        for (std::size_t j = 2 * i; j < 2 * (i + 1); ++j) {
          const std::array<std::size_t, 2> &multiindex = multiindices.at(j);
          const std::array<float, 16> &expected = expecteds.at(j);
          std::array<float, 16> v_ = u_;
          float *const v = v_.data();
          PA(multiindex, v);
          TrialTracker tracker;
          for (std::size_t k = 0; k < 16; ++k) {
            tracker += v_.at(k) == Approx(expected.at(k));
          }
          REQUIRE(tracker);
        }
      }
    }
  }
}

namespace {

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
  for (std::size_t l = hierarchy.L; l > 0; --l) {
    const MultilinearPolynomial<Real, N> p(generator,
                                           polynomial_coefficient_distribution);
    for (const mgard::TensorNode<N, Real> node : hierarchy.nodes(l)) {
      hierarchy.at(u, node.multiindex) = node.l == l ? 0 : p(node.coordinates);
    }
    const mgard::TensorProlongationAddition<N, Real> PA(hierarchy, l);
    PA(u);
    TrialTracker tracker;
    for (const mgard::TensorNode<N, Real> node : hierarchy.nodes(l)) {
      tracker += hierarchy.at(u, node.multiindex) ==
                 Approx(p(node.coordinates)).epsilon(0.001);
    }
    REQUIRE(tracker);
  }
}

} // namespace

TEST_CASE("tensor product prolongations", "[TensorProlongation]") {
  std::default_random_engine generator(176067);

  SECTION("dyadic") {
    test_tensor_product_prolongations<1, float>(generator, {129});
    test_tensor_product_prolongations<2, double>(generator, {17, 17});
    // Before increasing the `Approx` tolerance we got a handful of errors (all
    // quite small) with this one.
    test_tensor_product_prolongations<3, float>(generator, {9, 9, 17});
    test_tensor_product_prolongations<4, double>(generator, {33, 17, 33, 17});
  }

  SECTION("nondyadic") {
    test_tensor_product_prolongations<1, double>(generator, {82});
    test_tensor_product_prolongations<2, float>(generator, {15, 17});
    test_tensor_product_prolongations<3, double>(generator, {6, 10, 8});
    test_tensor_product_prolongations<4, float>(generator, {5, 12, 8, 9});
  }
}
