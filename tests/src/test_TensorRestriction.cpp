#include "catch2/catch.hpp"

#include <array>

#include "testing_random.hpp"
#include "testing_utilities.hpp"

#include "TensorMassMatrix.hpp"
#include "TensorMeshHierarchy.hpp"
#include "TensorProlongation.hpp"
#include "TensorRestriction.hpp"
#include "utilities.hpp"

TEST_CASE("constituent restrictions", "[TensorRestriction]") {
  SECTION("1D and default spacing") {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({9});
    const std::array<float, 9> u_ = {9, 2, 4, -4, 7, 5, -2, 5, 6};
    const std::size_t dimension = 0;
    const std::array<std::array<float, 9>, 3> expecteds = {
        {{10, 2, 3, -4, 7.5, 5, 3, 5, 8.5},
         {11, 2, 4, -4, 8, 5, -2, 5, 5},
         {12.5, 2, 4, -4, 7, 5, -2, 5, 9.5}}};
    for (std::size_t l = 3; l > 0; --l) {
      const std::size_t i = 3 - l;
      const std::array<float, 9> &expected = expecteds.at(i);
      const mgard::ConstituentRestriction<1, float> R(hierarchy, l, dimension);
      std::array<float, 9> v_ = u_;
      float *const v = v_.data();
      R({0}, v);
      TrialTracker tracker;
      for (std::size_t i = 0; i < 9; ++i) {
        tracker += v_.at(i) == expected.at(i);
      }
      REQUIRE(tracker);
    }
  }

  SECTION("2D and custom spacing") {
    const mgard::TensorMeshHierarchy<2, double> hierarchy(
        {3, 3}, {{{0, 0.75, 1}, {0, 0.25, 1}}});
    const std::array<double, 9> u_ = {-9, -5, 1, 9, 3, 2, 9, 5, 6};
    {
      const std::size_t l = 1;
      const std::size_t dimension = 0;
      const mgard::ConstituentRestriction<2, double> R(hierarchy, l, dimension);
      const std::array<std::array<std::size_t, 2>, 3> multiindices = {
          {{0, 0}, {0, 1}}};
      const std::array<std::array<double, 9>, 2> expecteds = {{
          {-6.75, -5, 1, 9, 3, 2, 15.75, 5, 6},
          {-9, -4.25, 1, 9, 3, 2, 9, 7.25, 6},
      }};
      for (std::size_t i = 0; i < 2; ++i) {
        const std::array<std::size_t, 2> &multiindex = multiindices.at(i);
        const std::array<double, 9> &expected = expecteds.at(i);
        std::array<double, 9> v_ = u_;
        double *const v = v_.data();
        R(multiindex, v);
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
      const mgard::ConstituentRestriction<2, double> R(hierarchy, l, dimension);
      const std::array<std::array<std::size_t, 2>, 3> multiindices = {
          {{1, 0}, {2, 0}}};
      const std::array<std::array<double, 9>, 2> expecteds = {{
          {-9, -5, 1, 11.25, 3, 2.75, 9, 5, 6},
          {-9, -5, 1, 9, 3, 2, 12.75, 5, 7.25},
      }};
      for (std::size_t i = 0; i < 2; ++i) {
        const std::array<std::size_t, 2> &multiindex = multiindices.at(i);
        const std::array<double, 9> &expected = expecteds.at(i);
        std::array<double, 9> v_ = u_;
        double *const v = v_.data();
        R(multiindex, v);
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
        const mgard::ConstituentRestriction<2, double> R(hierarchy, 0, 0);
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

// Prolongate a function on the coarse mesh to the fine mesh, project back down
// to the coarse mesh, and compare.
template <std::size_t N, typename Real>
void test_tensor_projection_identity(std::default_random_engine &generator,
                                     const std::array<std::size_t, N> shape) {
  std::uniform_real_distribution<Real> node_spacing_distribution(1, 1.5);
  std::uniform_real_distribution<Real> polynomial_coefficient_distribution(
      -0.5, 1.25);
  const mgard::TensorMeshHierarchy<N, Real> hierarchy =
      hierarchy_with_random_spacing(generator, node_spacing_distribution,
                                    shape);

  std::vector<Real> u_(hierarchy.ndof());
  Real *const u = u_.data();

  for (std::size_t l = hierarchy.L; l > 0; --l) {
    const MultilinearPolynomial<Real, N> p(generator,
                                           polynomial_coefficient_distribution);
    for (const mgard::SituatedCoefficient<N, Real> coeff :
         hierarchy.on_nodes(u, l)) {
      *coeff.value = 0;
    }
    for (const mgard::SituatedCoefficient<N, Real> coeff :
         hierarchy.on_nodes(u, l - 1)) {
      *coeff.value = p(coeff.coordinates);
    }

    const mgard::TensorProlongationAddition<N, Real> PA(hierarchy, l);
    const mgard::TensorRestriction<N, Real> R(hierarchy, l);
    const mgard::TensorMassMatrixInverse<N, Real> m_inv(hierarchy, l - 1);
    const mgard::TensorMassMatrix<N, Real> M(hierarchy, l);

    PA(u);
    M(u);
    R(u);
    m_inv(u);

    TrialTracker tracker;
    for (const mgard::SituatedCoefficient<N, Real> coeff :
         hierarchy.on_nodes(u, l - 1)) {
      tracker += hierarchy.at(u, coeff.multiindex) == Approx(*coeff.value);
    }
    REQUIRE(tracker);
  }
}

} // namespace

TEST_CASE("tensor product restrictions", "[TensorRestriction]") {
  {
    const mgard::TensorMeshHierarchy<2, double> hierarchy(
        {3, 3}, {{{0, 0.5, 1}, {-1, -0.5, 1}}});
    const std::array<double, 9> u_ = {6, 0, 7, -6, -10, 8, -6, 3, 9};
    const std::size_t l = 1;
    const mgard::TensorRestriction<2, double> R(hierarchy, l);
    const std::array<double, 9> expected = {-0.75, -5,    9.75, -13.5, -10,
                                            5.5,   -10.5, -2,   12.5};
    std::array<double, 9> v_ = u_;
    double *const v = v_.data();
    R(v);
    REQUIRE(v_ == expected);
  }

  {
    std::default_random_engine generator(445624);
    test_tensor_projection_identity<1, float>(generator, {129});
    test_tensor_projection_identity<2, double>(generator, {65, 65});
    test_tensor_projection_identity<3, float>(generator, {33, 9, 33});
    test_tensor_projection_identity<4, double>(generator, {9, 9, 17, 17});
  }
}
