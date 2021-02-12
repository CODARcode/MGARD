#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

#include <cstdlib>

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

#include "TensorMeshHierarchy.hpp"
#include "TensorNorms.hpp"
#include "TensorQuantityOfInterest.hpp"
#include "blas.hpp"

#include "testing_random.hpp"

namespace {

//! Functional given by an inner product with a function.
template <std::size_t N, typename Real> class RieszRepresentative {
public:
  //! Constructor.
  //!
  //! IMPORTANT: `representative` must be shuffled.
  //!
  //!\param hierarchy Mesh hierarchy associated to the function space.
  //!\param representative Function φ such that the functional is given by
  //! (φ, ·).
  RieszRepresentative(const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
                      Real const *const representative);

  //! Destructor.
  ~RieszRepresentative();

  //! Apply the functional to a function.
  //!
  //! IMPORTANT: `u` must be unshuffled.
  //!
  //!\param hierarchy Mesh hierarchy on which the function is defined.
  //!\param u Input to the functional.
  Real operator()(const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
                  Real const *const u) const;

private:
  //! Associated mesh hierarchy.
  const mgard::TensorMeshHierarchy<N, Real> &hierarchy;

  //! Size of the finest mesh in the hierarchy.
  std::size_t ndof;

  //! Buffer in which to store shuffled inputs.
  Real *u;

  //! Product of the mass matrix and the Riesz representative.
  Real *f;
};

template <std::size_t N, typename Real>
RieszRepresentative<N, Real>::RieszRepresentative(
    const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
    Real const *const representative)
    : hierarchy(hierarchy), ndof(hierarchy.ndof()),
      u(static_cast<Real *>(std::malloc(ndof * sizeof(Real)))),
      f(static_cast<Real *>(std::malloc(ndof * sizeof(Real)))) {
  std::copy(representative, representative + ndof, f);
  const mgard::TensorMassMatrix<N, Real> M(hierarchy, hierarchy.L);
  M(f);
}

template <std::size_t N, typename Real>
RieszRepresentative<N, Real>::~RieszRepresentative() {
  std::free(f);
  std::free(u);
}

template <std::size_t N, typename Real>
Real RieszRepresentative<N, Real>::
operator()(const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
           Real const *const u) const {
  if (hierarchy != this->hierarchy) {
    throw std::domain_error(
        "construction and calling hierarchies must be equal");
  }
  mgard::shuffle(hierarchy, u, this->u);
  return blas::dotu(hierarchy.ndof(), f, this->u);
}

template <std::size_t N, typename Real>
void test_qoi_norm_equality(std::default_random_engine &generator,
                            std::uniform_real_distribution<Real> &distribution,
                            const std::array<std::size_t, N> shape,
                            const Real s) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy =
      hierarchy_with_random_spacing(generator, distribution, shape);

  Real *const representative =
      static_cast<Real *>(std::malloc(hierarchy.ndof() * sizeof(Real)));
  generate_reasonable_function<N, Real>(hierarchy, static_cast<Real>(1),
                                        generator, representative);
  const Real representative_norm = mgard::norm(hierarchy, representative, -s);
  const RieszRepresentative<N, Real> functional(hierarchy, representative);
  const mgard::TensorQuantityOfInterest<N, Real> Q(hierarchy, functional);
  std::free(representative);

  REQUIRE(Q.norm(s) == Catch::Approx(representative_norm));
}

} // namespace

TEST_CASE("Riesz representative norm equality", "[qoi]") {
  std::default_random_engine gen;
  // Node spacing distribution.
  std::uniform_real_distribution<double> dis(0.05, 0.075);

  test_qoi_norm_equality<1, double>(gen, dis, {5}, 0);
  test_qoi_norm_equality<1, double>(gen, dis, {18}, -0.25);
  test_qoi_norm_equality<2, double>(gen, dis, {13, 13}, 0.5);
  test_qoi_norm_equality<2, double>(gen, dis, {40, 7}, -0.75);
  test_qoi_norm_equality<3, double>(gen, dis, {4, 4, 5}, 1.0);
  test_qoi_norm_equality<3, double>(gen, dis, {9, 5, 6}, -1.25);
}

namespace {

template <std::size_t N, typename Real>
void test_average_norms(const std::array<std::size_t, N> shape) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(shape);

  Real *const ones =
      static_cast<Real *>(std::malloc(hierarchy.ndof() * sizeof(Real)));
  std::fill(ones, ones + hierarchy.ndof(), static_cast<Real>(1));
  const RieszRepresentative<N, Real> functional(hierarchy, ones);
  std::free(ones);
  const mgard::TensorQuantityOfInterest<N, Real> Q(hierarchy, functional);

  const std::vector<Real> smoothness_parameters = {-1.5, -0.5, 0, 0.5, 1.5};
  Real minimum_norm = std::numeric_limits<Real>::max();
  Real maximum_norm = std::numeric_limits<Real>::min();
  for (const Real s : smoothness_parameters) {
    const Real norm = Q.norm(s);
    minimum_norm = std::min(minimum_norm, norm);
    maximum_norm = std::max(maximum_norm, norm);
  }
  REQUIRE(minimum_norm == Catch::Approx(maximum_norm).epsilon(1e-3));
}

} // namespace

TEST_CASE("average quantity of interest", "[qoi]") {
  // Could be any function contained in the coarsest function space here.
  test_average_norms<1, float>({25});
  test_average_norms<2, double>({12, 14});
  test_average_norms<3, float>({10, 11, 4});
  test_average_norms<4, double>({6, 3, 3, 5});
}
