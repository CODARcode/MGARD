#ifndef TENSORQUANTITYOFINTEREST_HPP
#define TENSORQUANTITYOFINTEREST_HPP
//!\file
//!\brief Functionals defined on tensor product function spaces.

#include <cstdlib>

#include <vector>

#include "TensorMeshHierarchy.hpp"

namespace mgard {

//! Functional defined on a tensor product function space.
template <std::size_t N, typename Real> class TensorQuantityOfInterest {
public:
  //! Constructor.
  //!
  //! We assume that `Functional` has a member function with signature
  //!
  //!     Real Functional::operator()(
  //!         const TensorMeshHierarchy<N, Real> &hierarchy,
  //!         Real const * const u
  //!     ) const;
  //!
  //! where `u` is assumed to be *unshuffled.*
  //!
  //!\param hierarchy Mesh hierarchy on which the functions in the function
  //! space are defined.
  //!\param functional Functional defined on the function space.
  template <typename Functional>
  TensorQuantityOfInterest(const TensorMeshHierarchy<N, Real> &hierarchy,
                           const Functional &functional);

  //! Compute the operator norm of the functional.
  //!
  //! Let `V` denote the function space on the finest mesh. Then this
  //! member function computes the norm of the functional as an operator
  //! `(V, ‖·‖_{s}) → (R, |·|)`. That is, we use the `s` norm on the domain.
  //!
  //!\param s Smoothness parameter for the norm.
  Real norm(const Real s) const;

private:
  //! Associated mesh hierarchy.
  const TensorMeshHierarchy<N, Real> &hierarchy;

  //! Square `L^2` norms of the orthogonal components of the Riesz
  //! representative, ordered from coarset to finest mesh.
  std::vector<Real> component_square_norms;
};

} // namespace mgard

#include "TensorQuantityOfInterest.tpp"
#endif
