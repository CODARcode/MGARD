// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney, Qing Liu
// Corresponding Author: Ben Whitney, Qing Liu
//
// version: 0.1.0
// See LICENSE for details.
#ifndef MGARD_TPP
#define MGARD_TPP

#include "mgard.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <numeric>

#ifdef MGARD_TIMING
#include <chrono>
#endif

#include "mgard_compress.hpp"

#include "TensorMassMatrix.hpp"
#include "TensorProlongation.hpp"
#include "TensorRestriction.hpp"
#include "shuffle.hpp"

namespace mgard {

namespace {

// Not documenting the parameters here. I think it'll be easiest to understand
// by reading the code.

template <std::size_t N, typename Real>
void add_on_old_add_on_new(const TensorMeshHierarchy<N, Real> &hierarchy,
                           Real const *const src, Real *const dst,
                           const std::size_t l) {
  for (const TensorNode<N> node : hierarchy.nodes(l)) {
    hierarchy.at(dst, node.multiindex) += hierarchy.at(src, node.multiindex);
  }
}

template <std::size_t N, typename Real>
void subtract_on_old_zero_on_new(const TensorMeshHierarchy<N, Real> &hierarchy,
                                 Real const *const src, Real *const dst,
                                 const std::size_t l) {
  for (const TensorNode<N> node : hierarchy.nodes(l)) {
    Real &out = hierarchy.at(dst, node.multiindex);
    out = hierarchy.date_of_birth(node.multiindex) == l
              ? 0
              : out - hierarchy.at(src, node.multiindex);
  }
}

template <std::size_t N, typename Real>
void copy_negation_on_old_subtract_on_new(
    const TensorMeshHierarchy<N, Real> &hierarchy, Real const *const src,
    Real *const dst, const std::size_t l) {
  for (const TensorNode<N> node : hierarchy.nodes(l)) {
    const Real in = hierarchy.at(src, node.multiindex);
    Real &out = hierarchy.at(dst, node.multiindex);
    out = hierarchy.date_of_birth(node.multiindex) == l ? out - in : -in;
  }
}

template <std::size_t N, typename Real>
void copy_on_old_zero_on_new(const TensorMeshHierarchy<N, Real> &hierarchy,
                             Real const *const src, Real *const dst,
                             const std::size_t l) {
  // `l` shouldn't be zero, but in that case we'll do the expected thing: zero
  // every value.
  for (const TensorNode<N> node : hierarchy.nodes(l)) {
    hierarchy.at(dst, node.multiindex) =
        hierarchy.date_of_birth(node.multiindex) == l
            ? 0
            : hierarchy.at(src, node.multiindex);
  }
}

template <std::size_t N, typename Real>
void zero_on_old_copy_on_new(const TensorMeshHierarchy<N, Real> &hierarchy,
                             Real const *const src, Real *const dst,
                             const std::size_t l) {
  for (const TensorNode<N> node : hierarchy.nodes(l)) {
    hierarchy.at(dst, node.multiindex) =
        hierarchy.date_of_birth(node.multiindex) == l
            ? hierarchy.at(src, node.multiindex)
            : 0;
  }
}

template <std::size_t N, typename Real>
void zero_on_old_subtract_and_copy_back_on_new(
    const TensorMeshHierarchy<N, Real> &hierarchy, Real *const subtrahend,
    Real *const minuend, const std::size_t l) {
  for (const TensorNode<N> node : hierarchy.nodes(l)) {
    // Note that we may also write to a value of `subtrahend`.
    Real &out = hierarchy.at(minuend, node.multiindex);
    out = hierarchy.date_of_birth(node.multiindex) == l
              ? (hierarchy.at(subtrahend, node.multiindex) -= out)
              : 0;
  }
}

} // namespace

template <std::size_t N, typename Real>
void decompose(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v) {
  const std::size_t ndof = hierarchy.ndof();
  Real *const buffer = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  for (std::size_t l = hierarchy.L; l > 0; --l) {
    // We start with `Q_{l}u` on `nodes(l)` of `v`. First we copy the values on
    // `old_nodes(l)` to `buffer`. At the same time, we zero the values on
    // `new_nodes(l)` of `buffer` in preparation for the interpolation routine.
    copy_on_old_zero_on_new(hierarchy, v, buffer, l);
    // Now we have `Π_{l - 1}Q_{l}u` on `old_nodes(l)` of `buffer` and zeros on
    // `new_nodes(l)` of `buffer`. Time to interpolate.
    {
      const TensorProlongationAddition<N, Real> PA(hierarchy, l);
      PA(buffer);
    }
    // Now we have `Π_{l - 1}Q_{l}u` on `nodes(l)` (that is, on both
    // `old_nodes(l)` and `new_nodes(l)`) of `buffer`. `Q_{l}u` is still on
    // `nodes(l)` of `v`. We want to end up with
    //     1. `(I - Π_{l - 1})Q_{l}u` on `new_nodes(l)` of `v` and
    //     2. `(I - Π_{l - 1})Q_{l}u` on `nodes(l)` of `v`.
    // So, we will subtract the values on `new_nodes(l)` of `buffer` from the
    // values on `new_nodes(l)` of `v`, store the difference in both `buffer`
    // and `v`, and also zero the values on `old_nodes(l)` of `buffer`.
    zero_on_old_subtract_and_copy_back_on_new(hierarchy, v, buffer, l);
    // Now we have `(I - Π_{l - 1})Q_{l}u` on `nodes(l)` of `buffer`. Time to
    // project.
    {
      const TensorMassMatrix<N, Real> M(hierarchy, l);
      const TensorRestriction<N, Real> R(hierarchy, l);
      const TensorMassMatrixInverse<N, Real> m_inv(hierarchy, l - 1);
      M(buffer);
      R(buffer);
      m_inv(buffer);
    }
    // Now we have `Q_{l - 1}u - Π_{l - 1}Q_{l}u` on `old_nodes(l)` of `buffer`.
    // Time to correct `Π_{l - 1}Q_{l}u` on `old_nodes(l)` of `v`.
    add_on_old_add_on_new(hierarchy, buffer, v, l - 1);
    // Now we have `(I - Π_{l - 1})Q_{l}u` on `new_nodes(l)` of `v` and
    // `Q_{l - 1}u` on `old_nodes(l)` of `v`.
  }
  std::free(buffer);
}

template <std::size_t N, typename Real>
void recompose(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v) {
  const std::size_t ndof = hierarchy.ndof();
  Real *const buffer = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  for (std::size_t l = 1; l <= hierarchy.L; ++l) {
    // We start with `Q_{l - 1}u` on `old_nodes(l)` of `v` and
    // `(I - Π_{l - 1})Q_{l}u` on `new_nodes(l)` of `v`. We begin by copying
    // `(I - Π_{l - 1})Q_{l}u` to `buffer`.
    // I think we could instead copy all of `v` to `buffer` at the beginning and
    // then just zero `old_nodes(l)` of `buffer` here.
    zero_on_old_copy_on_new(hierarchy, v, buffer, l);
    // Now we have `(I - Π_{l - 1})Q_{l}u` on `nodes(l)` of `buffer`. Time to
    // project.
    {
      const TensorMassMatrix<N, Real> M(hierarchy, l);
      const TensorRestriction<N, Real> R(hierarchy, l);
      const TensorMassMatrixInverse<N, Real> m_inv(hierarchy, l - 1);
      M(buffer);
      R(buffer);
      m_inv(buffer);
    }
    // Now we have `Q_{l - 1}u - Π_{l - 1}Q_{l}u` on `old_nodes(l)` of `buffer`.
    // We can subtract `Q_{l - 1}u` (on `old_nodes(l)` of `v`) to obtain
    // `-Π_{l - 1}Q_{l}u`.
    subtract_on_old_zero_on_new(hierarchy, v, buffer, l);
    // Now we have `-Π_{l - 1}Q_{l}u` on `old_nodes(l)` of buffer. In addition,
    // we have zeros on `new_nodes(l)` of buffer, so we're ready to use
    // `TensorProlongationAddition`.
    {
      const TensorProlongationAddition<N, Real> PA(hierarchy, l);
      PA(buffer);
    }
    // Now we have `-Π_{l - 1}Q_{l}u` on `nodes(l)` of `buffer`. Subtracting
    // from `(I - Π_{l - 1})Q_{l}u`, we'll recover the projection.
    copy_negation_on_old_subtract_on_new(hierarchy, buffer, v, l);
    // Now we have `Q_{l}u` on `nodes(l)` of `v`.
  }
  std::free(buffer);
}

} // end namespace mgard

#endif
