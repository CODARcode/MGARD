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

#include <algorithm>
#include <functional>
#include <numeric>

#ifdef MGARD_TIMING
#include <chrono>
#endif

#include "mgard_compress.hpp"

#include "TensorMassMatrix.hpp"
#include "TensorProlongation.hpp"
#include "TensorRestriction.hpp"
#include "blas.hpp"
#include "shuffle.hpp"

namespace mgard {

namespace {

// Not documenting the parameters here. I think it'll be easiest to understand
// by reading the code.

template <std::size_t N, typename Real>
void add_on_old_add_on_new(const TensorMeshHierarchy<N, Real> &hierarchy,
                           Real const *const src, Real *const dst,
                           const std::size_t l) {
  const PseudoArray<const Real> src_on_l = hierarchy.on_nodes(src, l);
  const PseudoArray<Real> dst_on_l = hierarchy.on_nodes(dst, l);
  blas::axpy(src_on_l.size, static_cast<Real>(1), src_on_l.data, dst_on_l.data);
}

template <std::size_t N, typename Real>
void subtract_on_old_zero_on_new(const TensorMeshHierarchy<N, Real> &hierarchy,
                                 Real const *const src, Real *const dst,
                                 const std::size_t l) {
  {
    const PseudoArray<const Real> src_on_old = hierarchy.on_nodes(src, l - 1);
    const PseudoArray<Real> dst_on_old = hierarchy.on_nodes(dst, l - 1);
    blas::axpy(src_on_old.size, static_cast<Real>(-1), src_on_old.data,
               dst_on_old.data);
  }
  {
    const PseudoArray<Real> dst_on_new = hierarchy.on_new_nodes(dst, l);
    std::fill(dst_on_new.begin(), dst_on_new.end(), 0);
  }
}

template <std::size_t N, typename Real>
void copy_negation_on_old_subtract_on_new(
    const TensorMeshHierarchy<N, Real> &hierarchy, Real const *const src,
    Real *const dst, const std::size_t l) {
  {
    const PseudoArray<const Real> src_on_old = hierarchy.on_nodes(src, l - 1);
    const PseudoArray<Real> dst_on_old = hierarchy.on_nodes(dst, l - 1);
    std::transform(src_on_old.begin(), src_on_old.end(), dst_on_old.begin(),
                   std::negate<Real>());
  }
  {
    const PseudoArray<const Real> src_on_new = hierarchy.on_new_nodes(src, l);
    const PseudoArray<Real> dst_on_new = hierarchy.on_new_nodes(dst, l);
    blas::axpy(src_on_new.size, static_cast<Real>(-1), src_on_new.data,
               dst_on_new.data);
  }
}

template <std::size_t N, typename Real>
void copy_on_old_zero_on_new(const TensorMeshHierarchy<N, Real> &hierarchy,
                             Real const *const src, Real *const dst,
                             const std::size_t l) {
  {
    const PseudoArray<const Real> src_on_old = hierarchy.on_nodes(src, l - 1);
    const PseudoArray<Real> dst_on_old = hierarchy.on_nodes(dst, l - 1);
    blas::copy(src_on_old.size, src_on_old.data, dst_on_old.data);
  }
  {
    const PseudoArray<Real> dst_on_new = hierarchy.on_new_nodes(dst, l);
    std::fill(dst_on_new.begin(), dst_on_new.end(), 0);
  }
}

template <std::size_t N, typename Real>
void zero_on_old_copy_on_new(const TensorMeshHierarchy<N, Real> &hierarchy,
                             Real const *const src, Real *const dst,
                             const std::size_t l) {
  {
    const PseudoArray<Real> dst_on_old = hierarchy.on_nodes(dst, l - 1);
    std::fill(dst_on_old.begin(), dst_on_old.end(), 0);
  }
  {
    const PseudoArray<const Real> src_on_new = hierarchy.on_new_nodes(src, l);
    const PseudoArray<Real> dst_on_new = hierarchy.on_new_nodes(dst, l);
    blas::copy(src_on_new.size, src_on_new.data, dst_on_new.data);
  }
}

template <std::size_t N, typename Real>
void zero_on_old_subtract_and_copy_back_on_new(
    const TensorMeshHierarchy<N, Real> &hierarchy, Real *const subtrahend,
    Real *const minuend, const std::size_t l) {
  {
    const PseudoArray<Real> minuend_on_old = hierarchy.on_nodes(minuend, l - 1);
    std::fill(minuend_on_old.begin(), minuend_on_old.end(), 0);
  }
  {
    const PseudoArray<Real> subtrahend_on_new =
        hierarchy.on_new_nodes(subtrahend, l);
    const PseudoArray<Real> minuend_on_new = hierarchy.on_new_nodes(minuend, l);
    Real *p = minuend_on_new.begin();
    for (Real &subtrahend_value : subtrahend_on_new) {
      Real &minuend_value = *p++;
      minuend_value = (subtrahend_value -= minuend_value);
    }
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
