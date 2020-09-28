#include <cassert>

namespace mgard {

template <std::size_t N, typename Real>
ConstituentRestriction<N, Real>::ConstituentRestriction(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l,
    const std::size_t dimension)
    : ConstituentLinearOperator<N, Real>(hierarchy, l, dimension),
      coarse_indices(hierarchy.indices(l - 1, dimension)) {
  // This is almost certainly superfluous, since `hierarchy.indices` checks that
  // `l - 1` is a valid mesh index.
  if (!l) {
    throw std::invalid_argument("cannot restrict from the coarsest level");
  }
  // Possibly we should check that neither `coarse_indices` not `indices` is
  // empty, since we will dereference `coarse_indices.begin()` and
  // `indices.begin()`. Assuming I haven't made a mistake, though, this is
  // enforced by the constructor of `TensorIndexRange` called by
  // `TensorMeshHierarchy::indices`. Possibly fragile.
}

template <std::size_t N, typename Real>
void ConstituentRestriction<N, Real>::do_operator_parentheses(
    const std::array<std::size_t, N> multiindex, Real *const v) const {
  const std::vector<Real> &xs = CLO::hierarchy->coordinates.at(CLO::dimension_);

  // `x_left` and `out_left` is declared and defined inside the loop.
  Real x_right;
  Real *out_right;

  std::array<std::size_t, N> alpha = multiindex;
  std::size_t &variable_index = alpha.at(CLO::dimension_);

  TensorIndexRange::iterator p = coarse_indices.begin();
  std::size_t i;

  variable_index = i = *p++;
  x_right = xs.at(i);
  out_right = &CLO::hierarchy->at(v, alpha);

  std::array<std::size_t, N> ALPHA = multiindex;
  std::size_t &VARIABLE_INDEX = ALPHA.at(CLO::dimension_);

  TensorIndexRange::iterator P = CLO::indices.begin();
  std::size_t I;

  VARIABLE_INDEX = I = *P++;

  const TensorIndexRange::iterator p_end = coarse_indices.end();
  while (p != p_end) {
    assert(I == i);

    const Real x_left = x_right;
    Real *const out_left = out_right;

    variable_index = i = *p++;
    x_right = xs.at(i);
    out_right = &CLO::hierarchy->at(v, alpha);

    const Real width_reciprocal = 1 / (x_right - x_left);

    while ((VARIABLE_INDEX = I = *P++) != i) {
      const Real x_middle = xs.at(I);
      assert(x_left < x_middle && x_middle < x_right);
      const Real v_middle = CLO::hierarchy->at(v, ALPHA);
      *out_left += v_middle * (x_right - x_middle) * width_reciprocal;
      *out_right += v_middle * (x_middle - x_left) * width_reciprocal;
    }
  }
}

namespace {

template <std::size_t N, typename Real>
std::array<ConstituentRestriction<N, Real>, N>
generate_restrictions(const TensorMeshHierarchy<N, Real> &hierarchy,
                      const std::size_t l) {
  std::array<ConstituentRestriction<N, Real>, N> restrictions;
  for (std::size_t i = 0; i < N; ++i) {
    restrictions.at(i) = ConstituentRestriction<N, Real>(hierarchy, l, i);
  }
  return restrictions;
}

} // namespace

template <std::size_t N, typename Real>
TensorRestriction<N, Real>::TensorRestriction(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l)
    : TensorLinearOperator<N, Real>(hierarchy, l),
      restrictions(generate_restrictions(hierarchy, l)) {
  for (std::size_t i = 0; i < N; ++i) {
    TLO::operators.at(i) = &restrictions.at(i);
  }
}

} // namespace mgard
