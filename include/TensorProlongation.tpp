#include <cassert>

namespace mgard {

template <std::size_t N, typename Real>
ConstituentProlongationAddition<N, Real>::ConstituentProlongationAddition(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l,
    const std::size_t dimension)
    : ConstituentLinearOperator<N, Real>(hierarchy, l, dimension),
      coarse_indices(hierarchy.indices(l - 1, dimension)) {
  // This is almost certainly superfluous, since `hierarchy.indices` checks that
  // `l - 1` is a valid mesh index.
  if (!l) {
    throw std::invalid_argument("cannot interpolate from the coarsest level");
  }
  // We'll dereference the iterator returned by `indices::begin()`.
  if (!this->dimension()) {
    throw std::invalid_argument("dimension must be nonzero");
  }
}

template <std::size_t N, typename Real>
void ConstituentProlongationAddition<N, Real>::do_operator_parentheses(
    const std::array<std::size_t, N> multiindex, Real *const v) const {
  const std::vector<Real> &xs = CLO::hierarchy->coordinates.at(CLO::dimension_);

  // `x_left` and `v_left` are declared and defined inside the loop.
  Real x_right;
  Real v_right;

  std::array<std::size_t, N> alpha = multiindex;
  std::size_t &variable_index = alpha.at(CLO::dimension_);

  std::vector<std::size_t>::const_iterator p = coarse_indices.begin();
  std::size_t i;

  variable_index = i = *p++;
  x_right = xs.at(i);
  v_right = CLO::hierarchy->at(v, alpha);

  std::array<std::size_t, N> ALPHA = multiindex;
  std::size_t &VARIABLE_INDEX = ALPHA.at(CLO::dimension_);

  std::vector<std::size_t>::const_iterator P = CLO::indices.begin();
  std::size_t I;

  VARIABLE_INDEX = I = *P++;

  const std::vector<std::size_t>::const_iterator p_end = coarse_indices.end();
  while (p != p_end) {
    assert(I == i);

    const Real x_left = x_right;
    const Real v_left = v_right;

    variable_index = i = *p++;
    x_right = xs.at(i);
    v_right = CLO::hierarchy->at(v, alpha);

    const Real width_reciprocal = 1 / (x_right - x_left);

    while ((VARIABLE_INDEX = I = *P++) != i) {
      const Real x_middle = xs.at(I);
      assert(x_left < x_middle && x_middle < x_right);
      CLO::hierarchy->at(v, ALPHA) +=
          (v_left * (x_right - x_middle) + v_right * (x_middle - x_left)) *
          width_reciprocal;
    }
  }
}

} // namespace mgard
