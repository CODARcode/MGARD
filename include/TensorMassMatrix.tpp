namespace mgard {

template <std::size_t N, typename Real>
ConstituentMassMatrix<N, Real>::ConstituentMassMatrix(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l,
    const std::size_t dimension)
    : ConstituentLinearOperator<N, Real>(hierarchy, l, dimension) {
  if (this->dimension() < 2) {
    throw std::invalid_argument("mass matrix implementation assumes that "
                                "'spear' has at least two nodes");
  }
}

template <std::size_t N, typename Real>
void ConstituentMassMatrix<N, Real>::do_operator_parentheses(
    const std::array<std::size_t, N> multiindex, Real *const v) const {
  std::array<std::size_t, N> alpha = multiindex;
  std::size_t &variable_index = alpha.at(CLO::dimension_);
  const std::vector<Real> &xs = CLO::hierarchy.coordinates.at(CLO::dimension_);
  const std::size_t n = CLO::dimension();

  // Node coordinates.
  Real x_middle;
  Real x_right;

  // Node spacings.
  Real h_left;
  Real h_right;

  // Function values at nodes.
  Real v_left;
  Real v_middle;
  Real v_right;

  // Pointers to use when overwriting input array.
  // TODO: Hoping to replace these with a function in the iterator.
  Real *out_middle;
  Real *out_right;

  std::vector<std::size_t>::const_iterator p = CLO::indices.begin();
  std::size_t i;

  // TODO: Maybe a combined sort of iterator here. Increment `p` index and
  // automatically change `alpha`, `v`, `x`, and whatever else.

  variable_index = i = *p++;
  x_middle = xs.at(i);
  out_middle = &CLO::hierarchy.at(v, alpha);
  v_middle = *out_middle;

  variable_index = i = *p++;
  x_right = xs.at(i);
  h_right = x_right - x_middle;
  out_right = &CLO::hierarchy.at(v, alpha);
  v_right = *out_right;

  // TODO: Figure out changing.
  *out_middle = h_right / 3 * v_middle + h_right / 6 * v_right;

  // TODO: Careful. We've already incremented twice. Check the limits.
  for (std::size_t j = 2; j < n; ++j) {
    variable_index = i = *p++;

    x_middle = x_right;
    h_left = h_right;
    v_left = v_middle;
    v_middle = v_right;
    out_middle = out_right;

    x_right = xs.at(i);
    h_right = x_right - x_middle;
    out_right = &CLO::hierarchy.at(v, alpha);
    v_right = *out_right;

    // TODO: Figure out changing.
    *out_middle = h_left / 6 * v_left + (h_left + h_right) / 3 * v_middle +
                  h_right / 6 * v_right;
  }

  x_middle = x_right;
  h_left = h_right;
  v_left = v_middle;
  v_middle = v_right;
  out_middle = out_right;

  // TODO: Figure out changing.
  *out_middle = h_left / 6 * v_left + h_left / 3 * v_middle;
}

} // namespace mgard
