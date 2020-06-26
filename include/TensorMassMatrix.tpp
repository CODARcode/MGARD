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
  const std::vector<Real> &xs = CLO::hierarchy->coordinates.at(CLO::dimension_);
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
  out_middle = &CLO::hierarchy->at(v, alpha);
  v_middle = *out_middle;

  variable_index = i = *p++;
  x_right = xs.at(i);
  h_right = x_right - x_middle;
  out_right = &CLO::hierarchy->at(v, alpha);
  v_right = *out_right;

  // TODO: Figure out changing.
  *out_middle = h_right / 3 * v_middle + h_right / 6 * v_right;

  // TODO: Careful. We've already incremented twice. Check the limits.
  for (std::size_t j = 2; j < n; ++j) {
    // `j` isn't the index of anything – just a count.
    variable_index = i = *p++;

    x_middle = x_right;
    h_left = h_right;
    v_left = v_middle;
    v_middle = v_right;
    out_middle = out_right;

    x_right = xs.at(i);
    h_right = x_right - x_middle;
    out_right = &CLO::hierarchy->at(v, alpha);
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

namespace {

template <std::size_t N, typename Real>
std::array<ConstituentMassMatrix<N, Real>, N>
generate_mass_matrices(const TensorMeshHierarchy<N, Real> &hierarchy,
                       const std::size_t l) {
  std::array<ConstituentMassMatrix<N, Real>, N> mass_matrices;
  for (std::size_t i = 0; i < N; ++i) {
    mass_matrices.at(i) = ConstituentMassMatrix<N, Real>(hierarchy, l, i);
  }
  return mass_matrices;
}

} // namespace

template <std::size_t N, typename Real>
TensorMassMatrix<N, Real>::TensorMassMatrix(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l)
    : TensorLinearOperator<N, Real>(hierarchy, l),
      mass_matrices(generate_mass_matrices(hierarchy, l)) {
  for (std::size_t i = 0; i < N; ++i) {
    TLO::operators.at(i) = &mass_matrices.at(i);
  }
}

template <std::size_t N, typename Real>
ConstituentMassMatrixInverse<N, Real>::ConstituentMassMatrixInverse(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l,
    const std::size_t dimension, Real *const buffer)
    : ConstituentLinearOperator<N, Real>(hierarchy, l, dimension),
      divisors(buffer) {
  if (this->dimension() < 2) {
    throw std::invalid_argument("mass matrix inverse implementation assumes "
                                "that 'spear' has at least two nodes");
  }
}

template <std::size_t N, typename Real>
void ConstituentMassMatrixInverse<N, Real>::do_operator_parentheses(
    const std::array<std::size_t, N> multiindex, Real *const v) const {
  // The system is solved using the Thomas algorithm. See <https://
  // en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm>. In case the article
  // changes, the algorithm is copied here.
  //    // [b_1, …, b_n] is the diagonal of the matrix. `[a_2, …, a_n]` is the
  //    // subdiagonal, and `[c_1, …, c_{n - 1}]` is the superdiagonal.
  //    // `[d_1, …, d_n]` is the righthand side, and `[x_1, …, x_n]` is the
  //    // solution.
  //    for i = 2, …, n do
  //      w_i = a_i / b_{i - 1}
  //      b_i := b_i - w_i * c_{i - 1}
  //      d_i := d_i - w_i * d_{i - 1}
  //    end
  //    x_n = d_n / b_n
  //    for i = n - 1, …, 1 do
  //      x_i = (d_i - c_i * x_{i + 1}) / b_i
  //    end
  // Because the mass matrix is symmetric, `a_i` is equal to `c_{i - 1}`.
  std::array<std::size_t, N> alpha = multiindex;
  std::size_t &variable_index = alpha.at(CLO::dimension_);
  const std::vector<Real> &xs = CLO::hierarchy->coordinates.at(CLO::dimension_);
  const std::size_t n = CLO::dimension();

  // Node coordinates.
  Real x_middle;
  Real x_right;

  // Node spacings.
  Real h_left;
  Real h_right;

  // Pointers to use when overwriting input array.
  Real *out_middle;
  Real *out_right;

  // Previous value of the input array *after* overwriting (in the algorithm
  // above, `d_{i - 1}` when we're updating `d_i`).
  Real rhs_previous;
  // Solution entry computed in the previous iteration (in the algorithm above,
  // `x_{i + 1}` when we're updating `x_i`.)
  Real x_next;

  std::vector<std::size_t>::const_iterator p = CLO::indices.begin();
  std::size_t i;

  variable_index = i = *p;
  x_middle = xs.at(i);
  out_middle = &CLO::hierarchy->at(v, alpha);

  variable_index = i = *++p;
  x_right = xs.at(i);
  out_right = &CLO::hierarchy->at(v, alpha);
  h_right = x_right - x_middle;

  divisors[0] = 2 * h_right / 6;
  rhs_previous = *out_middle;

  // Forward sweep (except for last entry).
  for (std::size_t j = 1; j + 1 < n; ++j) {
    // `j` is the index of the current ('middle') row.
    variable_index = i = *++p;

    x_middle = x_right;
    out_middle = out_right;
    h_left = h_right;

    x_right = xs.at(i);
    out_right = &CLO::hierarchy->at(v, alpha);
    h_right = x_right - x_middle;

    // Subdiagonal element `a_i` in the current row, equal to the superdiagonal
    // element `c_{i - 1}` in the previous row.
    const Real a_j = h_left / 6;
    const Real w = a_j / divisors[j - 1];
    // In general (for example, if the matrix weren't symmetric), `a_i` here is
    // `c_{i - 1}`.
    divisors[j] = 2 * (h_left + h_right) / 6 - w * a_j;
    rhs_previous = *out_middle -= w * rhs_previous;
  }

  // Forward sweep (last entry).
  {
    x_middle = x_right;
    out_middle = out_right;
    h_left = h_right;
    const Real a_j = h_left / 6;
    const Real w = a_j / divisors[n - 2];
    divisors[n - 1] = 2 * h_left / 6 - w * a_j;
    // TODO: We are done with `rhs_previous`, so we don't update it.
    *out_middle -= w * rhs_previous;
  }

  // Start of backward sweep (first entry).
  { x_next = *out_middle /= divisors[n - 1]; }

  // TODO: Remove.
  {
    std::vector<std::size_t>::const_iterator q = p;
    if (++q != CLO::indices.end()) {
      throw std::logic_error("you miscounted indices");
    }
  }

  // Up to now (apart from its very first usage), `p` has pointed to the 'right'
  // index. From now on it will point to the 'middle' index.

  // Backward sweep (remaining entries).
  for (std::size_t k = 2; k <= n; ++k) {
    const std::size_t j = n - k;
    variable_index = i = *--p;

    x_right = x_middle;

    x_middle = xs.at(i);
    out_middle = &CLO::hierarchy->at(v, alpha);
    h_right = x_right - x_middle;

    const Real c_j = h_right / 6;
    *out_middle -= c_j * x_next;
    x_next = *out_middle /= divisors[j];
  }
}

} // namespace mgard
