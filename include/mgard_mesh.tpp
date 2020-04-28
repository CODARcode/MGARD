#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mgard {

template <std::size_t N>
Dimensions2kPlus1<N>::Dimensions2kPlus1(const std::array<int, N> input_) {
  nlevel = std::numeric_limits<int>::max();
  bool nlevel_never_set = true;
  for (std::size_t i = 0; i < N; ++i) {
    const int size = input.at(i) = input_.at(i);
    if (size <= 0) {
      throw std::domain_error(
        "dataset must have size larger than 0 in every dimension"
      );
    } else if (size > 1) {
      const int exp = nlevel_from_size(size);
      rnded.at(i) = size_from_nlevel(exp);
      nlevel = std::min(nlevel, exp);
      nlevel_never_set = false;
    } else {
      rnded.at(i) = size;
    }
  }
  if (nlevel_never_set) {
    throw std::domain_error(
      "dataset must have size larger than 1 in some dimension"
    );
  }
}

template <std::size_t N> bool Dimensions2kPlus1<N>::is_2kplus1() const {
  for (const int n : input) {
    if (!(n == 1 || n == size_from_nlevel(nlevel_from_size(n)))) {
      return false;
    }
  }
  return true;
}

template <std::size_t N>
template <typename Real>
RangeSlice<LevelValuesIterator<N, Real>>
Dimensions2kPlus1<N>::on_nodes(Real *const coefficients,
                               const std::size_t index_difference) const {
  const std::array<std::size_t, N> begin_indices{};
  std::array<std::size_t, N> end_indices{};
  end_indices.at(0) = input.at(0);
  return {.begin_ = LevelValuesIterator<N, Real>(
              *this, coefficients, index_difference, begin_indices),
          .end_ = LevelValuesIterator<N, Real>(*this, coefficients,
                                               index_difference, end_indices)};
}

template <std::size_t N>
bool operator==(const Dimensions2kPlus1<N> &a, const Dimensions2kPlus1<N> &b) {
  return a.input == b.input;
}

template <std::size_t N>
bool operator!=(const Dimensions2kPlus1<N> &a, const Dimensions2kPlus1<N> &b) {
  return !operator==(a, b);
}

template <std::size_t N, typename Real>
LevelValuesIterator<N, Real>::LevelValuesIterator(
    const Dimensions2kPlus1<N> &dimensions, Real *const coefficients,
    const std::size_t index_difference,
    const std::array<std::size_t, N> &indices)
    : dimensions(dimensions), coefficients(coefficients),
      stride(stride_from_index_difference(index_difference)), indices(indices) {
}

template <std::size_t N, typename Real>
bool LevelValuesIterator<N, Real>::
operator==(const LevelValuesIterator<N, Real> &other) const {
  return dimensions == other.dimensions && coefficients == other.coefficients &&
         stride == other.stride && indices == other.indices;
}

template <std::size_t N, typename Real>
bool LevelValuesIterator<N, Real>::
operator!=(const LevelValuesIterator<N, Real> &other) const {
  return !operator==(other);
}

template <std::size_t N, typename Real>
LevelValuesIterator<N, Real> &LevelValuesIterator<N, Real>::operator++() {
  std::size_t i = N;
  while (i != 0) {
    const std::size_t j = i - 1;
    std::size_t &index = indices.at(j);
    index += stride;
    if (index < static_cast<std::size_t>(dimensions.input.at(j))) {
      break;
    } else {
      index = 0;
      --i;
    }
  }
  if (i == 0) {
    // We hit the end. Change the first index so we can distinguish from the
    // beginning.
    indices.at(0) = dimensions.input.at(0);
  }
  return *this;
}

template <std::size_t N, typename Real>
LevelValuesIterator<N, Real> LevelValuesIterator<N, Real>::operator++(int) {
  const LevelValuesIterator tmp = *this;
  operator++();
  return tmp;
}

template <std::size_t N, typename Real>
Real &LevelValuesIterator<N, Real>::operator*() const {
  std::size_t index = indices.at(0);
  for (std::size_t i = 1; i < N; ++i) {
    index *= dimensions.input.at(i);
    index += indices.at(i);
  }
  return coefficients[index];
}

} // namespace mgard
