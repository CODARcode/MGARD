#include <stdexcept>

namespace mgard {

template <typename T, std::size_t N>
MultidimensionalArray<T, N>::MultidimensionalArray(
    T *const data, const std::array<std::size_t, N> dimensions,
    const std::size_t stride)
    : data(data), dimensions(dimensions), stride(stride) {
  if (!stride) {
    throw std::invalid_argument("stride must be positive");
  }
}

template <typename T, std::size_t N>
std::size_t MultidimensionalArray<T, N>::size() const {
  std::size_t n = 1;
  for (const std::size_t d : dimensions) {
    n *= d;
  }
  return n;
}

template <typename T, std::size_t N>
T &MultidimensionalArray<T, N>::at(
    const std::array<std::size_t, N> multiindex) const {
  check_multiindex_bounds(multiindex);
  std::size_t index = multiindex.at(0);
  for (std::size_t i = 1; i < N; ++i) {
    index *= dimensions.at(i);
    index += multiindex.at(i);
  }
  return data[stride * index];
}

template <typename T, std::size_t N>
void MultidimensionalArray<T, N>::check_multiindex_bounds(
    const std::array<std::size_t, N> multiindex) const {
  for (std::size_t i = 0; i < N; ++i) {
    if (multiindex.at(i) >= dimensions.at(i)) {
      throw std::out_of_range("multiindex entry too large");
    }
  }
}

} // namespace mgard
