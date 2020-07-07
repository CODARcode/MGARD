#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mgard {

template <std::size_t N>
Dimensions2kPlus1<N>::Dimensions2kPlus1(
    const std::array<std::size_t, N> input_) {
  nlevel = std::numeric_limits<std::size_t>::max();
  bool nlevel_never_set = true;
  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t size = input.at(i) = input_.at(i);
    if (size == 0) {
      throw std::domain_error(
          "dataset must have size larger than 0 in every dimension");
    } else if (size == 1) {
      rnded.at(i) = size;
    } else {
      const std::size_t exp = nlevel_from_size(size);
      rnded.at(i) = size_from_nlevel(exp);
      nlevel = std::min(nlevel, exp);
      nlevel_never_set = false;
    }
  }
  if (nlevel_never_set) {
    throw std::domain_error(
        "dataset must have size larger than 1 in some dimension");
  }
}

template <std::size_t N> bool Dimensions2kPlus1<N>::is_2kplus1() const {
  for (const std::size_t n : input) {
    if (!(n == 1 || n == size_from_nlevel(nlevel_from_size(n)))) {
      return false;
    }
  }
  return true;
}

template <std::size_t N>
bool operator==(const Dimensions2kPlus1<N> &a, const Dimensions2kPlus1<N> &b) {
  return a.input == b.input;
}

template <std::size_t N>
bool operator!=(const Dimensions2kPlus1<N> &a, const Dimensions2kPlus1<N> &b) {
  return !operator==(a, b);
}

} // namespace mgard
