#include <algorithm>
#include <limits>

namespace mgard {

template <std::size_t N>
Dimensions2kPlus1<N>::Dimensions2kPlus1(const std::array<int, N> input_) {
  nlevel = std::numeric_limits<int>::max();
  for (std::size_t i = 0; i < N; ++i) {
    const int exp = nlevel_from_size(input.at(i) = input_.at(i));
    rnded.at(i) = size_from_nlevel(exp);
    nlevel = std::min(nlevel, (nlevels.at(i) = exp));
  }
}

} // namespace mgard
