#include <stdexcept>

namespace helpers {

template <typename T>
PseudoArray<T>::PseudoArray(T * const p, const std::size_t N):
    p(p),
    N(N)
{
}

template <typename T>
PseudoArray<T>::PseudoArray(T * const p, const int _N):
    PseudoArray<T>::PseudoArray(p, static_cast<std::size_t>(_N))
{
    if (_N < 0) {
        throw std::invalid_argument("length must be nonzero");
    }
}

template <typename T>
T * PseudoArray<T>::begin() const {
    return p;
}

template <typename T>
T * PseudoArray<T>::end() const {
    return p + N;
}

}
