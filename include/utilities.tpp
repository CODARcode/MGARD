#include <stdexcept>

namespace mgard {

template <typename T>
PseudoArray<T>::PseudoArray(T * const data, const std::size_t size):
    data(data),
    size(size)
{
}

template <typename T>
PseudoArray<T>::PseudoArray(T * const data, const int _size):
    PseudoArray<T>::PseudoArray(data, static_cast<std::size_t>(_size))
{
    if (_size < 0) {
        throw std::invalid_argument("size must be nonnegative");
    }
}

template <typename T>
T * PseudoArray<T>::begin() const {
    return data;
}

template <typename T>
T * PseudoArray<T>::end() const {
    return data + size;
}

template <typename T>
T PseudoArray<T>::operator[](const std::size_t i) const {
    if (i >= size) {
        throw std::out_of_range("index too large");
    }
    return data[i];
}

}
