#ifndef UTILITIES_HPP
#define UTILITIES_HPP
//!\file
//!\brief Utilities for use in the MGARD implementation.

#include <cstddef>

namespace helpers {

//!Mimic an array for range-based for loops.
template <typename T>
class PseudoArray {
    public:
        //!Constructor.
        //!
        //!\param p Pointer to the first element in the array.
        //!\param N Length of the array.
        PseudoArray(T * const p, const std::size_t N);

        //!Constructor.
        //!
        //!\override
        PseudoArray(T * const p, const int N);

        //!Return an iterator to the beginning of the array.
        T * begin() const;

        //!Return an iterator to the end of the array.
        T * end() const;

    private:
        T * const p;
        const std::size_t N;
};

}

#include "utilities.tpp"
#endif
