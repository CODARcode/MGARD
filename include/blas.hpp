#ifndef BLAS_HPP
#define BLAS_HPP
//!\file
//!\brief Level 1 BLAS functions needed for MGARD.

#include <cstddef>

namespace blas {

//!Find the (unconjugated) dot product of two vectors.
//!
//!\param [in] N Size of vectors.
//!\param [in] p First vector.
//!\param [in] q Second vector.
template <typename Real>
Real dotu(const std::size_t N, Real const *p, Real const *q);

//!Find the Euclidean norm of a vector.
//!
//!\param [in] N Size of the vector.
//!\param [in] p Vector to be measured.
template <typename Real>
Real nrm2(const std::size_t N, Real const *p);

template <typename Real>
void axpy(const std::size_t N, const Real alpha, Real const *p, Real *q);

//!Copy one vector to another.
//!
//!\param [in] N Size of the vectors.
//!\param [in] p Source vector.
//!\param [out] q Destination vector.
template <typename Real>
void copy(const std::size_t N, Real const *p, Real *q);

//!Scale a vector by a constant.
//!
//!\param [in] N Size of the vector.
//!\param [in] alpha Constant by which to scale the vector.
//!\param [in, out] p Vector to be scaled.
template <typename Real>
void scal(const std::size_t N, Real const alpha, Real *p);

}

#include "blas.tpp"
#endif
