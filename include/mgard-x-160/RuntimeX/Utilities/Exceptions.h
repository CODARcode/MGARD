/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 */

#ifndef MGARD_X_160_EXCEPTIONS_H
#define MGARD_X_160_EXCEPTIONS_H

#include <stdexcept>
#include <string>

#include "../../Utilities/Types.h"

namespace mgard_x_160 {

//! Base class for all exceptions thrown by the MGARD-X library.
//!
//! Errors that are internal to the (de)compression process are reported by
//! throwing an exception rather than calling `exit()`, so that the calling
//! application can handle the failure gracefully (clean up, report to a job
//! scheduler, keep other threads/GPUs alive, etc.). Every exception carries a
//! `compress_status_type` so the high-level API can translate a thrown
//! exception back into the status code returned by `compress()`/`decompress()`.
class Exception : public std::runtime_error {
public:
  explicit Exception(const std::string &msg, compress_status_type status =
                                                 compress_status_type::Failure)
      : std::runtime_error(msg), status_(status) {}

  //! Status code that the high-level API should return for this exception.
  compress_status_type status() const noexcept { return status_; }

private:
  compress_status_type status_;
};

//! Thrown when compressed data or its metadata header is malformed,
//! corrupted, or was produced by an incompatible version of MGARD.
class InvalidDataException : public Exception {
public:
  explicit InvalidDataException(const std::string &msg) : Exception(msg) {}
};

//! Thrown when an internal processing step fails (e.g. a lossless backend
//! error or a Huffman codebook that cannot be represented).
class ProcessingException : public Exception {
public:
  explicit ProcessingException(const std::string &msg) : Exception(msg) {}
};

} // namespace mgard_x_160

#endif // MGARD_X_160_EXCEPTIONS_H
