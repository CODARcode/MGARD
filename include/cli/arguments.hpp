#ifndef SUBCOMMAND_ARGUMENTS
#define SUBCOMMAND_ARGUMENTS
//!\file
//!\brief Arguments for the CLI subcommands.

#include <cstddef>

#include <string>
#include <vector>

#include <tclap/CmdLine.h>

namespace cli {

//! Shape of the dataset.
struct DataShape {
  //! Assignment operator.
  DataShape &operator=(const std::string &value);

  //! Shape of the dataset.
  std::vector<std::size_t> shape;
};

//! Smoothness parameter (`s`) used in compressing the dataset.
template <typename Real> class SmoothnessParameter {
public:
  //! Constructor.
  //!
  //!\param s Smoothness parameter.
  SmoothnessParameter(const Real s);

  //! Conversion operator.
  operator Real() const;

public:
  //! Value.
  Real s;
};

//! Parse a command line argument as a `SmoothnessParameter`.
template <typename Real>
std::istringstream &operator>>(std::istringstream &stream,
                               SmoothnessParameter<Real> &s) {
  if (stream.str() == "inf") {
    s = std::numeric_limits<Real>::infinity();
    std::string tmp;
    stream >> tmp;
  } else {
    Real tmp;
    stream >> tmp;
    s = tmp;
  }
  return stream;
}

} // namespace cli

namespace TCLAP {

//! Traits for `cli::DataShape`.
template <> struct ArgTraits<cli::DataShape> {
  //! Value category for `cli::DataShape`.
  typedef StringLike ValueCategory;
};

//! Traits for `cli::SmoothnessParameter<double>`.
template <> struct ArgTraits<cli::SmoothnessParameter<double>> {
  //! Value category for `cli::SmoothnessParameter<double>`.
  typedef ValueLike ValueCategory;
};

} // namespace TCLAP

namespace cli {

//! Arguments for the compression subcommand.
struct CompressionArguments {

  //! Constructor.
  //!
  //!\param datatype Type of the dataset.
  //!\param shape Shape of the dataset.
  //!\param smoothness Smoothness parameter to use in compression.
  //!\param tolerance Error tolerance to use in compression.
  //!\param input Filename of the input dataset.
  //!\param output Filename of the output buffer.
  CompressionArguments(
      const TCLAP::ValueArg<std::string> &datatype,
      const TCLAP::ValueArg<DataShape> &shape,
      const TCLAP::ValueArg<cli::SmoothnessParameter<double>> &smoothness,
      const TCLAP::ValueArg<double> &tolerance,
      const TCLAP::ValueArg<std::string> &input,
      const TCLAP::ValueArg<std::string> &output);

  //! Type of the dataset.
  std::string datatype;

  //! Shape of the dataset.
  std::vector<std::size_t> shape;

  //! Spatial dimension of the dataset.
  std::size_t dimension;

  //! Smoothness parameter to use in compression.
  double s;

  //! Error tolerance to use in compression.
  double tolerance;

  //! Filename of the input dataset.
  std::string input;

  //! Filename of the output buffer.
  std::string output;
};

//! Arguments for the decompression subcommand.
struct DecompressionArguments {
  //! Constructor.
  //!
  //!\param input Filename of the input buffer.
  //!\param output Filename of the output dataset.
  DecompressionArguments(const TCLAP::ValueArg<std::string> &input,
                         const TCLAP::ValueArg<std::string> &output);

  //! Filename of the input buffer.
  std::string input;

  //! Filename of the output dataset.
  std::string output;
};

} // namespace cli

#include "cli/arguments.tpp"
#endif
