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

} // namespace cli

namespace TCLAP {

template <> struct ArgTraits<cli::DataShape> {
  typedef StringLike ValueCategory;
};

} // namespace TCLAP

namespace cli {

//! Arguments for the compression subcommand.
struct CompressionArguments {

  //! Constructor.
  //!
  //!\param datatype Type of the dataset.
  //!\param shape Shape of the dataset.
  //!\param input Filename of the input dataset.
  //!\param smoothness Smoothness parameter to use in compression.
  //!\param tolerance Error tolerance to use in compression.
  //!\param output Filename of the output archive.
  CompressionArguments(TCLAP::ValueArg<std::string> &datatype,
                       TCLAP::ValueArg<DataShape> &shape,
                       TCLAP::ValueArg<std::string> &input,
                       TCLAP::ValueArg<double> &smoothness,
                       TCLAP::ValueArg<double> &tolerance,
                       TCLAP::ValueArg<std::string> &output);

  //! Type of the dataset.
  std::string datatype;

  //! Shape of the dataset.
  std::vector<std::size_t> shape;

  //! Spatial dimension of the dataset.
  std::size_t dimension;

  //! Filenames of the coordinates of the nodes in each dimension.
  std::vector<std::string> coordinate_filenames;

  //! Filename of the input dataset.
  std::string input;

  //! Smoothness parameter to use in compression.
  double s;

  //! Error tolerance to use in compression.
  double tolerance;

  //! Filename of the output archive.
  std::string output;
};

//! Arguments for the decompression subcommand.
struct DecompressionArguments {
  //! Constructor.
  //!
  //!\param input Filename of the input archive.
  //!\param input Filename of the output dataset.
  DecompressionArguments(TCLAP::ValueArg<std::string> &input,
                         TCLAP::ValueArg<std::string> &output);

  //! Filename of the input archive.
  std::string input;

  //! Filename of the output dataset.
  std::string output;
};

} // namespace cli

#endif
