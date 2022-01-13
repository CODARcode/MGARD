#include <tclap/ValueArg.h>
#include <tclap/ValuesConstraint.h>

#include <string>
#include <vector>

#include "MGARDConfig.hpp"

#include "cli/arguments.hpp"
#include "cli/cli_internal.hpp"
#include "cli/cmdline.hpp"

int main(const int argc, char const *const *const argv) {
  cli::SubCmdLine subcompress(MGARD_VERSION_STR,
                              "Compress a dataset using MGARD.");

  TCLAP::ValueArg<std::string> output_subcompress(
      "", "output", "file in which to store the compressed dataset", true, "",
      "filename");
  subcompress.add(output_subcompress);

  TCLAP::ValueArg<std::string> input_subcompress(
      "", "input", "file containing the dataset to be compressed", true, "",
      "filename");
  subcompress.add(input_subcompress);

  TCLAP::ValueArg<double> tolerance("", "tolerance", "absolute error tolerance",
                                    true, 0, "τ");
  subcompress.add(tolerance);

  // TODO: Change this to '--norm' or something.
  // Reading these in as `double`s for now.
  TCLAP::ValueArg<cli::SmoothnessParameter<double>> smoothness(
      "", "smoothness", "index of norm in which compression error is measured",
      true, 0, "s");
  subcompress.add(smoothness);

  TCLAP::ValueArg<cli::DataShape> shape(
      "", "shape",
      "shape of the data, given as an 'x'-delimited list of dimensions", true,
      {}, "list");
  subcompress.add(shape);

  std::vector<std::string> datatype_allowed = {"float", "double"};
  TCLAP::ValuesConstraint<std::string> datatype_constraint(datatype_allowed);
  TCLAP::ValueArg<std::string> datatype(
      "", "datatype", "floating-point format of the data", true,
      "floating point type", &datatype_constraint);
  subcompress.add(datatype);

  subcompress.addHelpArgument();

  cli::SubCmdLine subdecompress(MGARD_VERSION_STR,
                                "Decompress a dataset compressed using MGARD.");

  TCLAP::ValueArg<std::string> output_subdecompress(
      "", "output", "file in which to store the decompressed dataset", true, "",
      "filename");
  subdecompress.add(output_subdecompress);

  TCLAP::ValueArg<std::string> input_subdecompress(
      "", "input", "file containing the compressed dataset", true, "",
      "filename");
  subdecompress.add(input_subdecompress);

  subdecompress.addHelpArgument();

  std::map<std::string, cli::SubCmdLine *> subcommands = {
      {"compress", &subcompress}, {"decompress", &subdecompress}};
  cli::SuperCmdLine cmd(MGARD_VERSION_STR,
                        "MGARD is a compressor for scientific data.",
                        subcommands);

  cmd.parse(argc, argv);
  cli::SubCmdLine *const p = cmd.getSubcommand();
  if (p == &subcompress) {
    const cli::CompressionArguments arguments(datatype, shape, smoothness,
                                              tolerance, input_subcompress,
                                              output_subcompress);
    return cli::compress(arguments);
  } else if (p == &subdecompress) {
    const cli::DecompressionArguments arguments(input_subdecompress,
                                                output_subdecompress);
    return cli::decompress(arguments);
  } else {
    return 1;
  }

  return 0;
}
