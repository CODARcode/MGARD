#ifndef OUTPUT_HPP
#define OUTPUT_HPP
//!\file
//!\brief Outputters for the executable command line interface.

#include <ostream>
#include <sstream>
#include <string>

#include <tclap/ArgException.h>
#include <tclap/CmdLineInterface.h>
#include <tclap/CmdLineOutput.h>
#include <tclap/StdOutput.h>

#include "cli/cmdline.hpp"

namespace cli {

//! Printing function object making `TCLAP::StdOutput::spacePrint` accessible.
class SpacePrinter : private TCLAP::StdOutput {
public:
  //! Output a wrapped string to a stream.
  //!
  //!\param os Stream to which to output the wrapped string.
  //!\param s String to wrap and output.
  //!\param maxWidth Width to which to wrap the string.
  //!\param indentSpaces Number of spaces with which to indent the output.
  //!\param secondLineOffset Number of additional spaces with which to indent
  //! all lines of the output except the first.
  void operator()(std::ostream &os, const std::string &s, const int maxWidth,
                  const int indentSpaces, const int secondLineOffset) const;
};

//! Base class for sub- and supercommand outputs.
class BaseOutput : public TCLAP::CmdLineOutput {
public:
  //! Generate the usage message.
  virtual void usage(TCLAP::CmdLineInterface &c) override;

  //! Generate the failure message.
  virtual void failure(TCLAP::CmdLineInterface &c,
                       TCLAP::ArgException &e) override;

protected:
  //! Maximum width of wrapped output.
  int maxWidth{75};

  //! Number of spaces per indent level.
  int indentSpaces{2};

  //! Maximum additional indentation of output lines after the first.
  int maxSecondLineOffset{24};

  //! Output a wrapped string to a stream.
  //!
  //!\param os Stream to which to output the wrapped string.
  //!\param s String to wrap and output.
  //!\param indentLevel Number of times to indent the output.
  //!\param secondLineOffset Number of additional spaces with which to indent
  //! all lines of the output except the first.
  void spacePrint(std::ostream &os, const std::string &s, const int indentLevel,
                  const int secondLineOffset) const;

  //! Insert the usage command into a stream.
  //!
  //!\param [in] c Command in question.
  //!\param [in, out] stream Stream into which to insert the usage command.
  //!\param [out] secondLineOffset Offset determined by the program name.
  virtual void _usage_command(TCLAP::CmdLineInterface &c,
                              std::stringstream &stream,
                              int &secondLineOffset) const;

private:
  //! Printing function object.
  SpacePrinter spacePrinter;
};

//! Output class for subcommands.
class SubOutput : public BaseOutput {
public:
  //! Generate the version message.
  virtual void version(TCLAP::CmdLineInterface &c) override;
};

//! Output class for supercommands.
class SuperOutput : public BaseOutput {
public:
  //!\copydoc SubOutput::version(TCLAP::CmdLineInterface &)
  virtual void version(TCLAP::CmdLineInterface &c) override;

  virtual void usage(TCLAP::CmdLineInterface &c) override;

protected:
  virtual void _usage_command(TCLAP::CmdLineInterface &c,
                              std::stringstream &stream,
                              int &secondLineOffset) const override;
};

} // namespace cli

#endif
