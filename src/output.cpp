#include "output.hpp"

#include <stdexcept>

void SpacePrinter::operator()(std::ostream &os, const std::string &s,
                              const int maxWidth, const int indentSpaces,
                              const int secondLineOffset) const {
  return TCLAP::StdOutput::spacePrint(os, s, maxWidth, indentSpaces,
                                      secondLineOffset);
}

void BaseOutput::spacePrint(std::ostream &os, const std::string &s,
                            const int indentLevel,
                            const int secondLineOffset) const {
  // Not worrying about extremely indented output or anything like that.
  return spacePrinter(os, s, maxWidth, indentLevel * indentSpaces,
                      std::min(secondLineOffset, maxSecondLineOffset));
}

void BaseOutput::usage(TCLAP::CmdLineInterface &c) {
  int indentLevel;
  int secondLineOffset;

  std::stringstream stream;
  stream << "usage: ";
  _usage_command(c, stream, secondLineOffset);
  indentLevel = 0;
  spacePrint(std::cout, stream.str(), 0, secondLineOffset);

  std::cout << std::endl;

  indentLevel = 0;
  secondLineOffset = 0;
  spacePrint(std::cout, "arguments and options:", indentLevel,
             secondLineOffset);

  secondLineOffset = indentSpaces;
  for (TCLAP::Arg const *const p : c.getArgList()) {
    indentLevel = 1;
    spacePrint(std::cout, p->longID(), indentLevel, secondLineOffset);
    indentLevel = 2;
    spacePrint(std::cout, p->getDescription(), indentLevel, secondLineOffset);
  }
}

void BaseOutput::_usage_command(TCLAP::CmdLineInterface &c,
                                std::stringstream &stream,
                                int &secondLineOffset) const {
  stream << c.getProgramName();
  secondLineOffset =
      std::min(static_cast<int>(stream.tellp()) + 1, maxSecondLineOffset);
  for (TCLAP::Arg const *const p : c.getArgList()) {
    stream << " " << p->shortID();
  }
}

void BaseOutput::failure(TCLAP::CmdLineInterface &c, TCLAP::ArgException &e) {
  const int indentLevel = 0;
  int secondLineOffset;

  std::stringstream stream;
  stream << "parse error: ";
  secondLineOffset = stream.tellp();
  stream << e.what();
  spacePrint(std::cerr, stream.str(), indentLevel, secondLineOffset);

  std::cerr << std::endl;

  secondLineOffset = indentSpaces;
  stream.str("");
  stream.clear();
  stream << "Run '" << c.getProgramName() << " --help'"
         << " for more information.";
  spacePrint(std::cerr, stream.str(), indentLevel, secondLineOffset);
  throw TCLAP::ExitException(1);
}

// `git commit --version` doesn't work.
void SubOutput::version(TCLAP::CmdLineInterface &) {
  throw std::runtime_error("not implemented");
}

void SuperOutput::usage(TCLAP::CmdLineInterface &c) {
  BaseOutput::usage(c);

  std::cout << std::endl;

  std::stringstream stream;
  // `git --help` uses '<command>'.
  stream << "Run '" << c.getProgramName() << " <command> --help'"
         << " for more information.";
  const int indentLevel = 0;
  const int secondLineOffset = indentSpaces;
  spacePrint(std::cout, stream.str(), indentLevel, secondLineOffset);
}

void SuperOutput::_usage_command(TCLAP::CmdLineInterface &c,
                                 std::stringstream &stream,
                                 int &secondLineOffset) const {
  BaseOutput::_usage_command(c, stream, secondLineOffset);
  // Mimicking the output of `git --help`.
  stream << " "
         << "[<args>]";
}

void SuperOutput::version(TCLAP::CmdLineInterface &c) {
  std::cout << c.getProgramName() << " version " << c.getVersion() << std::endl;
}
