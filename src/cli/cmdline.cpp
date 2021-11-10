#include "cli/cmdline.hpp"

#include <cassert>

#include <stdexcept>

#include <tclap/ArgException.h>

#include "cli/output.hpp"

namespace cli {

BaseCmdLine::BaseCmdLine(const std::string &version, const std::string &message,
                         TCLAP::CmdLineOutput *const output)
    : _version(version), _message(message), _output(output),
      // Adapted from `tclap/CmdLine.h`.
      _helpVisitor(this, &_output),
      _helpArg("h", "help", "Displays usage information and exits.", false,
               &_helpVisitor) {
  // This doesn't accomplish anything, but in general I'd like to use the
  // 'official' setter member functions.
  setOutput(_output);
}

TCLAP::ArgContainer &BaseCmdLine::add(TCLAP::Arg &a) { return add(&a); }

TCLAP::ArgContainer &BaseCmdLine::add(TCLAP::Arg *a) {
  a->addToList(_argList);
  return *this;
}

void BaseCmdLine::addToArgList(TCLAP::Arg *) {
  throw std::logic_error("not implemented");
}

void BaseCmdLine::xorAdd(TCLAP::Arg &, TCLAP::Arg &) {
  throw std::logic_error("not implemented");
}

void BaseCmdLine::xorAdd(const std::vector<TCLAP::Arg *> &) {
  throw std::logic_error("not implemented");
}

TCLAP::ArgContainer &BaseCmdLine::add(TCLAP::ArgGroup &) {
  throw std::logic_error("not implemented");
}

void BaseCmdLine::setOutput(TCLAP::CmdLineOutput *co) { _output = co; }

std::string BaseCmdLine::getVersion() const { return _version; }

std::string BaseCmdLine::getProgramName() const { return _programName; }

std::list<TCLAP::Arg *> BaseCmdLine::getArgList() const { return _argList; }

std::list<TCLAP::ArgGroup *> BaseCmdLine::getArgGroups() {
  throw std::logic_error("not implemented");
}

char BaseCmdLine::getDelimiter() const { return ' '; }

std::string BaseCmdLine::getMessage() const { return _message; }

void BaseCmdLine::reset() { throw std::logic_error("not implemented"); }

void BaseCmdLine::beginIgnoring() { throw std::logic_error("not implemented"); }

bool BaseCmdLine::ignoreRest() { throw std::logic_error("not implemented"); }

SubCmdLine::SubCmdLine(const std::string &version, const std::string &message)
    : BaseCmdLine(version, message, new SubOutput()) {
  setOutput(_output);
}

SubCmdLine::~SubCmdLine() { delete _output; }

void SubCmdLine::parse(int argc, const char *const *const argv) {
  std::vector<std::string> args(argv, argv + argc);
  return parse(args);
}

void SubCmdLine::parse(std::vector<std::string> &args) {
  try {
    if (args.empty()) {
      throw TCLAP::CmdLineParseException("Program name not provided.", "");
    }
    _programName = args.front();

    int N = args.size();
    for (int i = 1; i < N; ++i) {
      bool matched = false;
      for (TCLAP::Arg *p : _argList) {
        TCLAP::Arg &arg = *p;
        if (arg.processArg(&i, args)) {
          matched = true;
          break;
        }
      }
      if (not matched) {
        throw TCLAP::CmdLineParseException("Couldn't find match for argument",
                                           args[i]);
      }
    }
    for (TCLAP::Arg *p : _argList) {
      TCLAP::Arg &arg = *p;
      if (arg.isRequired() and not arg.isSet()) {
        throw TCLAP::CmdLineParseException("Required argument not provided",
                                           arg.getName());
      }
    }
  } catch (TCLAP::ArgException &e) {
    try {
      _output->failure(*this, e);
    } catch (TCLAP::ExitException &e) {
      std::exit(e.getExitStatus());
    }
  } catch (TCLAP::ExitException &e) {
    std::exit(e.getExitStatus());
  }
}

bool SubCmdLine::hasHelpAndVersion() const { return false; }

void SubCmdLine::addHelpArgument() { add(_helpArg); }

namespace {

std::vector<std::string>
subcommandNames(const std::map<std::string, SubCmdLine *> &subcommands) {
  std::vector<std::string> names;
  names.reserve(subcommands.size());
  for (const std::pair<const std::string, SubCmdLine *> &pair : subcommands) {
    names.push_back(pair.first);
  }
  return names;
}

} // namespace

SuperCmdLine::SuperCmdLine(const std::string &version,
                           const std::string &message,
                           std::map<std::string, SubCmdLine *> &subcommands)
    : BaseCmdLine(version, message, new SuperOutput()),
      _versionVisitor(this, &_output),
      _versionArg("", "version", "Displays version information and exits.",
                  false, &_versionVisitor),
      _subcommands(subcommands), _subcommandNames(subcommandNames(subcommands)),
      _subcommandConstraint(_subcommandNames),
      _subcommandArg("subcommand", "Operation to perform.", true,
                     "[no subcommand provided]", &_subcommandConstraint) {
  add(_helpArg);
  add(_versionArg);
  add(_subcommandArg);
}

SuperCmdLine::~SuperCmdLine() { delete _output; }

void SuperCmdLine::parse(int argc, const char *const *const argv) {
  std::vector<std::string> args(argv, argv + argc);
  return parse(args);
}

// Adapted from `TCLAP::CmdLine::parse`.
void SuperCmdLine::parse(std::vector<std::string> &args) {
  try {
    if (args.empty()) {
      throw TCLAP::CmdLineParseException("Program name not provided.", "");
    }
    _programName = args.front();

    if (args.size() < 2) {
      throw TCLAP::ExitException(0);
    }

    int i = 1;

    _helpArg.processArg(&i, args);
    // `_helpVisitor` should force an exit if the argument matches.
    _versionArg.processArg(&i, args);
    // `_versionVisitor` should force an exit if the argument matches.

    // `SwitchArg::processArg` doesn't change `i` (as far as I can tell), so
    // its value will still be `1`.
    if (not _subcommandArg.processArg(&i, args)) {
      throw TCLAP::CmdLineParseException("Couldn't find match for argument",
                                         args[i]);
    } else {
      std::map<std::string, SubCmdLine *>::const_iterator p =
          _subcommands.find(_subcommandArg.getValue());
      assert(p != _subcommands.end());
      assert(i == 1);
      args.at(1) = args.at(0) + " " + args.at(1);
      args.erase(args.begin());
      return p->second->parse(args);
    }
  } catch (TCLAP::ArgException &e) {
    try {
      _output->failure(*this, e);
    } catch (TCLAP::ExitException &e) {
      std::exit(e.getExitStatus());
    }
  } catch (TCLAP::ExitException &e) {
    std::exit(e.getExitStatus());
  }
}

bool SuperCmdLine::hasHelpAndVersion() const { return true; }

SubCmdLine *SuperCmdLine::getSubcommand() {
  return _subcommands[_subcommandArg.getValue()];
}

} // namespace cli
