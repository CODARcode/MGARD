#ifndef CMDLINE_HPP
#define CMDLINE_HPP
//!\file
//!\brief Parsers for the executable command line interface.

#include <list>
#include <map>
#include <string>
#include <vector>

#include <tclap/ArgException.h>
#include <tclap/CmdLineInterface.h>
#include <tclap/HelpVisitor.h>
#include <tclap/SwitchArg.h>
#include <tclap/UnlabeledValueArg.h>
#include <tclap/ValuesConstraint.h>
#include <tclap/VersionVisitor.h>

// Base class for sub- and supercommand parsers.
class BaseCmdLine : public TCLAP::CmdLineInterface {
public:
  //! Constructor.
  //!
  //!\param version Version of the command.
  //!\param message Message describing the command.
  //!\param output Outputter to use for the command.
  BaseCmdLine(const std::string &version, const std::string &message,
              TCLAP::CmdLineOutput *const output);

  virtual void add(TCLAP::Arg &a) override;

  virtual void add(TCLAP::Arg *a) override;

  virtual void xorAdd(TCLAP::Arg &a, TCLAP::Arg &b) override;

  virtual void xorAdd(std::vector<TCLAP::Arg *> &xors) override;

  virtual TCLAP::CmdLineOutput *getOutput() override;

  virtual void setOutput(TCLAP::CmdLineOutput *co) override;

  virtual std::string &getVersion() override;

  virtual std::string &getProgramName() override;

  virtual std::list<TCLAP::Arg *> &getArgList() override;

  virtual TCLAP::XorHandler &getXorHandler() override;

  virtual char getDelimiter() override;

  virtual std::string &getMessage() override;

  virtual void reset() override;

protected:
  //! Name used to call the program.
  std::string _programName{"[program name not set]"};

  //! Version of the command.
  std::string _version{"[version number not set]"};

  //! Message describing the command.
  std::string _message{"[message not set]"};

  //! Outputter of the command.
  TCLAP::CmdLineOutput *_output;

  //! Visitor for the '--help' switch.
  TCLAP::HelpVisitor _helpVisitor;

  //! Argument for the '--help' switch.
  TCLAP::SwitchArg _helpArg;

  //! Arguments of the command.
  std::list<TCLAP::Arg *> _argList;
};

//! Subcommand parser.
class SubCmdLine : public BaseCmdLine {
public:
  //! Constructor.
  //!
  //!\param version Version of the subcommand.
  //!\param message Message describing the subcommand.
  SubCmdLine(const std::string &version, const std::string &message);

  //! Destructor.
  ~SubCmdLine();

  virtual void parse(int argc, const char *const *const argv) override;

  //! Parse the subcommand arguments.
  //!
  //!\param arg Subcommand arguments.
  void parse(std::vector<std::string> &args);

  virtual bool hasHelpAndVersion() override;

  //! Add the help argument to the argument list.
  //!
  //! This member function allows us to put the help argument at the front of
  //! the argument list.
  void addHelpArgument();
};

//! Supercommand parser.
class SuperCmdLine : public BaseCmdLine {
public:
  //! Constructor.
  //!
  //!\param version Version of the supercommand.
  //!\param message Message describing the supercommand.
  //!\param subcommands Associated subcommands.
  SuperCmdLine(const std::string &version, const std::string &message,
               std::map<std::string, SubCmdLine *> &subcommands);

  //! Destructor.
  ~SuperCmdLine();

  virtual void parse(int argc, const char *const *const argv) override;

  //! Parse the supercommand arguments.
  //!
  //!\param arg Supercommand arguments.
  void parse(std::vector<std::string> &args);

  virtual bool hasHelpAndVersion() override;

  //! Report the subcommand specified by the arguments.
  //!
  //! This should only be called after `parse`.
  SubCmdLine *getSubcommand();

private:
  //! Visitor for the '--version' switch.
  TCLAP::VersionVisitor _versionVisitor;

  //! Argument for the '--version' switch.
  TCLAP::SwitchArg _versionArg;

  //! Associated subcommands.
  std::map<std::string, SubCmdLine *> _subcommands;

  //! Names of the subcommands.
  std::vector<std::string> _subcommandNames;

  //! Constraint for the subcommand name.
  TCLAP::ValuesConstraint<std::string> _subcommandConstraint;

  //! Argument for the subcommand.
  TCLAP::UnlabeledValueArg<std::string> _subcommandArg;
};

#endif
