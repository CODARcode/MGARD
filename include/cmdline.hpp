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

//! Base class for sub- and supercommand parsers.
class BaseCmdLine : public TCLAP::CmdLineInterface {
public:
  //! Constructor.
  //!
  //!\param version Version of the command.
  //!\param message Message describing the command.
  //!\param output Outputter to use for the command.
  BaseCmdLine(const std::string &version, const std::string &message,
              TCLAP::CmdLineOutput *const output);

  //! Add an argument to the interface.
  virtual void add(TCLAP::Arg &a) override;

  //!\copydoc BaseCmdLine::add(TCLAP::Arg &)
  virtual void add(TCLAP::Arg *a) override;

  //! Add a mutually exclusive pair of arguments.
  virtual void xorAdd(TCLAP::Arg &a, TCLAP::Arg &b) override;

  //! Add a mutually exclusive list of arguments.
  virtual void xorAdd(std::vector<TCLAP::Arg *> &xors) override;

  //! Return associated output object.
  virtual TCLAP::CmdLineOutput *getOutput() override;

  //! Set associated output object.
  virtual void setOutput(TCLAP::CmdLineOutput *co) override;

  //! Return version string.
  virtual std::string &getVersion() override;

  //! Return program name.
  virtual std::string &getProgramName() override;

  //! Return list of arguments.
  virtual std::list<TCLAP::Arg *> &getArgList() override;

  //! Return handler for mutually exclusive arguments.
  virtual TCLAP::XorHandler &getXorHandler() override;

  //! Return the character used to separate argument names from values.
  virtual char getDelimiter() override;

  //! Return message describing program.
  virtual std::string &getMessage() override;

  //! Reset the `BaseCmdLine`.
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

  //! Parse the command arguments.
  virtual void parse(int argc, const char *const *const argv) override;

  //!\copydoc SubCmdLine::parse(int,const char * const * const)
  void parse(std::vector<std::string> &args);

  //! Return whether the parser has automatic help and version switches.
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

  //!\copydoc SubCmdLine::parse(int,const char * const * const)
  virtual void parse(int argc, const char *const *const argv) override;

  //!\copydoc SuperCmdLine::parse(int,const char * const * const)
  void parse(std::vector<std::string> &args);

  //!\copydoc SubCmdLine::hasHelpAndVersion()
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
