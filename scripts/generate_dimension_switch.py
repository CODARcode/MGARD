import argparse
import typing

#Overengineered. Feel free to change.

Statements = typing.Tuple[typing.Any, ...]

class CaseStatement(typing.NamedTuple):
    label: str
    statements: Statements

    def __str__(self) -> str:
        return '\n'.join((
            f'case {self.label}: {{',
            *map(str, self.statements),
            '}'
        ))

class DefaultStatement(typing.NamedTuple):
    statements: Statements

    def __str__(self) -> str:
        return '\n'.join((
            'default: {',
            *map(str, self.statements),
            '}'
        ))

#Exception message if an out-of-range dimension is encountered.
EXCEPTION_MESSAGE = 'unrecognized topology dimension'
#Common default statement.
DEFAULT_STATEMENT = DefaultStatement(statements=(
    f'throw std::runtime_error("{EXCEPTION_MESSAGE}");',
))

class SwitchStatement(typing.NamedTuple):
    condition: str
    cases: typing.Tuple[CaseStatement, ...]
    default: DefaultStatement

    def __str__(self) -> str:
        return '\n'.join((
            f'switch ({self.condition}) {{',
            *map(str, self.cases),
            str(self.default),
            '}'
        ))

class FunctionArgument(typing.NamedTuple):
    type_: str
    name: str

    def __str__(self) -> str:
        return f'{self.type_} {self.name}'

class FunctionDefinition(typing.NamedTuple):
    return_type: str
    name: str
    arguments: typing.Tuple[FunctionArgument, ...]
    statements: Statements

    def __str__(self) -> str:
        return '\n'.join((
            f'{self.return_type} {self.name}(',
            ',\n'.join(map(str, self.arguments)),
            ') {',
            '\n'.join(map(str, self.statements)),
            '}'
        ))

class Implementation(typing.NamedTuple):
    header: str
    namespace: str
    definitions: typing.Tuple[FunctionDefinition, ...]

    def __str__(self) -> str:
        return '\n'.join((
            f'#include "{self.header}"',
            f'namespace {self.namespace} {{',
            *map(str, self.definitions),
            '}'
        ))

parser = argparse.ArgumentParser(
    description=(
        'generate functions switching on topology dimension'
    ),
)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    '--compress',
    action='store_true',
    help='generate compression function',
)
group.add_argument(
    '--decompress',
    action='store_true',
    help='generate decompression function',
)
parser.add_argument(
    'max_dim',
    type=int,
    help='maximum topology dimension',
)
parser.add_argument(
    'outfile',
    help='generated C++ source file',
)
args = parser.parse_args()

#Name of header the generated file will implement.
header: str
#Name of containing namespace.
namespace: str
#Name of the generated function.
f_name: str
#Return type of the generated function.
f_ret_type: str
#Arguments of the generated function.
f_arguments: typing.Tuple[FunctionArgument, ...]
#Argument the generated function switches on.
f_switch_argument: FunctionArgument
#Name of the function the generated function delegates to.
g_name: str
#Arguments of the delegate function.
g_arguments: typing.Tuple[FunctionArgument, ...]

if args.compress:
    header = 'cli_internal.hpp'
    namespace = 'cli'
    f_name = 'compress'
    f_ret_type = 'int'
    f_arguments = (
        FunctionArgument('const CompressionArguments &', 'arguments'),
        FunctionArgument('const std::size_t', 'dimension')
    )
    f_switch_argument = f_arguments[1]
    g_name = 'compress_N'
    g_arguments = f_arguments[: 1]
elif args.decompress:
    header = 'compress_internal.hpp'
    namespace = 'mgard'
    f_name = 'decompress'
    f_ret_type = 'MemoryBuffer<const unsigned char>'
    f_arguments = (
        FunctionArgument('const pb::Header &', 'header'),
        FunctionArgument('const std::size_t', 'dimension'),
        FunctionArgument('void const * const', 'data'),
        FunctionArgument('const std::size_t', 'size'),
    )
    f_switch_argument = f_arguments[1]
    g_name = 'decompress_N'
    g_arguments = f_arguments[: 1] + f_arguments[2 :]
else:
    raise RuntimeError

definition = FunctionDefinition(
    return_type=f_ret_type,
    name=f_name,
    arguments=f_arguments,
    statements=(
        SwitchStatement(
            condition=f_switch_argument.name,
            cases=tuple(
                CaseStatement(
                    label=str(d),
                    statements=(
                        ''.join((
                            f'return {g_name}<{d}>(',
                            ', '.join(arg.name for arg in g_arguments),
                            ');'
                        )),
                    )
                ) for d in range(1, args.max_dim + 1)
            ),
            default=DEFAULT_STATEMENT
        ),
    )
)
with open(args.outfile, 'w') as f:
    f.write(str(Implementation(
        header=header,
        namespace=namespace,
        definitions=(definition,)
    )))
