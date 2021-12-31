import argparse
import typing

#C++ function argument.
class Argument:
    type_: str
    name: str

    def __init__(self, type_: str, name: str) -> None:
        self.type_ = type_
        self.name = name

    def __str__(self) -> str:
        return f'{self.type_} {self.name}'

#Name of header the generated file will implement.
HEADER: str = 'compress_internal.hpp'
#Name of containing namespace.
NAMESPACE: str = 'mgard'
#Name of the generated function.
F_NAME: str = 'decompress'
#Return type of the generated function.
F_RET_TYPE: str = 'std::unique_ptr<unsigned char const []>'
#Arguments of the generated function.
F_ARGUMENTS: typing.Tuple[Argument, ...] = (
    Argument('const pb::Header &', 'header'),
    Argument('const std::size_t', 'dimension'),
    Argument('void const * const', 'data'),
    Argument('const std::size_t', 'size'),
)
#Index of argument the generated function switches on.
_F_SW_ARG_INDEX: int = 1
#Argument the generated function switches on.
F_SWITCH_ARGUMENT: Argument = F_ARGUMENTS[_F_SW_ARG_INDEX]
#Name of the function the generated function delegates to.
G_NAME: str = 'decompress_N'
#Arguments of the delegate function.
G_ARGUMENTS: typing.Tuple[Argument, ...] = \
    F_ARGUMENTS[: _F_SW_ARG_INDEX] + F_ARGUMENTS[_F_SW_ARG_INDEX + 1 :]
#Exception message if an out-of-range dimension is encountered.
EXCEPTION_MESSAGE = 'unrecognized topology dimension'

parser = argparse.ArgumentParser(
    description=(
        'generate decompression function switching on topology dimension'
    ),
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

with open(args.outfile, 'w') as f:
    f.write(f'#include "{HEADER}"\n\n')
    f.write(f'namespace {NAMESPACE} {{\n')
    f.write(f'{F_RET_TYPE} {F_NAME}(')
    f.write(', '.join(map(str, F_ARGUMENTS)))
    f.write(') {\n')
    f.write(f'switch ({F_SWITCH_ARGUMENT.name}) {{\n')
    d: int
    for d in range(1, args.max_dim + 1) :
        f.write(f'case {d}: return {G_NAME}<{d}>(')
        f.write(', '.join(arg.name for arg in G_ARGUMENTS))
        f.write(');\n')
    f.write('default: throw std::runtime_error(')
    f.write(f'"{EXCEPTION_MESSAGE}"')
    f.write(');\n')
    f.write('}\n')
    f.write('}\n')
    f.write('}')
